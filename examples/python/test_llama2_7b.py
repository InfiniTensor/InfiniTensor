from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM
from tqdm import tqdm
import argparse
import torch
import onnx
import onnx_graphsurgeon as gs
import os
import numpy as np
from pyinfinitensor.onnx import OnnxStub, backend
import time
import nvtx
from mpi4py import MPI

parser = argparse.ArgumentParser(description='')
parser.add_argument('--batchsize', dest='batchsize', type=int, default=1)
parser.add_argument('--layer', dest='n_layers', type=int, default=32)
parser.add_argument("--num_nodes", dest='num_nodes',
                    type=int, default=1, help="number of nodes")
parser.add_argument("--nproc_per_node", dest="nproc_per_node",
                    type=int, default=1, help="number of processes per node")
parser.add_argument("--world_size", dest="world_size",
                    type=int, default=1, help="")
parser.add_argument("--n_max_length", dest="n_max_length",
                    type=int, default=1024, help="")
parser.add_argument("--vocab_size", dest="vocab_size",
                    type=int, default=32000, help="vocabulary size")
parser.add_argument("--hidden_size", dest="hidden_size",
                    type=int, default=4096)
parser.add_argument("--head_size", dest="head_size",
                    type=int, default=32)
parser.add_argument("--head_dim", dest="head_dim",
                    type=int, default=128)
parser.add_argument('--rank', dest='rank', type=int, default=0)
parser.add_argument('--no_cudagraph', action='store_true')
parser.add_argument('--fp16', action='store_true')
parser.add_argument('--is_1st_graph', action='store_true')
parser.add_argument('--speedup', action='store_true')
args = parser.parse_args()

comm = MPI.COMM_WORLD
args.rank = comm.Get_rank()
args.nproc_per_node = comm.Get_size()
args.world_size = args.num_nodes * args.nproc_per_node

PRETRAINED_LLAMA_PATH = "/data0/shared/data/public/opensource_models/meta-llama/Llama-2-7b-hf/"
ONNX_MODEL_PATH = "/data3/shared/xnsong/llama2/" + ("1st" if args.is_1st_graph else "2nd")
ONNX_MODEL_ORIGIN_PATH = ONNX_MODEL_PATH + "/origin/llama2_origin_bs{}_layer{}.onnx".format(
    args.batchsize, args.n_layers)
ONNX_MODEL_SIM_PATH = ONNX_MODEL_PATH + "/sim/llama2_sim_bs{}_layer{}.onnx".format(
    args.batchsize, args.n_layers)
ONNX_MODEL_FUSION_PATH = ONNX_MODEL_PATH + "/fusion/llama2_fusion_bs{}_layer{}.onnx".format(
    args.batchsize, args.n_layers)
ONNX_MODEL_SPECIAL_PATH = ONNX_MODEL_PATH + "/special/llama2_special_bs{}_layer{}.onnx".format(
    args.batchsize, args.n_layers)
ONNX_MODEL_FP16_PATH = ONNX_MODEL_PATH + "/fp16/llama2_fp16_bs{}_layer{}.onnx".format(
    args.batchsize, args.n_layers)
ONNX_MODEL_DIST_PATH = ONNX_MODEL_PATH + "/dist/llama2_dist_bs{}_layer{}_fp{}_worldsize{}_rank{}.onnx".format(
    args.batchsize, args.n_layers, 16 if args.fp16 else 32, args.world_size, args.rank)

def parallel_model(onnx_model, world_size, rank):
    graph = gs.import_onnx(onnx_model)
    tmap = graph.tensors()
    
    for i in range(args.n_layers):
        tmap[graph.inputs[2+i*2].name].shape[1] = tmap[graph.inputs[2+i*2].name].shape[1]//world_size
        tmap[graph.inputs[3+i*2].name].shape[1] = tmap[graph.inputs[3+i*2].name].shape[1]//world_size
        for node in graph.nodes:
            if node.name == "/model/layers." + str(i) + "/self_attn/q_proj/MatMul":
                node.inputs[1].values = np.hsplit(node.inputs[1].values, world_size)[rank]
            elif node.name == "/model/layers." + str(i) + "/self_attn/k_proj/MatMul":
                node.inputs[1].values = np.hsplit(node.inputs[1].values, world_size)[rank]
            elif node.name == "/model/layers." + str(i) + "/self_attn/v_proj/MatMul":
                node.inputs[1].values = np.hsplit(node.inputs[1].values, world_size)[rank]
            elif node.name == "/model/layers." + str(i) + "/self_attn/o_proj/MatMul":
                node.inputs[1].values = np.vsplit(node.inputs[1].values, world_size)[rank]
                reduce_sum_output = gs.Variable("reduce_sum_output_" + str(i) + "_0", 
                                                dtype=np.float32)
                reduce_sum = gs.Node(op="ReduceSum", name="reduce_sum_"+str(i)+"_0",
                                     inputs=node.outputs, outputs=[reduce_sum_output], 
                                     attrs={"noop_with_empty_axes":1, "communicator":0})
                graph.nodes.append(reduce_sum)
                next_node = node.outputs[0].outputs[0]
                next_node.inputs[1] = reduce_sum_output
            elif node.name == "/model/layers." + str(i) + "/self_attn/Reshape_0" or \
                 node.name == "/model/layers." + str(i) + "/self_attn/Reshape_1":
                node.inputs[1].values = np.array(
                    [1, 1,
                     args.head_size//world_size,
                     args.hidden_size//args.head_size])
            elif node.name == "/model/layers." + str(i) + "/self_attn/Reshape_2":
                 node.inputs[1] = gs.Constant(name="/model/layers."+str(i)+"/self_attn/vreshape_input",
                                             values=np.array(
                                                 [1, 1,
                                                  args.head_size//world_size,
                                                  args.hidden_size//args.head_size]))
            elif node.name == "/model/layers." + str(i) + "/self_attn/Reshape_3":
                node.inputs[1] = gs.Constant(name="/model/layers." + str(i) + "/self_attn/Reshape_3_shape",
                                             values=np.array(
                                                 [1, 1, args.hidden_size//world_size]))

            elif node.name == "/model/layers." + str(i) + "/mlp/up_proj/MatMul":
                node.inputs[1].values = np.hsplit(node.inputs[1].values, world_size)[rank]
            elif node.name == "/model/layers." + str(i) + "/mlp/gate_proj/MatMul":
                node.inputs[1].values = np.hsplit(node.inputs[1].values, world_size)[rank]
            elif node.name == "/model/layers." + str(i) + "/mlp/down_proj/MatMul":
                node.inputs[1].values = np.vsplit(node.inputs[1].values, world_size)[rank]
                reduce_sum_output_1 = gs.Variable("reduce_sum_output_" + str(i) + "_1", 
                                                  dtype=np.float32)
                reduce_sum_1 = gs.Node(op="ReduceSum", inputs=node.outputs, outputs=[reduce_sum_output_1],
                                     attrs={"noop_with_empty_axes":1, "communicator":0})
                graph.nodes.append(reduce_sum_1)
                next_node = node.outputs[0].outputs[0]
                next_node.inputs[1] = reduce_sum_output_1

    # new_out_1 = tmap["/model/layers.0/mlp/down_proj/MatMul_output_0"] #reduce_sum_output
    # new_out_1.dtype = np.float32
    # new_out_1.shape = [1,1,4096]
    # graph.outputs.append(new_out_1)
    graph.cleanup(True).toposort()
    return gs.export_onnx(graph)

def simplify(onnx_model):
    graph = gs.import_onnx(onnx_model)
    for node in graph.nodes:
        if node.op == "Cast":
            inp_node = node.i()
            inp_node.outputs = node.outputs
            node.outputs.clear()
    
    for i in range(args.n_layers):
        nodename = "/model/layers." + str(i) + "/self_attn/Add_2"
        node = [node for node in graph.nodes if node.name == nodename][0]
        inp_node = node.i()
        inp_node.outputs = node.outputs
        node.outputs.clear()

    graph.cleanup().toposort()
    return gs.export_onnx(graph)

@gs.Graph.register()
def replace_with_RMSNorm(self, inputs, outputs):
    inputs[0].outputs.pop(0)
    inputs[0].outputs.pop(0)

    for out in outputs:
        out.inputs.clear()
    return self.layer(op="RMSNorm", inputs=inputs, outputs=outputs, name="rmsnorm")

@gs.Graph.register()
def replace_with_silu(self, inputs, outputs):
    for inp in inputs:
        inp.outputs.clear()
    for out in outputs:
        out.inputs.clear()
    return self.layer(op="Silu", inputs=inputs, outputs=outputs, name="silu")

@gs.Graph.register()
def replace_with_RoPE(self, a, b):
    return self.layer(op="RoPE", inputs=a, outputs=b, name="rope")

@gs.Graph.register()
def replace_with_attention(self, inputs, outputs, inputs_added, outputs_removed):
    for inp in inputs:
        inp.outputs.clear()
    for out in outputs:
        out.inputs.clear()
    for inp in inputs_added:
        inputs.append(inp)
    for out in outputs_removed:
        out.inputs.clear()
    return self.layer(op="AttentionKVCache", inputs=inputs, outputs=outputs, name="attention")

def fusion(model):
    graph = gs.import_onnx(model)
    tmap = graph.tensors()

    tmap["onnx::Reshape_1"].outputs.clear()

    inputs = [tmap["/model/layers.0/input_layernorm/Cast_output_0"], tmap["model.layers.0.input_layernorm.weight"]]
    rmsnorm_outputs = [tmap["/model/layers.0/input_layernorm/Mul_1_output_0"]]
    graph.replace_with_RMSNorm(inputs, rmsnorm_outputs) 

    for i in range(args.n_layers):
        # rotary embedding op
        tmap["/model/layers." + str(i) + "/self_attn/Add_output_0"].inputs.clear()
        tmap["/model/layers." + str(i) + "/self_attn/Add_1_output_0"].inputs.clear()
        attn_qreshape_input = gs.Constant(name="/model/layers." + str(i) + "/self_attn/qreshape_input", 
                                          values=np.array([1,1,args.head_size,args.hidden_size//args.head_size]))
        attn_kreshape_input = gs.Constant(name="/model/layers." + str(i) + "/self_attn/kreshape_input", 
                                          values=np.array([1,1,args.head_size,args.hidden_size//args.head_size]))
        attn_qrope_output   = gs.Variable(name="/model/layers." + str(i) + "/self_attn/qrope_output")
        attn_krope_output    = gs.Variable(name="/model/layers." + str(i) + "/self_attn/krope_output")
        attn_qreshape_output = gs.Variable(name="/model/layers." + str(i) + "/self_attn/qreshape_output") 
        attn_kreshape_output = gs.Variable(name="/model/layers." + str(i) + "/self_attn/kreshape_output") 

        attn_qreshape = gs.Node(op="Reshape", name = "/model/layers." + str(i) + "/self_attn/Reshape_0", inputs=[attn_qrope_output, attn_qreshape_input], outputs=[attn_qreshape_output])
        attn_kreshape = gs.Node(op="Reshape", name = "/model/layers." + str(i) + "/self_attn/Reshape_1", inputs=[attn_krope_output, attn_kreshape_input], outputs=[attn_kreshape_output])
        attn_qtrans = gs.Node(op="Transpose", attrs={"perm":np.array([0,2,1,3])}, inputs=[attn_qreshape_output], 
                              outputs=[tmap["/model/layers." + str(i) + "/self_attn/Add_output_0"]])
        attn_ktrans = gs.Node(op="Transpose", attrs={"perm":np.array([0,2,1,3])}, inputs=[attn_kreshape_output], 
                              outputs=[tmap["/model/layers." + str(i) + "/self_attn/Add_1_output_0"]])

        graph.nodes.append(attn_qreshape)
        graph.nodes.append(attn_kreshape)
        graph.nodes.append(attn_qtrans)
        graph.nodes.append(attn_ktrans)
        inputs = [tmap["onnx::Reshape_1"], tmap["/model/layers." + str(i) + "/self_attn/q_proj/MatMul_output_0"]]
        graph.replace_with_RoPE(inputs, [attn_qrope_output])
        inputs = [tmap["onnx::Reshape_1"], tmap["/model/layers." + str(i) + "/self_attn/k_proj/MatMul_output_0"]]
        graph.replace_with_RoPE(inputs, [attn_krope_output])

        # rms-norm op
        inputs = [tmap["/model/layers." + str(i) + "/post_attention_layernorm/Cast_output_0"], \
                  tmap["model.layers." + str(i) + ".post_attention_layernorm.weight"]]
        outputs = [tmap["/model/layers." + str(i) + "/post_attention_layernorm/Mul_1_output_0"]]
        graph.replace_with_RMSNorm(inputs, outputs)
        inputs = [tmap["/model/layers." + str(i+1) + "/input_layernorm/Cast_output_0"] if i != args.n_layers-1 else \
                  tmap["/model/norm/Cast_output_0"], \
                  tmap["model.layers." + str(i+1) + ".input_layernorm.weight"] if i != args.n_layers-1 else \
                  tmap["model.norm.weight"]]
        outputs = [tmap["/model/layers."+ str(i+1) + "/input_layernorm/Mul_1_output_0"]] if i != args.n_layers-1 else \
                  [tmap["/model/norm/Mul_1_output_0"]]
        graph.replace_with_RMSNorm(inputs, outputs)

        # silu op
        inputs = [tmap["/model/layers." + str(i) + "/mlp/gate_proj/MatMul_output_0"]]
        outputs = [tmap["/model/layers." + str(i) + "/mlp/act_fn/Mul_output_0"]]
        graph.replace_with_silu(inputs, outputs)

        inputs = [
            tmap["onnx::Concat_" + str((i+1)*2)],
            tmap["onnx::Concat_" + str((i+1)*2+1)],
            tmap["/model/layers." + str(i) + "/self_attn/Add_output_0"],
            tmap["/model/layers." + str(i) + "/self_attn/Add_1_output_0"],
            tmap["/model/layers." + str(i) + "/self_attn/Transpose_2_output_0"]]
        outputs = [
            tmap["/model/layers." + str(i) + "/self_attn/MatMul_1_output_0"],]

        inputs_added = [graph.inputs[1]]
        outputs_removed = []
        graph.replace_with_attention(
            inputs, outputs, inputs_added, outputs_removed)
    
    graph.outputs = [tmap[graph.outputs[0].name]]
    graph.cleanup(True).toposort()

    return gs.export_onnx(graph)

def special_pass(model):
    graph = gs.import_onnx(model)
    tmap = graph.tensors()
    for node in graph.nodes:
        if node.op == "Transpose" or node.op == "Reshape":
            inp_node = node.i()
            inp_node.outputs = node.outputs
            node.outputs.clear()
    graph.cleanup(True).toposort()
    return gs.export_onnx(graph)

def convert_to_fp16(model):
    graph = gs.import_onnx(model)
    
    for node in graph.nodes:
        if node.op == "Gather" and node.name == "/model/embed_tokens/Gather":
            node.inputs[0].values = np.float16(node.inputs[0].values)
        
        if node.op == "RMSNorm":
            node.inputs[1].values = np.float16(node.inputs[1].values)

        if node.op == "MatMul":
            node.inputs[1].values = np.float16(node.inputs[1].values)
            if node.name == "/lm_head/MatMul":
                cast_1_out = gs.Variable(node.name+"_cast_out_output_0", dtype=np.float32, shape=node.outputs[0].shape)
                cast_1 = gs.Node(op="Cast", inputs=[node.outputs[0]], outputs=[cast_1_out])
                cast_1.attrs["to"] = np.float32
                cast_1.name = node.name+"_cast_out_0"
                graph.nodes.append(cast_1)
                graph.outputs[0] = cast_1_out
                node.outputs[0].dtype = np.float16

    graph.cleanup(True).toposort()
    return gs.export_onnx(graph)

def export_onnx(model: AutoModelForCausalLM):
    if not os.path.exists(ONNX_MODEL_ORIGIN_PATH):
        print("exporting origin onnx model...")
        with torch.no_grad():
            param = torch.zeros(
                (args.batchsize, model.config.max_position_embeddings-1), dtype=torch.long)
            logits = model(param, past_key_values=None)
            
            if not args.is_1st_graph:
                param_kvcache = torch.zeros((args.batchsize, 1), dtype=torch.long)
                torch.onnx.export(model, (param_kvcache, {"past_key_values": logits.past_key_values,
                                                          "position_ids": param_kvcache}), \
                                ONNX_MODEL_ORIGIN_PATH, verbose=False,
                                do_constant_folding=True,)
            else:
                position_ids = torch.tile(torch.arange(0, model.config.max_position_embeddings-1), (args.batchsize, 1))
                attention_mask = torch.ones((args.batchsize, model.config.max_position_embeddings-1), dtype=torch.bool)
                torch.onnx.export(model, (param, {"attention_mask": attention_mask,
                                                  "position_ids": position_ids}),\
                                ONNX_MODEL_ORIGIN_PATH, verbose=False,
                                do_constant_folding=True,)
        print("export origin onnx finished.")

    if not args.is_1st_graph and not os.path.exists(ONNX_MODEL_SIM_PATH):
        print("exporting sim onnx model...")
        onnx_model = onnx.load(ONNX_MODEL_ORIGIN_PATH)
        onnx_model = simplify(onnx_model)
        onnx.save(onnx_model, ONNX_MODEL_SIM_PATH, save_as_external_data=True, \
                    location="llama2_sim_bs{}_layer{}.pb".format(args.batchsize, args.n_layers))
        print("exporting sim onnx model finished.")

    if not args.is_1st_graph and not os.path.exists(ONNX_MODEL_FUSION_PATH):
        print("exporting fusion onnx model...")
        onnx_model = onnx.load(ONNX_MODEL_SIM_PATH)
        onnx_model = fusion(onnx_model)
        onnx.save(onnx_model, ONNX_MODEL_FUSION_PATH, save_as_external_data=True, \
                    location="llama2_fusion_bs{}_layer{}.pb".format(args.batchsize, args.n_layers))
        print("exporting fusion onnx model finished.")
    
    if not args.is_1st_graph and not os.path.exists(ONNX_MODEL_SPECIAL_PATH):
        print("exporting special onnx model...")
        onnx_model = onnx.load(ONNX_MODEL_FUSION_PATH)
        onnx_model = special_pass(onnx_model)
        onnx.save(onnx_model, ONNX_MODEL_SPECIAL_PATH, save_as_external_data=True, \
                    location="llama2_special_bs{}_layer{}.pb".format(args.batchsize, args.n_layers))
        print("exporting special onnx model finished.")

    if not args.is_1st_graph and args.fp16 and not os.path.exists(ONNX_MODEL_FP16_PATH):
        print("exporting fp16 onnx model...")
        onnx_model = onnx.load(ONNX_MODEL_SPECIAL_PATH)
        onnx_model = convert_to_fp16(onnx_model)
        onnx.save(onnx_model, ONNX_MODEL_FP16_PATH, save_as_external_data=True, \
                    location="llama2_fp16_bs{}_layer{}.pb".format(args.batchsize, args.n_layers))
        print("exporting fp16 onnx model finished.")

    print("world_size =", args.world_size)
    if not args.is_1st_graph and args.world_size > 1 and not os.path.exists(ONNX_MODEL_DIST_PATH):
        print("exporting dist onnx model...")    
        onnx_model = onnx.load(ONNX_MODEL_FP16_PATH) if args.fp16 else onnx.load(ONNX_MODEL_SPECIAL_PATH)
        onnx_model = parallel_model(onnx_model, args.world_size, args.rank)
        onnx.save(onnx_model, ONNX_MODEL_DIST_PATH, save_as_external_data=True, \
            location="llama2_dist_bs{}_layer{}_fp{}_worldsize{}_rank{}.pb".format(
                args.batchsize, args.n_layers, 
                16 if args.fp16 else 32, args.world_size, args.rank))
        print("exporting dist onnx model finished.")

def get_it_logit(onnx_model, input_ids):
    # initialization
    runtime = backend.CudaRuntime(args.rank)
    runtime.init_comm(
        "dist",
        args.world_size,
        args.rank,
    )
    print("[{}] comm init.".format(args.rank))
    stub = OnnxStub(onnx_model, runtime)
    print("[{}] stub init.".format(args.rank))

    # warm up
    for i in range(10):
        if args.no_cudagraph:
            stub.run()
        else:
            stub.run_with_cudagraph()
    print("[{}] stub warmup.".format(args.rank))

    logits = np.zeros((args.batchsize, args.n_max_length, args.vocab_size), dtype=np.float32)
    output_ids = np.zeros((args.batchsize, args.n_max_length), dtype=np.int64)
    avg_inference_time = 0
    t0 = time.time()
    for i in tqdm(range(0, args.n_max_length)):
        with nvtx.annotate("seq_length = {}".format(i), color="red"):
            assert input_ids.shape[0] == args.batchsize
            input_id = input_ids[:, i] if i < input_ids.shape[1] else output_ids[:, i-1]
            position_id = i*np.ones((args.batchsize, 1), dtype=np.int32)

            # copyin input
            with nvtx.annotate("[it] copyin", color="blue"):
                (list(stub.inputs.items()))[0][1].copyin_int64(
                    input_id.reshape(-1).tolist())
                (list(stub.inputs.items()))[1][1].copyin_int64(
                    position_id.reshape(-1).tolist())
                
            # run
            t10 = time.time()
            with nvtx.annotate("[it] run", color="green"):
                if args.no_cudagraph:
                    stub.run()
                else:
                    stub.run_with_cudagraph()
            t11 = time.time()
            avg_inference_time += (t11 - t10)
                    
            # copyout output
            if not args.speedup:
                with nvtx.annotate("[it] copyout", color="blue"):
                    logits[:,i, :] = np.array((list(stub.outputs.items()))[0][1].copyout_float()).reshape(args.batchsize, -1)
                    output_ids[:, i] = np.argmax(logits[:, i, :], -1).astype(np.int64)
                    

    t1 = time.time()
    if args.rank == 0:
        result = "[it] e2e: {} gpus, {} layers, e2e time: {:.2f}s, average inference time: {:.2f}ms"\
                .format(args.num_nodes * args.nproc_per_node, args.n_layers, t1-t0, \
                        avg_inference_time*1000/args.n_max_length)
        print(result)
    del stub
    return output_ids

if __name__ == "__main__":
    torch_model = LlamaForCausalLM.from_pretrained(
        PRETRAINED_LLAMA_PATH, num_hidden_layers=int(args.n_layers)).eval()
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_LLAMA_PATH)
    #prompt = "Hey, are you conscious? Can you talk to me?"
    #prompt = "What is PTX?"
    #prompt = "Tell me a joke."
    #prompt = "What are the key principles of smart investing?"
    prompt = "What is DeepSpeed?"
    prompts=[prompt]*args.batchsize
    inputs = tokenizer(prompts, return_tensors="pt")
    
    input_ids = inputs.input_ids
    print("prompt ids =", input_ids)

    ##########################################################
    # inference with InfiniTensor
    ##########################################################
    print("exporting onnx...")
    export_onnx(torch_model)
    print("exporting onnx finished.")
    
    onnx_to_run_path = ONNX_MODEL_DIST_PATH if args.world_size > 1 else \
                       (ONNX_MODEL_FP16_PATH if args.fp16 else ONNX_MODEL_SPECIAL_PATH)
    print("loading onnx", onnx_to_run_path, "...")
    onnx_model = onnx.load(onnx_to_run_path)
    print("loading onnx finished.")
    output_ids_it = get_it_logit(onnx_model, input_ids)
    it_output_text = tokenizer.batch_decode(output_ids_it[:, input_ids.shape[-1]:output_ids_it.shape[-1]])
    if args.rank == 0:
        for i in range(args.batchsize):
            print("prompt: ", prompts[i])
            print("answer: [it]", it_output_text[i])

    ##########################################################
    # validation with pytorch
    ##########################################################
    """
    generate_ids = torch_model.generate(inputs.input_ids, max_length=args.n_max_length)#, num_beams=4, do_sample=True)
    outputs = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    """
    if not args.speedup and not args.is_1st_graph:
        kvcache_torch = None
        output_ids_pt = torch.zeros(args.batchsize, args.n_max_length).int() # + input_ids.shape[-1] - 1).int()
        if args.fp16:
            torch_model = torch_model.half()
        
        torch_model = torch_model.cuda()
        # print(torch.cuda.memory_summary())

        avg_inference_time = 0
        with torch.no_grad():
            t0 = time.time()
            for i in range(args.n_max_length):
                input_id = input_ids[:,i] if i < input_ids.shape[1] else out_token
                input_id = input_id.view(args.batchsize,1).cuda()
                t00 = time.time()
                outputs = torch_model(input_id, past_key_values=kvcache_torch)
                t01 = time.time()
                avg_inference_time += (t01-t00)

                logits = outputs['logits']
                kvcache_torch = outputs['past_key_values']
                out_token = torch.argmax(logits, dim=-1)
                output_ids_pt[:, i:i+1] = out_token
            t1 = time.time()
        avg_inference_time /= args.n_max_length
        result = "[pt] e2e time: {:.2f}s, average inference time: {:.2f}ms"\
                .format(t1-t0, avg_inference_time*1000)

        if args.rank == 0:
            print(result)
            pt_output_text = tokenizer.batch_decode(output_ids_pt[:,input_ids.shape[-1]:args.n_max_length])
            for i in range(args.batchsize):
                print("[pt]", args.rank, pt_output_text[i])
            
            if not args.is_1st_graph:
                assert(output_ids_it.shape[-1] == args.n_max_length)
                np.testing.assert_equal(output_ids_pt[:, input_ids.shape[-1]:args.n_max_length], output_ids_it[:,input_ids.shape[-1]:args.n_max_length])
