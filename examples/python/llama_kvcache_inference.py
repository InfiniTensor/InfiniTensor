import os
from pyinfinitensor.onnx import OnnxStub, backend
import numpy as np
import onnx
import torch
from transformers import LlamaModel, LlamaForCausalLM
from tqdm import tqdm
import onnx_graphsurgeon as gs
from onnxsim import simplify
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('--batchsize', dest='batchsize', type=int, default=1)
parser.add_argument('--layer', dest='n_layers', type=int, default=2)
parser.add_argument('--iter', dest='n_iter', type=int, default=1)
parser.add_argument('--n_max_length', dest='n_max_length', type=int, default=1024)
parser.add_argument('--pretrained_llama_path', dest='pretrained_llama_path', type=str, 
                    default="/data0/shared/data/public/opensource_models/meta-llama/Llama-2-7b-hf/")
parser.add_argument('--onnx_model_path', dest='onnx_model_path', type=str, 
                    default="/data1/shared/llama")
args = parser.parse_args()

ONNX_MODEL_PATH = "{}/llama_bs{}_layer{}.onnx".format(args.onnx_model_path, args.batchsize, args.n_layers)
ONNX_WEIGHT_PATH = "./llama_bs{}_layer{}.pb".format(args.batchsize, args.n_layers)

def export_onnx(model: LlamaModel, ONNX_MODEL_PATH):
    param = torch.zeros(
        (args.batchsize, 1024), dtype=torch.long)
    logits = model(param, past_key_values=None)
    param_kvcache = torch.zeros((args.batchsize, 1), dtype=torch.long)

    torch.onnx.export(model, (param_kvcache, {"past_key_values": logits.past_key_values,
                                              "position_ids": param_kvcache}), ONNX_MODEL_PATH, verbose=False,
                      do_constant_folding=True,)
    onnx_model = onnx.load(ONNX_MODEL_PATH)
    print("simplifing onnx model")
    onnx_model, check = simplify(onnx_model, skipped_optimizers=[
                                 'eliminate_duplicate_initializer'])
    assert check
    
    onnx.save(onnx_model, ONNX_MODEL_PATH, save_as_external_data=True, location=ONNX_WEIGHT_PATH)
    print("simlifing finished.")


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
    return self.layer(op="AttentionKVCache", inputs=inputs, outputs=outputs)


def replace_onnx_with_attention_op():
    graph = gs.import_onnx(
        onnx.load(ONNX_MODEL_PATH))
    tmap = graph.tensors()
    for i in range(args.n_layers):
        inputs = [
            tmap["onnx::Concat_" + str((i+1)*2)],
            tmap["onnx::Concat_" + str((i+1)*2+1)],
            tmap["/model/layers." + str(i) + "/self_attn/Add_output_0"],
            tmap["/model/layers." + str(i) + "/self_attn/Add_1_output_0"],
            tmap["/model/layers." + str(i) + "/self_attn/Transpose_2_output_0"]]
        outputs = [
            tmap["/model/layers." + str(i) + "/self_attn/MatMul_1_output_0"]]

        inputs_added = [graph.inputs[1]]
        outputs_removed = []

        graph.replace_with_attention(
            inputs, outputs, inputs_added, outputs_removed)
        
    graph.outputs = [tmap[graph.outputs[0].name]]
    graph.cleanup(True).toposort()
    onnx.save(gs.export_onnx(graph), ONNX_MODEL_PATH, save_as_external_data=True)


if __name__ == "__main__":
    kvcache_torch = None
    torch_model = LlamaForCausalLM.from_pretrained(
        args.pretrained_llama_path, num_hidden_layers=int(args.n_layers)).eval()
    
    n_heads = torch_model.config.num_attention_heads
    n_dims = torch_model.config.hidden_size // n_heads
    
    if not os.path.exists(ONNX_MODEL_PATH):
        print("exporting onnx graph")
        export_onnx(torch_model, ONNX_MODEL_PATH)
        replace_onnx_with_attention_op()
    else:
        print("will use exsiting onnx graph")

    onnx_model = onnx.load(ONNX_MODEL_PATH)
    stub = OnnxStub(onnx_model, backend.cuda_runtime())

    count_wrong = 0
    for i in tqdm(range(0, args.n_max_length)):
        query = np.random.randint(
            torch_model.config.vocab_size, size=(args.batchsize, 1), dtype=np.int32)
        position_id = i*np.ones((args.batchsize, 1), dtype=np.int32)

        ####################################
        # pytorch
        ####################################
        outputs_torch = torch_model(
            torch.tensor(query), past_key_values=kvcache_torch)
        logit_torch = outputs_torch['logits']
        kvcache_torch = outputs_torch['past_key_values']

        ####################################
        # infinitensor
        ####################################
        # copyin input
        (list(stub.inputs.items()))[0][1].copyin_int64(
            query.reshape(-1).tolist())
        (list(stub.inputs.items()))[1][1].copyin_int64(
            position_id.reshape(-1).tolist())

        stub.run()

        ####################################
        # validation
        ####################################
        # copyout output
        logits_it = np.array((list(stub.outputs.items()))
                                [0][1].copyout_float())
        
        try:
            np.testing.assert_allclose(
                logit_torch[:, -1, :].detach().cpu().numpy().flatten(), logits_it, rtol=1e-3, atol=1e-3)
        except Exception as e: 
            try:
                np.testing.assert_allclose(
                    np.argmax(logit_torch[:, -1, :].detach().cpu().numpy().flatten()), np.argmax(logits_it), rtol=1e-3, atol=1e-3)
            except:
                count_wrong = count_wrong + 1

    result = "{}/{} failed.".format(count_wrong, args.n_max_length)
    print(result)
    del stub
