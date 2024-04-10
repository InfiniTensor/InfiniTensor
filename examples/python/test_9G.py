import os
from pyinfinitensor.onnx import OnnxStub, backend
import numpy as np
import onnx
import torch
from tqdm import tqdm
import onnx_graphsurgeon as gs
import time
import nvtx
import argparse
from mpi4py import MPI
from pytrie import StringTrie
import io
import json
import re
from typing import (
    Dict,
    List,
    IO,
)

parser = argparse.ArgumentParser(description='')
parser.add_argument('--batchsize', dest='batchsize', type=int, default=1)
parser.add_argument('--layer', dest='n_layers', type=int, default=48)
parser.add_argument("--num_nodes", dest='num_nodes',
                    type=int, default=1, help="number of nodes")
parser.add_argument("--world_size", dest="world_size",
                    type=int, default=1, help="")
parser.add_argument("--nproc_per_node", dest="nproc_per_node",
                    type=int, default=1, help="number of processes per node")
parser.add_argument("--n_max_length", dest="n_max_length",
                    type=int, default=1024, help="number of processes per node")
parser.add_argument("--vocab_size", dest="vocab_size",
                    type=int, default=119696, help="vocabulary size")
parser.add_argument("--hidden_size", dest="hidden_size",
                    type=int, default=4096, help="vocabulary size")
parser.add_argument('--rank', dest='rank', type=int, default=0)
parser.add_argument('--speedup', action='store_true')
parser.add_argument('--no_cudagraph', action='store_true')
parser.add_argument('--fp16', action='store_true')
args = parser.parse_args()
comm = MPI.COMM_WORLD
args.rank = comm.Get_rank()
args.nproc_per_node = comm.Get_size()
args.world_size = args.num_nodes * args.nproc_per_node

ONNX_MODEL_PATH = "/data3/shared/xnsong/9G/dist/9g_dist_bs{}_layer{}_fp{}_worldsize{}_rank{}.onnx".format(
    args.batchsize, args.n_layers, 16 if args.fp16 else 32, args.world_size, args.rank)

weight_path = "9g_dist_bs{}_layer{}_fp{}_worldsize{}_rank{}.pb".format(
    args.batchsize, args.n_layers, 16 if args.fp16 else 32, args.world_size, args.rank)

model_dir = "/data1/shared/9G-Infer/models/11B-Chat-QY-epoch-8/cpm9g-11b-sft.pt"

@gs.Graph.register()
def RMSNorm(self, a, b):
    return self.layer(op="RMSNorm", inputs=a, outputs=b)

@gs.Graph.register()
def RoPE(self, a, b):
    return self.layer(op="RoPE", inputs=a, outputs=b)

@gs.Graph.register()
def AttentionKVCache(self, a, b):
    return self.layer(op="AttentionKVCache", inputs=a, outputs=b)

def to_numpy(dict):
    ret = dict
    if args.fp16:
        ret = np.float16(ret)
    else:
        ret = np.float32(ret)
    return ret

def parallel(array, split='replicate'):
    if args.world_size > 1 and split == 'partial_column':
        return np.hsplit(array, args.world_size)[args.rank]
    elif args.world_size > 1 and split == 'partial_row':
        return np.vsplit(array, args.world_size)[args.rank]
    return array


def generate_onnx(ONNX_MODEL_PATH):
    state_dict = torch.load(f'{model_dir}', map_location='cpu')
    new_state_dict = {name: param.cpu().numpy()
        for name, param in state_dict.items()
    }

    operators = []
    graph = gs.Graph(nodes=operators)
    gather_input = gs.Variable(name="gather_input.0", dtype=np.int64, shape=(1,1))
    pos_input = gs.Variable(name="pos_input.0", dtype=np.int64, shape=(1,1))

    embedding_weight = gs.Constant(name="embedding.weight", values=to_numpy(new_state_dict["input_embedding.weight"]))
    gather_output = gs.Variable(name="gather_output.0") 
    gather = gs.Node(op="Gather", inputs=[embedding_weight, gather_input], outputs=[gather_output])
    operators.append(gather)
    input = gather_output
    
    graph.inputs=[gather_input, pos_input]
    graph.outputs=[]

    for i in tqdm(range(args.n_layers)):
        # global input
        attn_kcache_input = gs.Variable(name="/layers." + str(i) + "/attn/kcache_input", dtype=np.float32, shape=(1,32,1023,128))
        attn_vcache_input = gs.Variable(name="/layers." + str(i) + "/attn/vcache_input", dtype=np.float32, shape=(1,32,1023,128))
        graph.inputs.append(attn_kcache_input)
        graph.inputs.append(attn_vcache_input)

        # weight
        layernorm_0_mul_weight = gs.Constant(name="/layers." + str(i) + "/layernorm.0/mul_weight", 
                                             values=to_numpy(new_state_dict["encoder.layers." + str(i) + ".self_att.layernorm_before_attention.weight"]))
        attn_qproj_weight = gs.Constant(name="/layers." + str(i) + "/attn/qproj_weight", 
                                        values=parallel(
                                            np.transpose(
                                                to_numpy(
                                                    new_state_dict["encoder.layers." + str(i) + ".self_att.self_attention.project_q.weight"]))
                                            , 'partial_column'))
        attn_kproj_weight = gs.Constant(name="/layers." + str(i) + "/attn/kproj_weight", 
                                        values=parallel(
                                            np.transpose(
                                                to_numpy(
                                                    new_state_dict["encoder.layers." + str(i) + ".self_att.self_attention.project_k.weight"]))
                                            , 'partial_column'))
        attn_vproj_weight = gs.Constant(name="/layers." + str(i) + "/attn/vproj_weight", 
                                        values=parallel(
                                            np.transpose(
                                                to_numpy(
                                                    new_state_dict["encoder.layers." + str(i) + ".self_att.self_attention.project_v.weight"]))
                                            , 'partial_column'))
        attn_outmatmul_input = gs.Constant(name="/layers." + str(i) + "/attn/outmatmul_weight", 
                                           values=parallel(
                                               np.transpose(
                                                   to_numpy(
                                                       new_state_dict["encoder.layers." + str(i) + ".self_att.self_attention.attention_out.weight"]))
                                                , 'partial_row'))
        
        layernorm_1_mul_weight = gs.Constant(name="/layers." + str(i) + "/layernorm.1/mul_weight", 
                                             values=to_numpy(new_state_dict["encoder.layers." + str(i) + ".ffn.layernorm_before_ffn.weight"]))
        ffn_matmul_0_input = gs.Constant(name="/layers." + str(i) + "/ffn/matmul_0_weight", 
                                         values=parallel(
                                             np.transpose(
                                                 to_numpy(
                                                     new_state_dict["encoder.layers." + str(i) + ".ffn.ffn.w_in.w_0.weight"]))
                                             , 'partial_column'))
        ffn_matmul_1_input = gs.Constant(name="/layers." + str(i) + "/ffn/matmul_1_weight", 
                                         values=parallel(
                                             np.transpose(
                                                 to_numpy(
                                                     new_state_dict["encoder.layers." + str(i) + ".ffn.ffn.w_in.w_1.weight"]))
                                             , 'partial_column'))
        ffn_matmul_out_input = gs.Constant(name="/layers." + str(i) + "/ffn/matmul_out_weight", 
                                           values=parallel(
                                               np.transpose(
                                                   to_numpy(
                                                       new_state_dict["encoder.layers." + str(i) + ".ffn.ffn.w_out.weight"]))
                                                , 'partial_row'))
        
        attn_qrope_output = gs.Variable(name="/layers." + str(i) + "/attn/qrope_output")
        attn_krope_output = gs.Variable(name="/layers." + str(i) + "/attn/krope_output")
        attn_kvcache_output = gs.Variable(name="/layers." + str(i) + "/attn/kvcache_output")
        layernorm_0_mul_output_1 = gs.Variable(name="/layers." + str(i) + "/layernorm.0/mul_output_1")
        layernorm_1_mul_output_1 = gs.Variable(name="/layers." + str(i) + "/layernorm.1/mul_output_1")
        attn_qproj_output = gs.Variable(name="/layers." + str(i) + "/attn/qproj_output")
        attn_kproj_output = gs.Variable(name="/layers." + str(i) + "/attn/kproj_output")
        attn_vproj_output = gs.Variable(name="/layers." + str(i) + "/attn/vproj_output")
        attn_outmatmul_output = gs.Variable(name="/layers." + str(i) + "/attn/outmatmul_output")
        attn_outadd_output = gs.Variable(name="/layers." + str(i) + "/attn/outadd_output")
        ffn_matmul_0_output = gs.Variable(name="/layers." + str(i) + "/ffn/matmul_0_output")
        ffn_silu_output = gs.Variable(name="/layers." + str(i) + "/ffn/silu_output")
        ffn_matmul_1_output = gs.Variable(name="/layers." + str(i) + "/ffn/matmul_1_output")
        ffn_mul_output = gs.Variable(name="/layers." + str(i) + "/ffn/mul_output")
        ffn_matmul_out_output = gs.Variable(name="/layers." + str(i) + "/ffn/matmul_out_output")
        ffn_add_output = gs.Variable(name="/layers." + str(i) + "/ffn/add_output")

        graph.RMSNorm([input, layernorm_0_mul_weight], [layernorm_0_mul_output_1])
        attn_qproj = gs.Node(op="MatMul", inputs=[layernorm_0_mul_output_1, attn_qproj_weight], outputs=[attn_qproj_output])
        operators.append(attn_qproj)
        attn_kproj = gs.Node(op="MatMul", inputs=[layernorm_0_mul_output_1, attn_kproj_weight], outputs=[attn_kproj_output])
        operators.append(attn_kproj)
        attn_vproj = gs.Node(op="MatMul", inputs=[layernorm_0_mul_output_1, attn_vproj_weight], outputs=[attn_vproj_output])
        operators.append(attn_vproj)
        graph.RoPE([pos_input, attn_qproj_output], [attn_qrope_output])
        graph.RoPE([pos_input, attn_kproj_output], [attn_krope_output])
        graph.AttentionKVCache([attn_kcache_input, attn_vcache_input, attn_qrope_output, attn_krope_output, attn_vproj_output, pos_input],[attn_kvcache_output])
        attn_outproj = gs.Node(op="MatMul", inputs=[attn_kvcache_output, attn_outmatmul_input], outputs=[attn_outmatmul_output])
        operators.append(attn_outproj)

        attn_reduce_sum_output = gs.Variable(name="/layers." + str(i) + "/attn/reducesum_output")
        if args.world_size > 1:
            reduce_sum = gs.Node(op="ReduceSum", name="/layers." + str(i) + "/attn/reducesum",
                                    inputs=[attn_outmatmul_output], outputs=[attn_reduce_sum_output], 
                                    attrs={"noop_with_empty_axes":1, "communicator":0})
            graph.nodes.append(reduce_sum)
        
        attn_outadd = gs.Node(op="Add", inputs=[input, attn_outmatmul_output if args.world_size == 1 else attn_reduce_sum_output], outputs=[attn_outadd_output])
        operators.append(attn_outadd)

        graph.RMSNorm([attn_outadd_output, layernorm_1_mul_weight], [layernorm_1_mul_output_1])

        ffn_matmul_0 = gs.Node(op="MatMul", inputs=[layernorm_1_mul_output_1, ffn_matmul_0_input], outputs=[ffn_matmul_0_output])
        operators.append(ffn_matmul_0)
        ffn_silu = gs.Node(op="Silu", inputs=[ffn_matmul_0_output], outputs=[ffn_silu_output])
        operators.append(ffn_silu)
        ffn_matmul_1 = gs.Node(op="MatMul", inputs=[layernorm_1_mul_output_1, ffn_matmul_1_input], outputs=[ffn_matmul_1_output])
        operators.append(ffn_matmul_1)
        ffn_mul = gs.Node(op="Mul", inputs=[ffn_silu_output, ffn_matmul_1_output], outputs=[ffn_mul_output])
        operators.append(ffn_mul)
        ffn_matmul_out = gs.Node(op="MatMul", inputs=[ffn_mul_output, ffn_matmul_out_input], outputs=[ffn_matmul_out_output])
        operators.append(ffn_matmul_out)

        ffn_reduce_sum_output = gs.Variable(name="/layers." + str(i) + "/ffn/reducesum_output")
        if args.world_size > 1:
            reduce_sum = gs.Node(op="ReduceSum", name="/layers." + str(i) + "/ffn/reducesum",
                                    inputs=[ffn_matmul_out_output], outputs=[ffn_reduce_sum_output], 
                                    attrs={"noop_with_empty_axes":1, "communicator":0})
            graph.nodes.append(reduce_sum)

        ffn_add = gs.Node(op="Add", inputs=[attn_outadd_output, ffn_matmul_out_output if args.world_size == 1 else ffn_reduce_sum_output], outputs=[ffn_add_output])
        operators.append(ffn_add)
        input = ffn_add_output

    layernorm_mul_weight = gs.Constant(name="/output/layernorm/mul_weight", values=to_numpy(new_state_dict["encoder.output_layernorm.weight"]))
    layernorm_mul_output_1 = gs.Variable(name="/output/layernorm/mul_output_1")

    graph.RMSNorm([input, layernorm_mul_weight], [layernorm_mul_output_1])

    lm_head_weight = gs.Constant(name="/output/lm_head/weight", values=np.transpose(to_numpy(new_state_dict["lm_head.weight"])))
    lm_head_output = gs.Variable(name="/output/lm_head/output") 
    lm_head = gs.Node(op="MatMul", inputs=[layernorm_mul_output_1, lm_head_weight], outputs=[lm_head_output])
    operators.append(lm_head)

    if args.fp16:
        final_cast_output = gs.Variable(name="/output/cast/output", dtype=np.float32, shape=(1,1,args.vocab_size))
        final_cast = gs.Node(op="Cast", inputs=[lm_head_output], outputs=[final_cast_output])
        final_cast.attrs["to"] = np.float32
        operators.append(final_cast)
        graph.outputs.append(final_cast_output)
    else:
        lm_head_output.dtype=np.float32
        lm_head_output.shape=(1,1,args.vocab_size)
        graph.outputs.append(lm_head_output)

    onnx.save(gs.export_onnx(graph), ONNX_MODEL_PATH, save_as_external_data=True, location=weight_path) 
    return


def load_vocab(fp: IO[bytes]) -> Dict[str, int]:
    """Loads a vocabulary file into a dictionary."""
    vocab: Dict[str, int] = {}

    reader = io.TextIOWrapper(fp, encoding="utf-8")
    for token in reader.readlines():
        token = token.strip()
        if len(token) == 0:
            continue
        token = json.loads(token)
        vocab[token] = len(vocab)
    return vocab


class CPM9GTokenizer(object):
    def __init__(self, path):
        self.unk_token = "<unk>"
        self.bos_token = "<s>"
        self.eos_token = "</s>"
        self.byte_list = ["<0x0{}>".format(hex(i).upper()[2:]) for i in range(0x10)] + [
            "<0x{}>".format(hex(i).upper()[2:]) for i in range(0x10, 0x100)
        ]

        self._special_token_set = set([self.unk_token, self.bos_token, self.eos_token] + self.byte_list)

        all_tokens = load_vocab(io.FileIO(path, "rb"))

        self.encoder: Dict[str, int] = {}
        self._special_encoder: Dict[str, int] = {}
        for token, token_id in all_tokens.items():
            if token in self._special_token_set:
                self._special_encoder[token] = token_id
            else:
                self.encoder[token] = token_id

        self.decoder = {v: k for k, v in self.encoder.items()}
        self._byte_decoder = {self._special_encoder[token]: i for i, token in enumerate(self.byte_list)}

        self._max_word_len = max([len(x) for x in self.encoder.keys()])

        self._len_word_first = {}
        for x in self.encoder.keys():
            if not x[0] in self._len_word_first:
                self._len_word_first[x[0]] = 1
            if len(x) > self._len_word_first[x[0]]:
                self._len_word_first[x[0]] = len(x)
        self.tencoder = StringTrie(self.encoder)

    def get_piece(self, text: str) -> str:
        if text[0] in self._len_word_first:
            text = text[: self._len_word_first[text[0]]]
            len_text = len(text)
            for i in range(len(text)):
                sub = text[: len_text - i]
                if sub in self.encoder:
                    return sub
        return text[0]

    @property
    def vocab_size(self):
        return len(self)

    @property
    def eos_id(self):
        return self._special_encoder[self.eos_token]

    @property
    def bos_id(self):
        return self._special_encoder[self.bos_token]

    @property
    def unk_id(self):
        return self._special_encoder[self.unk_token]

    def __len__(self):
        return len(self.encoder) + len(self._special_encoder)

    def tokenize(self, text: str) -> List[str]:
        output_tokens: List[str] = []
        st = 0
        while st < len(text):
            piece = self.get_piece(text[st:])
            output_tokens.append(piece)
            st += len(piece)
        return output_tokens

    @staticmethod
    def escape(text: str) -> str:
        return text

    @staticmethod
    def unescape(text: str) -> str:
        return text

    def encode(self, text: str, with_bos = True) -> List[int]:
        ret = []
        if with_bos:
            ret.append(self.bos_id)
        for x in self.tokenize(text):
            if x in self.encoder:
                ret.append(self.encoder[x])
            else:
                ret.extend(self._encode_unicode(x))
        return ret

    def decode(self, tokens: List[int]):
        """Decode ids into a string."""
        ret = []
        st = 0
        while st < len(tokens):
            if tokens[st] in self.decoder:
                ret.append(self.decoder[tokens[st]])
                st += 1
            elif tokens[st] in self._byte_decoder:
                first = self._byte_decoder[tokens[st]]
                length = 1 if first < 128 else len(re.search('^1+0', bin(first)[2:])[0])-1
                code = 0
                try:
                    for j in range(length):
                        code = code << 8 | self._byte_decoder[tokens[st + j]]
                    code = int.to_bytes(code, length, "big").decode("utf-8")
                    ret.append(code)
                except:
                    pass
                st = st + length
            elif tokens[st] == self.eos_id:
                ret.append(self.eos_token)
                st += 1
            elif tokens[st] == self.bos_id:
                ret.append(self.bos_token)
                st += 1
            else:
                ret.append(self.unk_token)
                st += 1
        return "".join(ret)

    def _encode_unicode(self, token):
        # wrap unicode encoding into a helper function
        ids = []
        utf8_id = token.encode("utf-8")
        for _id in utf8_id:
            ids.append(self._special_encoder[self.byte_list[_id]])
        return ids

    def next_token(self, text):
        # fast next token matching
        token, token_id = self.tencoder.longest_prefix_item(text, (None, None))
        if token is None:
            token = text[0]
            token_ids = self._encode_unicode(token)
        else:
            token_ids = [token_id]
        return token, token_ids


def start_worker(
    world_size: int, rank: int, local_rank: int, model: onnx.ModelProto, query
):
    model = onnx.load(ONNX_MODEL_PATH)
    runtime = backend.CudaRuntime(local_rank)
    if args.nproc_per_node > 1:
        runtime.init_comm(
            "9g",
            world_size,
            rank,
        )
        print("[{}] comm init.".format(rank))

    stub = OnnxStub(model, runtime)
    print("[{}] stub init.".format(rank))

    for i in range(10):
        if args.no_cudagraph:
            stub.run()
        else:
            stub.run_with_cudagraph()
    print("[{}] stub warmup.".format(rank))

    tokenizer = CPM9GTokenizer("/data1/shared/9G-Infer/models/11B-Chat-QY-epoch-8/vocabs.txt")
    query = tokenizer.encode(query)

    output_tokens = []
    for i in range(len(query)):
        q = np.array(query[i])
        (list(stub.inputs.items()))[0][1].copyin_int64(q.reshape(-1).tolist())
        pos = i * np.ones((args.batchsize, 1), dtype=np.int64)
        (list(stub.inputs.items()))[1][1].copyin_int64(pos.reshape(-1).tolist())

        if args.no_cudagraph:
            stub.run()
        else:
            stub.run_with_cudagraph()

        if i == len(query) - 1:
            output = np.array((list(stub.outputs.items()))[-1][1].copyout_float16()) if False \
                else np.array((list(stub.outputs.items()))[-1][1].copyout_float())
            q = np.argmax(output)
            output_tokens.append(q)
        
    avg_time = 0
    count = 0
    while i < 1000:
        count = count + 1
        torch.cuda.synchronize()
        with nvtx.annotate("gen {}-th token".format(i), color="red"):
            i = i + 1
            (list(stub.inputs.items()))[0][1].copyin_int64(q.reshape(-1).tolist())
            pos = i * np.ones((args.batchsize, 1), dtype=np.int64)
            (list(stub.inputs.items()))[1][1].copyin_int64(pos.reshape(-1).tolist())

            t0 = time.time()
            if args.no_cudagraph:
                stub.run()
            else:
                stub.run_with_cudagraph()
            t1 = time.time()
            avg_time += t1 - t0

            output = np.array((list(stub.outputs.items()))[-1][1].copyout_float16()) if False \
                else np.array((list(stub.outputs.items()))[-1][1].copyout_float())

            # print(output)

            with nvtx.annotate("argmax".format(i), color="green"):
                q = np.argmax(output)
            if q == 2:
                break

            output_tokens.append(q)
    avg_time = avg_time / count
    print("avg_time_cost =", avg_time*1000, "ms")
    text = tokenizer.decode(output_tokens)
    return text


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    args.rank = comm.Get_rank()
    args.nproc_per_node = comm.Get_size()
    world_size = args.num_nodes * args.nproc_per_node

    if not os.path.exists(ONNX_MODEL_PATH):
        print("exporting onnx graph")
        generate_onnx(ONNX_MODEL_PATH)
    else:
        print("will use exsiting onnx graph")
    onnx_model = onnx.load(ONNX_MODEL_PATH)
    print("data loaded")
    

    #query = '''Beijing is the captial'''
    #query = '''什么是PTX？'''
    #query = '''生病了怎么办？'''
    #query = '''Happy'''
    query = '''def gcd(a, b):'''

    ####################################
    # infinitensor dist
    ####################################
    # run distributed parallel.
    pred = start_worker(world_size, args.rank, args.rank %
                  args.nproc_per_node, onnx_model, query)
    if args.rank == 0:
        print("输入：\n\n", query, "\n")
        print("输出：", pred)
