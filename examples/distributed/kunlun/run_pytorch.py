import argparse
import torch
from transformers import BertModel, BertConfig
from transformers import GPT2Model, GPT2Config
from transformers import OPTModel, OPTConfig
from transformers import LlamaModel, LlamaConfig
import time
import numpy as np
import onnx
import os
import sys
from onnx.external_data_helper import convert_model_to_external_data
from onnxsim import simplify

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
def parse_args():
    parser = argparse.ArgumentParser(description="Run pytorch gpt2/bert/opt and optionally export onnx.")
    parser.add_argument(
        "--model", type=str, choices=["gpt2", "bert", "opt", "llama"], required=True, help="model type"
    )
    parser.add_argument("--batch_size", type=int, default=1, help="batch size.")
    parser.add_argument("--length", type=int, default=1, help="sequence length.")
    parser.add_argument(
        "--export_onnx",
        type=str,
        nargs="?",
        default=None,
        const="./",
        help="whether and where to export onnx file",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="./",
        help="path to save pytorch model input data"
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        default="./",
        help="path to save pytorch model output data"
    )
    parser.add_argument(
        "--export_only",
        action="store_true"
    )
    args = parser.parse_args()
    print("arg setting: ", args)
    return (
        args.model,
        args.batch_size,
        args.length,
        args.export_onnx,
        args.input_dir,
        args.result_dir,
        args.export_only
    )


def get_model(modelname):
    if modelname == "bert":
        model = BertModel.from_pretrained("bert-base-uncased", add_pooling_layer=False, hidden_act="gelu_new") # erf is not impl by infini
        voc_size = BertConfig().vocab_size
    elif modelname == "gpt2":
        model = GPT2Model.from_pretrained("gpt2")
        voc_size = GPT2Config().vocab_size
    elif modelname == "opt":
        model = OPTModel.from_pretrained("./opt-125m")
        voc_size = OPTConfig().vocab_size
    elif modelname == "llama":
        model = LlamaModel.from_pretrained("meta-llama/Llama-2-7b-hf")
        voc_size = LlamaConfig().vocab_size
    else :
        raise KeyError(modelname)

    model = model.eval()
    return model, voc_size

def run_pytorch(torch_model, voc_size, batchsize, len, model_name):
    data = np.random.randint(0, voc_size, (batchsize, len), dtype=np.int32)
    np.save(os.path.join(input_dir, f"{model_name}_input_0.npy"), data)
    inputs = torch.from_numpy(data).to("cuda")
    torch_model = torch_model.to("cuda")

    n_iter = 10
    with torch.no_grad():
        for _ in range(10):
            outputs = torch_model(inputs)
    torch.cuda.synchronize()
    begin = time.time()
    with torch.no_grad():
        for _ in range(n_iter):
            torch.cuda.synchronize()
            outputs = torch_model(inputs)
            #
            torch.cuda.synchronize()
    torch.cuda.synchronize()
    end = time.time()

    avg_time = (end - begin) / n_iter
    outputs = outputs.last_hidden_state.to("cpu")
    print("outputs abs mean:", abs(np.array(outputs)).mean())
    print(f"average time: {avg_time}")
    torch.cuda.memory.empty_cache()
    np.save(os.path.join(result_dir, f"{model_name}_output.npy"), \
                                        np.array(outputs))
    print(f"Save input & output as {model_name}_input_0.npy and {model_name}_output.npy")


def export_onnx(model_name, model, data, path, extern=False):
    # torch.onnx.export(model, data, path, verbose=False, do_constant_folding=True)

    if model_name != "llama":
        onnx_model = onnx.load(path)
        onnx_model, check = simplify(onnx_model,
                                 skipped_optimizers=['fuse_qkv', 'eliminate_duplicate_initializer'])
                                 # skipped_optimizers=['fuse_qkv'])
        assert check
        add_value_info_for_constants(onnx_model)
        onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
        if extern:
            extern_path = path.replace('.onnx', '.pb')
            if os.path.exists(extern_path):
                os.remove(extern_path)
            convert_model_to_external_data(
                onnx_model,
                all_tensors_to_one_file=True,
                location=extern_path.split("/")[-1],
                size_threshold=1024,
                convert_attribute=False,
            )
        onnx.save(onnx_model, path)
    else:
        sys.path.append("onnxsim_large_model")
        from onnx_utils import set_onnx_input_shape
        from compress_model import SIZE_1MB, compress_onnx_model, uncompress_onnx_model

        in_model_path = path
        out_model_path = in_model_path[:-5] + ".sim.onnx"

        onnx_model = onnx.load(in_model_path)
        print(f"load model from {in_model_path} success")

        size_th_bytes = 1024 * 1024
        onnx_model, removed_inits = compress_onnx_model(onnx_model, size_th_bytes=size_th_bytes)
        print("compress model success")

        onnx_model = set_onnx_input_shape(onnx_model, "")
        tensor_size_threshold = f"1024KB"
        skipped_optimizers = []
        skipped_optimizers.append("eliminate_duplicate_initializer")
        onnx_model, check = simplify(onnx_model, skipped_optimizers=skipped_optimizers,
                                    tensor_size_threshold=tensor_size_threshold)
        if not check:
            raise ValueError(f"simplify compressed model {in_model_path} failed")

        print(f"simplify model success")

        onnx_model = uncompress_onnx_model(onnx_model, removed_inits)
        print(f"uncompress model success")

        add_value_info_for_constants(onnx_model)

        onnx.save(onnx_model, out_model_path, save_as_external_data=True)


def add_value_info_for_constants(model : onnx.ModelProto):
    """
    Currently onnx.shape_inference doesn't use the shape of initializers, so add
    that info explicitly as ValueInfoProtos.
    Mutates the model.
    Args:
        model: The ModelProto to update.
    """
    # All (top-level) constants will have ValueInfos before IRv4 as they are all inputs
    if model.ir_version < 4:
        return

    def add_const_value_infos_to_graph(graph : onnx.GraphProto):
        inputs = {i.name for i in graph.input}
        existing_info = {vi.name: vi for vi in graph.value_info}
        for init in graph.initializer:
            # Check it really is a constant, not an input
            if init.name in inputs:
                continue

            # The details we want to add
            elem_type = init.data_type
            shape = init.dims

            # Get existing or create new value info for this constant
            vi = existing_info.get(init.name)
            if vi is None:
                vi = graph.value_info.add()
                vi.name = init.name

            # Even though it would be weird, we will not overwrite info even if it doesn't match
            tt = vi.type.tensor_type
            if tt.elem_type == onnx.TensorProto.UNDEFINED:
                tt.elem_type = elem_type
            if not tt.HasField("shape"):
                # Ensure we set an empty list if the const is scalar (zero dims)
                tt.shape.dim.extend([])
                for dim in shape:
                    tt.shape.dim.add().dim_value = dim

        # Handle subgraphs
        for node in graph.node:
            for attr in node.attribute:
                # Ref attrs refer to other attrs, so we don't need to do anything
                if attr.ref_attr_name != "":
                    continue

                if attr.type == onnx.AttributeProto.GRAPH:
                    add_const_value_infos_to_graph(attr.g)
                if attr.type == onnx.AttributeProto.GRAPHS:
                    for g in attr.graphs:
                        add_const_value_infos_to_graph(g)


    return add_const_value_infos_to_graph(model.graph)


def main():
    global input_dir, result_dir

    modelname, batchsize, seqlen, \
        export_path, input_dir, result_dir, export_only = parse_args()

    model, voc_size = get_model(modelname) # pytorch model

    if export_path is not None:
        os.makedirs(export_path, exist_ok=True)
        filename = "{}_{}_{}.onnx".format(modelname, batchsize, seqlen)
        path = os.path.join(export_path, filename)
        param = torch.zeros((batchsize, seqlen), dtype=torch.int)
        export_onnx(modelname, model, param, path, True) # export pytorch model to onnx model
        if export_only:
            return

    run_pytorch(model, voc_size, batchsize, seqlen, modelname)

if __name__ == "__main__":
    main()
