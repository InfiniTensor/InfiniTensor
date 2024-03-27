import argparse
import torch
from transformers import BertModel, BertConfig
from transformers import GPT2Model, GPT2Config
from transformers import OPTModel, OPTConfig
import time
import numpy as np
import onnx
import os
from onnx.external_data_helper import convert_model_to_external_data
from onnxsim import simplify

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
def parse_args():
    parser = argparse.ArgumentParser(description="Run pytorch gpt2/bert/opt and optionally export onnx.")
    parser.add_argument(
        "--model", type=str, choices=["gpt2", "bert", "opt"], required=True, help="model type"
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
    args = parser.parse_args()
    args = parser.parse_args()
    print("arg setting: ", args)
    return (
        args.model,
        args.batch_size,
        args.length,
        args.export_onnx
    )


def get_model(modelname):
    match modelname:
        case "bert":
            model = BertModel.from_pretrained("bert-base-uncased", add_pooling_layer=False, hidden_act="gelu_new") # erf is not impl by infini
            voc_size = BertConfig().vocab_size
        case "gpt2":
            model = GPT2Model.from_pretrained("gpt2")
            voc_size = GPT2Config().vocab_size
        case "opt":
            model = model = OPTModel.from_pretrained("./opt-125m")
            voc_size = OPTConfig().vocab_size
        case _:
            raise KeyError(modelname)

    model = model.eval()
    return model, voc_size

def run_pytorch(torch_model, voc_size, batchsize, len):
    data = np.random.randint(0, voc_size, (batchsize, len), dtype=np.int32)
    np.save("test_inputs", data)
    inputs = torch.from_numpy(data).to("cuda")
    torch_model = torch_model.to("cuda")

    n_iter = 20
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
    np.save("test_results", np.array(outputs))
    print("Save input & output as test_inputs.npy and test_results.npy")


def export_onnx(model, data, path, extern=False):
    torch.onnx.export(model, data, path, verbose=False, do_constant_folding=True)
    onnx_model = onnx.load(path)
    onnx_model, check = simplify(onnx_model, skipped_optimizers=['eliminate_duplicate_initializer'])
    #onnx_model, check = simplify(onnx_model, skipped_optimizers=['fuse_qkv', 'eliminate_duplicate_initializer'])
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
            location=extern_path,
            size_threshold=1024,
            convert_attribute=False,
        )
    onnx.save(onnx_model, path)

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
    modelname, batchsize, seqlen, export_path = parse_args()
    model, voc_size = get_model(modelname)
    if export_path is not None:
        filename = "{}_{}_{}.onnx".format(modelname, batchsize, seqlen)
        path = os.path.join(export_path, filename)
        param = torch.zeros((batchsize, seqlen), dtype=torch.int)
        export_onnx(model, param, path, True)

    run_pytorch(model, voc_size, batchsize, seqlen)

if __name__ == "__main__":
    main()

