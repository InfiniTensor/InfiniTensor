import torch
import torchvision
import os
from models.drn import drn_a_50, drn_c_26  


def export_model(modelfn, file_name):
    files = os.listdir()
    file = next(
        (name for name in files if name.endswith(file_name)), None
    )
    if file == None:
        # Load the pretrained model
        model = modelfn(pretrained = True)
        # Set the model to evaluation mode
        model.eval()

        # Create a sample input tensor
        input_tensor = torch.randn(1, 3, 224, 224)
        # Export the model to ONNX format
        torch.onnx.export(model, input_tensor, file_name)



from transformers import BertTokenizer, BertModel
def export_bert():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained("bert-base-uncased")
    text = "Replace me by any text you'd like."
    encoded_input = tokenizer(text, return_tensors='pt')
    #output = model(**encoded_input)
    #inputs = torch.randn(1, 128, dtype=torch.float32)
    input_names = ["input_ids", "attention_mask", "token_type_ids"]
    torch.onnx.export(model,tuple(encoded_input.values()),
                    f="bert.onnx",
                    input_names=input_names,
                
                    output_names=["output"])
    # https://huggingface.co/blog/convert-transformers-to-onnx#4-how-can-i-convert-a-transformers-model-bert-to-onnx

#export_model(torchvision.models.resnet18, "resnet18.onnx")
#export_model(torchvision.models.resnet50, "resnet50.onnx")
#export_model(drn_c_26, "drn_c_26.onnx")
export_model(torchvision.models.inception_v3, "inception_v3.onnx")
