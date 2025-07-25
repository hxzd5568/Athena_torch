import os
import json
import torch, torchvision
import numpy as np
from torchvision import transforms
from torch.export import export
import utils
from torch import nn
from utils import convert_param_name, indent_with_tab, apply_templates

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

batch_size = 1
height, width = 1, 5
num_channels = 3

random_input = torch.rand(batch_size, num_channels, height, width)
normalized_input = normalize(random_input)


all_models = torchvision.models.list_models()
print(all_models)
print(len(all_models))

# Initialize models
model_name = "resnet18"
model = torchvision.models.get_model(model_name, weights="DEFAULT")
# print(model)
print(type(model))
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
normalized_input = normalized_input.to(device)

exported = export(model, args=(normalized_input,))

params = exported.state_dict
new_params = dict()
for k, v in params.items():
    new_params[convert_param_name(k)] = v

base_code = exported.graph_module.__str__()
write_code = apply_templates(base_code)
if not os.path.exists(f"{model_name}/"):
    os.mkdir(f"{model_name}")
with open(f'{model_name}/model.py', 'w') as fp:
    print(write_code, file=fp)
data = {"framework": "torch"}
file_path = f'{model_name}/attribute.json'
with open(file_path, 'w') as f:
    json.dump(data, f, indent=4)

converted = utils.convert_state_and_inputs(params, exported.example_inputs[0])
utils.save_converted_to_text(converted, file_path=f'{model_name}/source_tensor_meta.py')
utils.save_constraints_text(converted, file_path=f'{model_name}/input_tensor_constraints.py')
