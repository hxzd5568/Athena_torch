import torch, torchvision
import numpy as np
from torchvision import transforms
from torch.export import export
from utils import convert_param_name, indent_with_tab, apply_templates

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

batch_size = 2
height, width = 224, 220
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
torch.save(new_params, "model_weights.pth")

inputs = exported.example_inputs[0][0]
torch.save(inputs, "inputs.pt")

base_code = exported.graph_module.__str__()
write_code = indent_with_tab(base_code)
write_code = apply_templates(write_code)
with open(f'test_export_{model_name}.py', 'w') as fp:
    print(write_code, file=fp)
print(model(inputs))
