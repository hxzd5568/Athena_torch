import re

dyn_template = """
import torch

%MODULE

model = GraphModule()

params = torch.load('model_weights.pth', weights_only=True)
# x = torch.randn(16, 3, 32, 32, dtype=torch.float32)
x = torch.load("inputs.pt")
y = model(**params, x=x)
print(y)
"""

def convert_param_name(original_name):
    if original_name.endswith(('.weight', '.bias')):
        prefix = 'p_'
        base_name = original_name

    elif any(x in original_name for x in ['running_mean', 'running_var', 'num_batches_tracked']):
        prefix = 'b_'
        base_name = original_name
    else:
        raise ValueError(f"Unrecognized parameter type: {original_name}")
    
    if '.' in base_name:
        parts = base_name.split('.')
        if len(parts) == 2 and not parts[0].startswith('layer'):
            return prefix + parts[0] + '_' + parts[1]
        else:
            # layer1.0 -> layer1___0___
            pattern = r'(layer\d+)\.(\d+)\.'
            replacement = r'\1___\2___'
            converted = re.sub(pattern, replacement, base_name)
            converted = converted.replace('.', '_')
            return f"{prefix}getattr_l__self___{converted}"
    else:
        return prefix + base_name

def indent_with_tab(code: str) -> str:
    lines = code.splitlines()
    indented_lines = [f"\t{line}" for line in lines]
    return "\n".join(indented_lines)


def apply_templates(code: str) -> str:
    code = code.replace("\tGraphModule()", "class GraphModule(torch.nn.Module):")
    py_code = dyn_template.replace('%MODULE', code)
    return py_code


""" # test convert_param_name
original_names = ['conv1.weight', 'bn1.weight', 'bn1.bias', 'layer1.0.conv1.weight', 'layer1.0.bn1.weight', 'layer1.0.bn1.bias', 'layer1.0.conv2.weight', 'layer1.0.bn2.weight', 'layer1.0.bn2.bias', 'layer1.1.conv1.weight', 'layer1.1.bn1.weight', 'layer1.1.bn1.bias', 'layer1.1.conv2.weight', 'layer1.1.bn2.weight', 'layer1.1.bn2.bias', 'layer2.0.conv1.weight', 'layer2.0.bn1.weight', 'layer2.0.bn1.bias', 'layer2.0.conv2.weight', 'layer2.0.bn2.weight', 'layer2.0.bn2.bias', 'layer2.0.downsample.0.weight', 'layer2.0.downsample.1.weight', 'layer2.0.downsample.1.bias', 'layer2.1.conv1.weight', 'layer2.1.bn1.weight', 'layer2.1.bn1.bias', 'layer2.1.conv2.weight', 'layer2.1.bn2.weight', 'layer2.1.bn2.bias', 'layer3.0.conv1.weight', 'layer3.0.bn1.weight', 'layer3.0.bn1.bias', 'layer3.0.conv2.weight', 'layer3.0.bn2.weight', 'layer3.0.bn2.bias', 'layer3.0.downsample.0.weight', 'layer3.0.downsample.1.weight', 'layer3.0.downsample.1.bias', 'layer3.1.conv1.weight', 'layer3.1.bn1.weight', 'layer3.1.bn1.bias', 'layer3.1.conv2.weight', 'layer3.1.bn2.weight', 'layer3.1.bn2.bias', 'layer4.0.conv1.weight', 'layer4.0.bn1.weight', 'layer4.0.bn1.bias', 'layer4.0.conv2.weight', 'layer4.0.bn2.weight', 'layer4.0.bn2.bias', 'layer4.0.downsample.0.weight', 'layer4.0.downsample.1.weight', 'layer4.0.downsample.1.bias', 'layer4.1.conv1.weight', 'layer4.1.bn1.weight', 'layer4.1.bn1.bias', 'layer4.1.conv2.weight', 'layer4.1.bn2.weight', 'layer4.1.bn2.bias', 'fc.weight', 'fc.bias', 'bn1.running_mean', 'bn1.running_var', 'bn1.num_batches_tracked', 'layer1.0.bn1.running_mean', 'layer1.0.bn1.running_var', 'layer1.0.bn1.num_batches_tracked', 'layer1.0.bn2.running_mean', 'layer1.0.bn2.running_var', 'layer1.0.bn2.num_batches_tracked', 'layer1.1.bn1.running_mean', 'layer1.1.bn1.running_var', 'layer1.1.bn1.num_batches_tracked', 'layer1.1.bn2.running_mean', 'layer1.1.bn2.running_var', 'layer1.1.bn2.num_batches_tracked', 'layer2.0.bn1.running_mean', 'layer2.0.bn1.running_var', 'layer2.0.bn1.num_batches_tracked', 'layer2.0.bn2.running_mean', 'layer2.0.bn2.running_var', 'layer2.0.bn2.num_batches_tracked', 'layer2.0.downsample.1.running_mean', 'layer2.0.downsample.1.running_var', 'layer2.0.downsample.1.num_batches_tracked', 'layer2.1.bn1.running_mean', 'layer2.1.bn1.running_var', 'layer2.1.bn1.num_batches_tracked', 'layer2.1.bn2.running_mean', 'layer2.1.bn2.running_var', 'layer2.1.bn2.num_batches_tracked', 'layer3.0.bn1.running_mean', 'layer3.0.bn1.running_var', 'layer3.0.bn1.num_batches_tracked', 'layer3.0.bn2.running_mean', 'layer3.0.bn2.running_var', 'layer3.0.bn2.num_batches_tracked', 'layer3.0.downsample.1.running_mean', 'layer3.0.downsample.1.running_var', 'layer3.0.downsample.1.num_batches_tracked', 'layer3.1.bn1.running_mean', 'layer3.1.bn1.running_var', 'layer3.1.bn1.num_batches_tracked', 'layer3.1.bn2.running_mean', 'layer3.1.bn2.running_var', 'layer3.1.bn2.num_batches_tracked', 'layer4.0.bn1.running_mean', 'layer4.0.bn1.running_var', 'layer4.0.bn1.num_batches_tracked', 'layer4.0.bn2.running_mean', 'layer4.0.bn2.running_var', 'layer4.0.bn2.num_batches_tracked', 'layer4.0.downsample.1.running_mean', 'layer4.0.downsample.1.running_var', 'layer4.0.downsample.1.num_batches_tracked', 'layer4.1.bn1.running_mean', 'layer4.1.bn1.running_var', 'layer4.1.bn1.num_batches_tracked', 'layer4.1.bn2.running_mean', 'layer4.1.bn2.running_var', 'layer4.1.bn2.num_batches_tracked']
ans = ['p_conv1_weight', 'p_bn1_weight', 'p_bn1_bias', 'p_getattr_l__self___layer1___0___conv1_weight', 'p_getattr_l__self___layer1___0___bn1_weight', 'p_getattr_l__self___layer1___0___bn1_bias', 'p_getattr_l__self___layer1___0___conv2_weight', 'p_getattr_l__self___layer1___0___bn2_weight', 'p_getattr_l__self___layer1___0___bn2_bias', 'p_getattr_l__self___layer1___1___conv1_weight', 'p_getattr_l__self___layer1___1___bn1_weight', 'p_getattr_l__self___layer1___1___bn1_bias', 'p_getattr_l__self___layer1___1___conv2_weight', 'p_getattr_l__self___layer1___1___bn2_weight', 'p_getattr_l__self___layer1___1___bn2_bias', 'p_getattr_l__self___layer2___0___conv1_weight', 'p_getattr_l__self___layer2___0___bn1_weight', 'p_getattr_l__self___layer2___0___bn1_bias', 'p_getattr_l__self___layer2___0___conv2_weight', 'p_getattr_l__self___layer2___0___bn2_weight', 'p_getattr_l__self___layer2___0___bn2_bias', 'p_getattr_l__self___layer2___0___downsample_0_weight', 'p_getattr_l__self___layer2___0___downsample_1_weight', 'p_getattr_l__self___layer2___0___downsample_1_bias', 'p_getattr_l__self___layer2___1___conv1_weight', 'p_getattr_l__self___layer2___1___bn1_weight', 'p_getattr_l__self___layer2___1___bn1_bias', 'p_getattr_l__self___layer2___1___conv2_weight', 'p_getattr_l__self___layer2___1___bn2_weight', 'p_getattr_l__self___layer2___1___bn2_bias', 'p_getattr_l__self___layer3___0___conv1_weight', 'p_getattr_l__self___layer3___0___bn1_weight', 'p_getattr_l__self___layer3___0___bn1_bias', 'p_getattr_l__self___layer3___0___conv2_weight', 'p_getattr_l__self___layer3___0___bn2_weight', 'p_getattr_l__self___layer3___0___bn2_bias', 'p_getattr_l__self___layer3___0___downsample_0_weight', 'p_getattr_l__self___layer3___0___downsample_1_weight', 'p_getattr_l__self___layer3___0___downsample_1_bias', 'p_getattr_l__self___layer3___1___conv1_weight', 'p_getattr_l__self___layer3___1___bn1_weight', 'p_getattr_l__self___layer3___1___bn1_bias', 'p_getattr_l__self___layer3___1___conv2_weight', 'p_getattr_l__self___layer3___1___bn2_weight', 'p_getattr_l__self___layer3___1___bn2_bias', 'p_getattr_l__self___layer4___0___conv1_weight', 'p_getattr_l__self___layer4___0___bn1_weight', 'p_getattr_l__self___layer4___0___bn1_bias', 'p_getattr_l__self___layer4___0___conv2_weight', 'p_getattr_l__self___layer4___0___bn2_weight', 'p_getattr_l__self___layer4___0___bn2_bias', 'p_getattr_l__self___layer4___0___downsample_0_weight', 'p_getattr_l__self___layer4___0___downsample_1_weight', 'p_getattr_l__self___layer4___0___downsample_1_bias', 'p_getattr_l__self___layer4___1___conv1_weight', 'p_getattr_l__self___layer4___1___bn1_weight', 'p_getattr_l__self___layer4___1___bn1_bias', 'p_getattr_l__self___layer4___1___conv2_weight', 'p_getattr_l__self___layer4___1___bn2_weight', 'p_getattr_l__self___layer4___1___bn2_bias', 'p_fc_weight', 'p_fc_bias', 'b_bn1_running_mean', 'b_bn1_running_var', 'b_bn1_num_batches_tracked', 'b_getattr_l__self___layer1___0___bn1_running_mean', 'b_getattr_l__self___layer1___0___bn1_running_var', 'b_getattr_l__self___layer1___0___bn1_num_batches_tracked', 'b_getattr_l__self___layer1___0___bn2_running_mean', 'b_getattr_l__self___layer1___0___bn2_running_var', 'b_getattr_l__self___layer1___0___bn2_num_batches_tracked', 'b_getattr_l__self___layer1___1___bn1_running_mean', 'b_getattr_l__self___layer1___1___bn1_running_var', 'b_getattr_l__self___layer1___1___bn1_num_batches_tracked', 'b_getattr_l__self___layer1___1___bn2_running_mean', 'b_getattr_l__self___layer1___1___bn2_running_var', 'b_getattr_l__self___layer1___1___bn2_num_batches_tracked', 'b_getattr_l__self___layer2___0___bn1_running_mean', 'b_getattr_l__self___layer2___0___bn1_running_var', 'b_getattr_l__self___layer2___0___bn1_num_batches_tracked', 'b_getattr_l__self___layer2___0___bn2_running_mean', 'b_getattr_l__self___layer2___0___bn2_running_var', 'b_getattr_l__self___layer2___0___bn2_num_batches_tracked', 'b_getattr_l__self___layer2___0___downsample_1_running_mean', 'b_getattr_l__self___layer2___0___downsample_1_running_var', 'b_getattr_l__self___layer2___0___downsample_1_num_batches_tracked', 'b_getattr_l__self___layer2___1___bn1_running_mean', 'b_getattr_l__self___layer2___1___bn1_running_var', 'b_getattr_l__self___layer2___1___bn1_num_batches_tracked', 'b_getattr_l__self___layer2___1___bn2_running_mean', 'b_getattr_l__self___layer2___1___bn2_running_var', 'b_getattr_l__self___layer2___1___bn2_num_batches_tracked', 'b_getattr_l__self___layer3___0___bn1_running_mean', 'b_getattr_l__self___layer3___0___bn1_running_var', 'b_getattr_l__self___layer3___0___bn1_num_batches_tracked', 'b_getattr_l__self___layer3___0___bn2_running_mean', 'b_getattr_l__self___layer3___0___bn2_running_var', 'b_getattr_l__self___layer3___0___bn2_num_batches_tracked', 'b_getattr_l__self___layer3___0___downsample_1_running_mean', 'b_getattr_l__self___layer3___0___downsample_1_running_var', 'b_getattr_l__self___layer3___0___downsample_1_num_batches_tracked', 'b_getattr_l__self___layer3___1___bn1_running_mean', 'b_getattr_l__self___layer3___1___bn1_running_var', 'b_getattr_l__self___layer3___1___bn1_num_batches_tracked', 'b_getattr_l__self___layer3___1___bn2_running_mean', 'b_getattr_l__self___layer3___1___bn2_running_var', 'b_getattr_l__self___layer3___1___bn2_num_batches_tracked', 'b_getattr_l__self___layer4___0___bn1_running_mean', 'b_getattr_l__self___layer4___0___bn1_running_var', 'b_getattr_l__self___layer4___0___bn1_num_batches_tracked', 'b_getattr_l__self___layer4___0___bn2_running_mean', 'b_getattr_l__self___layer4___0___bn2_running_var', 'b_getattr_l__self___layer4___0___bn2_num_batches_tracked', 'b_getattr_l__self___layer4___0___downsample_1_running_mean', 'b_getattr_l__self___layer4___0___downsample_1_running_var', 'b_getattr_l__self___layer4___0___downsample_1_num_batches_tracked', 'b_getattr_l__self___layer4___1___bn1_running_mean', 'b_getattr_l__self___layer4___1___bn1_running_var', 'b_getattr_l__self___layer4___1___bn1_num_batches_tracked', 'b_getattr_l__self___layer4___1___bn2_running_mean', 'b_getattr_l__self___layer4___1___bn2_running_var', 'b_getattr_l__self___layer4___1___bn2_num_batches_tracked']
for i, name in enumerate(original_names):
    if convert_param_name(name) == ans[i]:
        pass
    else:
        print(f"{name} -> {convert_param_name(name)} !=  {ans[i]}")
        exit()
"""