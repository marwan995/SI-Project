import argparse
import torch
import torch.nn as nn
from unet_plus_plus import UNetPlusPlus,ConvBlock

def count_conv2d(m, x, y):
    x = x[0]
    cin = m.in_channels // m.groups
    cout = m.out_channels // m.groups
    kh, kw = m.kernel_size
    batch_size = x.size()[0]

    kernel_mul = kh * kw * cin
    kernel_add = kh * kw * cin - 1
    bias_ops = 1 if m.bias is not None else 0
    ops = kernel_mul + kernel_add + bias_ops

    num_out_elements = y.numel()
    total_ops = num_out_elements * ops

    m.total_ops += torch.Tensor([int(total_ops)]).to(m.total_ops.device)

def count_bn2d(m, x, y):
    x = x[0]
    nelements = x.numel()
    total_sub = nelements
    total_div = nelements
    total_ops = total_sub + total_div

    m.total_ops += torch.Tensor([int(total_ops)]).to(m.total_ops.device)

def count_relu(m, x, y):
    x = x[0]
    nelements = x.numel()
    total_ops = nelements

    m.total_ops += torch.Tensor([int(total_ops)]).to(m.total_ops.device)

def count_softmax(m, x, y):
    x = x[0]
    batch_size, nfeatures = x.size()
    total_exp = nfeatures
    total_add = nfeatures - 1
    total_div = nfeatures
    total_ops = batch_size * (total_exp + total_add + total_div)

    m.total_ops += torch.Tensor([int(total_ops)]).to(m.total_ops.device)

def count_maxpool(m, x, y):
    kernel_ops = torch.prod(torch.Tensor([m.kernel_size])).to(x[0].device) - 1
    num_elements = y.numel()
    total_ops = kernel_ops * num_elements

    m.total_ops += torch.Tensor([int(total_ops)]).to(m.total_ops.device)

def count_avgpool(m, x, y):
    total_add = torch.prod(torch.Tensor([m.kernel_size])).to(x[0].device) - 1
    total_div = 1
    kernel_ops = total_add + total_div
    num_elements = y.numel()
    total_ops = kernel_ops * num_elements

    m.total_ops += torch.Tensor([int(total_ops)]).to(m.total_ops.device)

def count_linear(m, x, y):
    total_mul = m.in_features
    total_add = m.in_features - 1
    num_elements = y.numel()
    total_ops = (total_mul + total_add) * num_elements

    m.total_ops += torch.Tensor([int(total_ops)]).to(m.total_ops.device)

def count_upsample(m, x, y):
    if m.mode == 'bilinear':
        # 4 ops per pixel (bilinear interpolation)
        total_ops = y.numel() * 4
    elif m.mode == 'nearest':
        total_ops = 0  # nearest just copies values
    else:
        total_ops = 0  # unknown mode
    
    m.total_ops += torch.Tensor([int(total_ops)]).to(m.total_ops.device)

def profile(model, input_size, custom_ops={}):
    model.eval()
    device = next(model.parameters()).device

    def add_hooks(m):
        if len(list(m.children())) > 0: 
            return
        
        # Initialize buffers on the correct device
        m.register_buffer('total_ops', torch.zeros(1, device=device))
        m.register_buffer('total_params', torch.zeros(1, device=device))

        for p in m.parameters():
            m.total_params += torch.Tensor([p.numel()]).to(device)

        if isinstance(m, nn.Conv2d):
            m.register_forward_hook(count_conv2d)
        elif isinstance(m, nn.BatchNorm2d):
            m.register_forward_hook(count_bn2d)
        elif isinstance(m, nn.ReLU):
            m.register_forward_hook(count_relu)
        elif isinstance(m, (nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d)):
            m.register_forward_hook(count_maxpool)
        elif isinstance(m, (nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d)):
            m.register_forward_hook(count_avgpool)
        elif isinstance(m, nn.Linear):
            m.register_forward_hook(count_linear)
        elif isinstance(m, nn.Upsample):
            m.register_forward_hook(count_upsample)
        elif isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            pass
        else:
            print("Not implemented for ", m)

    model.apply(add_hooks)

    # Create input tensor on the same device as model
    x = torch.zeros(input_size, device=device)
    
    with torch.no_grad():
        model(x)

    total_ops = torch.Tensor([0]).to(device)
    total_params = torch.Tensor([0]).to(device)
    for m in model.modules():
        if len(list(m.children())) > 0: 
            continue
        total_ops += m.total_ops
        total_params += m.total_params

    return total_ops.item(), total_params.item()

def main(args):
    # Load model
    model = torch.load(args.model, weights_only=False)
    model.eval()
    
    # Prepare input size (add batch dimension)
    input_size = [1] + args.input_size
    
    # Profile
    total_ops, total_params = profile(model, input_size)
    print("#Ops: %.2f GOps" % (total_ops/1e9))
    print("#Parameters: %.2f M" % (total_params/1e6))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch model profiler")
    parser.add_argument("model", help="Path to model file")
    parser.add_argument("input_size", nargs=3, type=int,
                       help="Input size as channels height width")
    args = parser.parse_args()
    main(args)