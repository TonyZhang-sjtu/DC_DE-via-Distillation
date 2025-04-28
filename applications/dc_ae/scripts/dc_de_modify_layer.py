# build DC-AE models
# full DC-AE model list: https://huggingface.co/collections/mit-han-lab/dc-ae-670085b9400ad7197bb1009b
# export HF_ENDPOINT=https://hf-mirror.com
from efficientvit.ae_model_zoo import DCAE_HF

import argparse
import torch
import torch.nn as nn

# encode
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from efficientvit.apps.utils.image import DMCrop
from efficientvit.models.nn.norm import TritonRMSNorm2d
from safetensors.torch import save_file


def create_convolution_layer(original_conv, in_channels=1024, out_channels=1024, kernel_size=1, groups=1, bias=False, method="direct"):
    """
    Create a new convolution layer and copy a portion of the weights from the original layer.

    Args:
    - original_conv: The original convolution layer from which to copy weights.
    - in_channels: The number of input channels for the new convolution layer.
    - out_channels: The number of output channels for the new convolution layer.
    - kernel_size: The size of the convolution kernel (int or tuple).
    - bias: Boolean indicating whether to include bias terms in the new layer (default False).
    - method: Method for weight extraction, default is "direct". If "direct", copy the first part of weights.

    Returns:
    - new_weight: The new weight tensor that has been resized and potentially clipped.
    - new_conv: The newly created convolutional layer.
    """
    if method == "random":
        new_conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, kernel_size), stride=(1, 1), bias=bias, groups=groups)

    else:
        # Get the original convolution layer's weights
        original_weight = original_conv.weight.data

        # Handle different methods for cropping the weights
        if method == "direct":
            # Directly crop the original weights: Take the first `out_channels` rows and `in_channels` columns
            new_weight = original_weight[:out_channels, :in_channels, :, :]  # shape: [out_channels, in_channels, kernel_size, kernel_size]
        elif method == "gap":
            # Calculate step size for pruning
            step_out = original_weight.size(0) // out_channels  # Step size for output channels
            step_in = original_weight.size(1) // in_channels  # Step size for input channels
            if step_in == 0:
                step_in = 1
            if step_out == 0:
                step_out = 1
            # print("original_weight.size(0): ", original_weight.size(0), "out_channels: ", out_channels)
            # print("original_weight.size(1): ", original_weight.size(1), "in_channels", in_channels)
            # print("step_in: ", step_in, "step_out: ", step_out)
            
            # Select channels to keep (0, 2, 4, ..., for both in_channels and out_channels)
            selected_out_channels = list(range(0, original_weight.size(0), step_out))  # Select output channels
            selected_in_channels = list(range(0, original_weight.size(1), step_in))  # Select input channels
            # Ensure selected_out_channels and selected_in_channels have compatible shapes
            selected_out_channels = torch.tensor(selected_out_channels)
            selected_in_channels = torch.tensor(selected_in_channels)
            # print("selected_out_channels: ", selected_out_channels)
            # print("selected_in_channels: ", selected_in_channels)
            # # Using meshgrid to select combinations of out_channels and in_channels
            # out_idx, in_idx = torch.meshgrid(selected_out_channels, selected_in_channels)
            # out_idx = out_idx.flatten()
            # in_idx = in_idx.flatten()
            # print("in_idx: ", in_idx, "out_idx: ", out_idx)
            
            # Crop the original weight by selecting every second channel for both in and out channels
            # new_weight = original_weight[out_idx, in_idx, :, :]
            new_weight = original_weight[selected_out_channels[:, None], selected_in_channels]
            # new_weight = original_weight[out_idx, :, :, :]
            # new_weight = original_weight[:, in_idx, :, :]
        else:
            raise ValueError("Unsupported method: Please use 'direct' or 'random.")
        
        # Ensure the new weight tensor is contiguous
        new_weight = new_weight.contiguous()

        # Create the new convolution layer
        new_conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, kernel_size), stride=(1, 1), bias=bias, groups=groups)

        # Assign the cropped weight to the new convolution layer
        new_conv.weight.data = new_weight

    # Return the new weight and convolution layer
    return new_conv


def create_norm_layer(original_norm, channels=1024, norm_type="trms2d", eps=1e-05,  momentum=0.1, elementwise_affine=True, affine=True, track_running_stats=True, method="direct"):
    """
    Create a new norm layer by copying a portion of the weights from the original norm layer.
    
    Args:
    - original_norm: The original norm layer from which to copy parameters.
    - channels: The number of channels to keep in the new norm layer.
    - norm_type: Type of norm layer, "bn2d" for BatchNorm2d or "trms2d" for TritonRMSNorm2d.
    - eps: Epsilon value for normalization (default 1e-05).
    - elementwise_affine: Whether to apply element-wise affine transformation (default True).
    
    Returns:
    - new_norm: The new norm layer with the cropped parameters.
    """

    # Check the norm type and create the new norm layer accordingly
    if norm_type == "bn2d":
        if method == "random":
            new_norm = nn.BatchNorm2d(channels, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)
        else:
            if method == "direct":
                # For BatchNorm2d, crop the weight and bias to the first `channels` elements
                weight = original_norm.weight.data[:channels]
                bias = original_norm.bias.data[:channels] if original_norm.bias is not None else None
            elif method == "gap":
                # Calculate step size for pruning
                step = original_norm.weight.data.size(0) // channels  # Step size for output channels
                selected_channels = list(range(0, original_norm.weight.data.size(0), step)) 
                weight = original_norm.weight.data[selected_channels]
                bias = original_norm.bias.data[selected_channels] if original_norm.bias is not None else None
            else:
                raise ValueError("Unsupported method type. Please use 'direct' or 'random.")
            
            weight = weight.contiguous()
            bias = bias.contiguous()
            # Create the new BatchNorm2d layer
            new_norm = nn.BatchNorm2d(channels, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)
            new_norm.weight.data = weight
            if bias is not None:
                new_norm.bias.data = bias
    
    elif norm_type == "trms2d":
        if method == "random":
            new_norm = TritonRMSNorm2d(channels, eps=eps, elementwise_affine=elementwise_affine)
        else:
            if method == "direct":
                # For TritonRMSNorm2d, crop the weight to the first `channels` elements
                weight = original_norm.weight.data[:channels]
            elif method == "gap":
                step = original_norm.weight.data.size(0) // channels  # Step size for output channels
                selected_channels = list(range(0, original_norm.weight.data.size(0), step)) 
                weight = original_norm.weight.data[selected_channels]
            else:
                raise ValueError("Unsupported method type. Please use 'direct' or 'random.")
            
            weight = weight.contiguous()
            # Create the new TritonRMSNorm2d layer
            new_norm = TritonRMSNorm2d(channels, eps=eps, elementwise_affine=elementwise_affine)
            new_norm.weight.data = weight
    else:
        raise ValueError("Unsupported norm type. Please use 'bn2d' or 'trms2d'.")

    return new_norm


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Modify and save a modified DC-AE model.")
    parser.add_argument("--pretrained_model", type=str, required=False, default="/data2/user/jyzhang/MIT/efficientvit/applications/dc_ae/pretrained_models/dc-ae-f32c32-in-1.0",
                        help="Path to the pretrained DC-AE model.")
    parser.add_argument("--save_path", type=str, required=False, default="/data2/user/jyzhang/MIT/efficientvit/applications/dc_ae/pruned_models/dc-ae-f32c32-in-1.0-v1/model.safetensors",
                        help="Path to save the modified DC-AE model.")
    parser.add_argument("--prune_method", type=str, required=False, default="direct",
                        help="Method to prune the model, including 'random', 'direct', 'gap'")
    parser.add_argument("--prune_version", type=str, required=True, default="v1",
                        help="Method to prune the model, including 'v1', 'v2', 'v3'")
    return parser.parse_args()


def modify_and_save_model(args):
    """Modify layers of the DC-AE model and save the modified model."""
    
    # Load the pretrained model
    dc_vae = DCAE_HF.from_pretrained(args.pretrained_model).to(device).eval()

    # Modify Convolution Layers in the model
    # For brand new layer substitution, use parameter method = "random"; 
    # For direct truncation to desired shape, use method = "direct" or omit this parameter;
    # For gapping trunction, use method = "gap"

    if args.prune_version == "v1":
        dc_vae.decoder.stages[3].op_list[0].main.conv.conv = create_convolution_layer(dc_vae.decoder.stages[3].op_list[0].main.conv.conv, 512, 2048, 3, method=args.prune_method)
        dc_vae.decoder.stages[4].op_list[0].main.conv.conv = create_convolution_layer(dc_vae.decoder.stages[4].op_list[0].main.conv.conv, 1024, 2048, 3, True, method=args.prune_method)

        dc_vae.decoder.stages[4].op_list[1].context_module.main.qkv.conv = create_convolution_layer(dc_vae.decoder.stages[4].op_list[1].context_module.main.qkv.conv, 512, 1536, method=args.prune_method)
        dc_vae.decoder.stages[4].op_list[2].context_module.main.qkv.conv = create_convolution_layer(dc_vae.decoder.stages[4].op_list[2].context_module.main.qkv.conv, 512, 1536, method=args.prune_method)

        dc_vae.decoder.stages[4].op_list[1].context_module.main.proj.conv = create_convolution_layer(dc_vae.decoder.stages[4].op_list[1].context_module.main.proj.conv, 512, 512, method=args.prune_method)
        dc_vae.decoder.stages[4].op_list[2].context_module.main.proj.conv = create_convolution_layer(dc_vae.decoder.stages[4].op_list[2].context_module.main.proj.conv, 512, 512, method=args.prune_method)
        dc_vae.decoder.stages[4].op_list[1].context_module.main.proj.norm = create_norm_layer(dc_vae.decoder.stages[4].op_list[1].context_module.main.proj.norm, 512, norm_type="trms2d", eps=1e-05, method=args.prune_method)
        dc_vae.decoder.stages[4].op_list[2].context_module.main.proj.norm = create_norm_layer(dc_vae.decoder.stages[4].op_list[2].context_module.main.proj.norm, 512, norm_type="trms2d", eps=1e-05, method=args.prune_method)

        dc_vae.decoder.stages[4].op_list[1].local_module.main.inverted_conv.conv = create_convolution_layer(dc_vae.decoder.stages[4].op_list[1].local_module.main.inverted_conv.conv, 512, 4096, method=args.prune_method)
        dc_vae.decoder.stages[4].op_list[2].local_module.main.inverted_conv.conv = create_convolution_layer(dc_vae.decoder.stages[4].op_list[2].local_module.main.inverted_conv.conv, 512, 4096, method=args.prune_method)

        dc_vae.decoder.stages[4].op_list[1].local_module.main.depth_conv.conv = create_convolution_layer(dc_vae.decoder.stages[4].op_list[1].local_module.main.depth_conv.conv, 4096, 4096, 3, 4096, method=args.prune_method)
        dc_vae.decoder.stages[4].op_list[2].local_module.main.depth_conv.conv = create_convolution_layer(dc_vae.decoder.stages[4].op_list[2].local_module.main.depth_conv.conv, 4096, 4096, 3, 4096, method=args.prune_method)

        dc_vae.decoder.stages[4].op_list[1].local_module.main.point_conv.conv = create_convolution_layer(dc_vae.decoder.stages[4].op_list[1].local_module.main.point_conv.conv, 2048, 512, method=args.prune_method)
        dc_vae.decoder.stages[4].op_list[2].local_module.main.point_conv.conv = create_convolution_layer(dc_vae.decoder.stages[4].op_list[2].local_module.main.point_conv.conv, 2048, 512, method=args.prune_method)
        dc_vae.decoder.stages[4].op_list[1].local_module.main.point_conv.norm = create_norm_layer(dc_vae.decoder.stages[4].op_list[1].local_module.main.point_conv.norm, 512, norm_type="trms2d", eps=1e-05, method=args.prune_method)
        dc_vae.decoder.stages[4].op_list[2].local_module.main.point_conv.norm = create_norm_layer(dc_vae.decoder.stages[4].op_list[2].local_module.main.point_conv.norm, 512, norm_type="trms2d", eps=1e-05, method=args.prune_method)

    elif args.prune_version == "v2":
        # 获取当前 decoder 的其他部分
        # new_stages = dc_vae.decoder.stages[:4] + dc_vae.decoder.stages[5]
        # 其实是变成了f16c8,final output[1,8,32,32]

        # # trying f16c8
        # dc_vae.encoder.project_out.main.op_list[0].conv = create_convolution_layer(dc_vae.encoder.project_out.main.op_list[0].conv, 1024, 8, kernel_size=3, bias=True, method="direct")

        # dc_vae.decoder.project_in.main.conv = create_convolution_layer(dc_vae.decoder.project_in.main.conv, 8, 512, kernel_size=3, method=args.prune_method)

        # dc_vae.decoder.stages[2].op_list[0].main.conv.conv = create_convolution_layer(dc_vae.decoder.stages[2].op_list[0].main.conv.conv, 32, 512, kernel_size=3, bias=True, method=args.prune_method)

        # trying f32c32 delete stage[5]
        del dc_vae.decoder.stages[5]
        
        # dc_vae.decoder.stages[2].op_list[0].main.conv.conv = create_convolution_layer(dc_vae.decoder.stages[2].op_list[0].main.conv.conv, 512, 2048, 3, method=args.prune_method)
        # dc_vae.decoder.stages[3].op_list[0].main.conv.conv = create_convolution_layer(dc_vae.decoder.stages[3].op_list[0].main.conv.conv, 1024, 2048, 3, True, method=args.prune_method)

        # dc_vae.decoder.stages[3].op_list[1].context_module.main.qkv.conv = create_convolution_layer(dc_vae.decoder.stages[3].op_list[1].context_module.main.qkv.conv, 512, 1536, method=args.prune_method)
        # dc_vae.decoder.stages[3].op_list[2].context_module.main.qkv.conv = create_convolution_layer(dc_vae.decoder.stages[3].op_list[2].context_module.main.qkv.conv, 512, 1536, method=args.prune_method)

        # dc_vae.decoder.stages[3].op_list[1].context_module.main.proj.conv = create_convolution_layer(dc_vae.decoder.stages[3].op_list[1].context_module.main.proj.conv, 512, 512, method=args.prune_method)
        # dc_vae.decoder.stages[3].op_list[2].context_module.main.proj.conv = create_convolution_layer(dc_vae.decoder.stages[3].op_list[2].context_module.main.proj.conv, 512, 512, method=args.prune_method)
        # dc_vae.decoder.stages[3].op_list[1].context_module.main.proj.norm = create_norm_layer(dc_vae.decoder.stages[3].op_list[1].context_module.main.proj.norm, 512, norm_type="trms2d", eps=1e-05, method=args.prune_method)
        # dc_vae.decoder.stages[3].op_list[2].context_module.main.proj.norm = create_norm_layer(dc_vae.decoder.stages[3].op_list[2].context_module.main.proj.norm, 512, norm_type="trms2d", eps=1e-05, method=args.prune_method)

        # dc_vae.decoder.stages[3].op_list[1].local_module.main.inverted_conv.conv = create_convolution_layer(dc_vae.decoder.stages[3].op_list[1].local_module.main.inverted_conv.conv, 512, 4096, method=args.prune_method)
        # dc_vae.decoder.stages[3].op_list[2].local_module.main.inverted_conv.conv = create_convolution_layer(dc_vae.decoder.stages[3].op_list[2].local_module.main.inverted_conv.conv, 512, 4096, method=args.prune_method)

        # dc_vae.decoder.stages[3].op_list[1].local_module.main.depth_conv.conv = create_convolution_layer(dc_vae.decoder.stages[3].op_list[1].local_module.main.depth_conv.conv, 4096, 4096, 3, 4096, method=args.prune_method)
        # dc_vae.decoder.stages[3].op_list[2].local_module.main.depth_conv.conv = create_convolution_layer(dc_vae.decoder.stages[3].op_list[2].local_module.main.depth_conv.conv, 4096, 4096, 3, 4096, method=args.prune_method)

        # dc_vae.decoder.stages[3].op_list[1].local_module.main.point_conv.conv = create_convolution_layer(dc_vae.decoder.stages[3].op_list[1].local_module.main.point_conv.conv, 2048, 512, method=args.prune_method)
        # dc_vae.decoder.stages[3].op_list[2].local_module.main.point_conv.conv = create_convolution_layer(dc_vae.decoder.stages[3].op_list[2].local_module.main.point_conv.conv, 2048, 512, method=args.prune_method)
        # dc_vae.decoder.stages[3].op_list[1].local_module.main.point_conv.norm = create_norm_layer(dc_vae.decoder.stages[3].op_list[1].local_module.main.point_conv.norm, 512, norm_type="trms2d", eps=1e-05, method=args.prune_method)
        # dc_vae.decoder.stages[3].op_list[2].local_module.main.point_conv.norm = create_norm_layer(dc_vae.decoder.stages[3].op_list[2].local_module.main.point_conv.norm, 512, norm_type="trms2d", eps=1e-05, method=args.prune_method)

        # dc_vae.decoder.stages[4].op_list[1].context_module.main.qkv.conv = create_convolution_layer(dc_vae.decoder.stages[4].op_list[1].context_module.main.qkv.conv, 1024, 3072, method=args.prune_method)

        # dc_vae.decoder.stages[4].op_list[1].context_module.main.proj.conv = create_convolution_layer(dc_vae.decoder.stages[4].op_list[1].context_module.main.proj.conv, 1024, 1024, method=args.prune_method)

        # dc_vae.decoder.stages[4].op_list[1].context_module.main.proj.norm = create_norm_layer(dc_vae.decoder.stages[4].op_list[1].context_module.main.proj.norm, 1024, norm_type="trms2d", eps=1e-05, method=args.prune_method)

        # dc_vae.decoder.stages[4].op_list[1].local_module.main.inverted_conv.conv = create_convolution_layer(dc_vae.decoder.stages[4].op_list[1].context_module.main.proj.conv, 1024, 8192, bias=True, method=args.prune_method)

        # dc_vae.decoder.stages[4].op_list[1].local_module.main.depth_conv.conv = create_convolution_layer(dc_vae.decoder.stages[4].op_list[1].local_module.main.depth_conv.conv, 8192, 8192, 3, 8192, bias=True, method=args.prune_method)

        # dc_vae.decoder.stages[4].op_list[1].local_module.main.point_conv.conv = create_convolution_layer(dc_vae.decoder.stages[4].op_list[1].local_module.main.point_conv.conv, 4096, 1024, bias=True, method=args.prune_method)

        # dc_vae.decoder.stages[4].op_list[1].local_module.main.point_conv.norm = create_norm_layer(dc_vae.decoder.stages[4].op_list[1].local_module.main.point_conv.norm, 1024, norm_type="trms2d", eps=1e-05, method=args.prune_method)


        # Delete Stage[4] f32c32
        # del dc_vae.decoder.stages[4]


        # 替换原有的 decoder
        # dc_vae.decoder.stages = new_stages

    elif args.prune_version == "v3":
        # 对f64c128的砍掉最后一个block 比较输出结果 channel变成32
        del dc_vae.decoder.stages[5]

        dc_vae.encoder.project_out.main.op_list[0].conv = create_convolution_layer(dc_vae.encoder.project_out.main.op_list[0].conv, 2048, 32, 3, bias=True, method=args.prune_method)

        dc_vae.decoder.project_in.main.conv = create_convolution_layer(dc_vae.decoder.project_in.main.conv, 32, 1024, 3, bias=True, method=args.prune_method)

    elif args.prune_version == "v4": # f32c32 [128, 256, 256, 512, 512, 1024]
        dc_vae.decoder.stages[3].op_list[0].main.conv.conv = create_convolution_layer(dc_vae.decoder.stages[3].op_list[0].main.conv.conv, 512, 2048, 3, method=args.prune_method)
        dc_vae.decoder.stages[4].op_list[0].main.conv.conv = create_convolution_layer(dc_vae.decoder.stages[4].op_list[0].main.conv.conv, 1024, 2048, 3, True, method=args.prune_method)

        for i in range(1, 3): # i = 1, 2
            dc_vae.decoder.stages[4].op_list[i].context_module.main.qkv.conv = create_convolution_layer(dc_vae.decoder.stages[4].op_list[i].context_module.main.qkv.conv, 512, 1536, method=args.prune_method)
            dc_vae.decoder.stages[4].op_list[i].context_module.main.proj.conv = create_convolution_layer(dc_vae.decoder.stages[4].op_list[i].context_module.main.proj.conv, 512, 512, method=args.prune_method)
            dc_vae.decoder.stages[4].op_list[i].context_module.main.proj.norm = create_norm_layer(dc_vae.decoder.stages[4].op_list[i].context_module.main.proj.norm, 512, norm_type="trms2d", eps=1e-05, method=args.prune_method)
            
            dc_vae.decoder.stages[4].op_list[i].local_module.main.inverted_conv.conv = create_convolution_layer(dc_vae.decoder.stages[4].op_list[i].local_module.main.inverted_conv.conv, 512, 4096, method=args.prune_method)
            
            dc_vae.decoder.stages[4].op_list[i].local_module.main.depth_conv.conv = create_convolution_layer(dc_vae.decoder.stages[4].op_list[i].local_module.main.depth_conv.conv, 4096, 4096, 3, 4096, method=args.prune_method)
            
            dc_vae.decoder.stages[4].op_list[i].local_module.main.point_conv.conv = create_convolution_layer(dc_vae.decoder.stages[4].op_list[i].local_module.main.point_conv.conv, 2048, 512, method=args.prune_method)
            dc_vae.decoder.stages[4].op_list[i].local_module.main.point_conv.norm = create_norm_layer(dc_vae.decoder.stages[4].op_list[i].local_module.main.point_conv.norm, 512, norm_type="trms2d", eps=1e-05, method=args.prune_method)

        dc_vae.decoder.stages[1].op_list[0].main.conv.conv = create_convolution_layer(dc_vae.decoder.stages[1].op_list[0].main.conv.conv, 256, 1024, 3, method=args.prune_method)

        dc_vae.decoder.stages[2].op_list[0].main.conv.conv = create_convolution_layer(dc_vae.decoder.stages[2].op_list[0].main.conv.conv, 512, 1024, 3, bias=True, method=args.prune_method)
        
        for i in range(1, 11):
            dc_vae.decoder.stages[2].op_list[i].main.conv1.conv = create_convolution_layer(dc_vae.decoder.stages[2].op_list[i].main.conv1.conv, 256, 256, 3, bias=True, method=args.prune_method)
            dc_vae.decoder.stages[2].op_list[i].main.conv2.conv = create_convolution_layer(dc_vae.decoder.stages[2].op_list[i].main.conv2.conv, 256, 256, 3, method=args.prune_method)

            dc_vae.decoder.stages[2].op_list[i].main.conv2.norm = create_norm_layer(dc_vae.decoder.stages[2].op_list[i].main.conv2.norm, 256, norm_type="bn2d", eps=1e-05, method=args.prune_method)

            
    elif args.prune_version == "v5": # f32c32 [128, 256, 256, 512, 512, 512] to be finished!
        del dc_vae.decoder.stages[5]
        del dc_vae.encoder.stages[5]

    elif args.prune_version == "w3-v1":
        del dc_vae.decoder.stages[2].op_list[9]
        del dc_vae.decoder.stages[2].op_list[8]
        del dc_vae.decoder.stages[2].op_list[7]
        del dc_vae.decoder.stages[2].op_list[6]
        del dc_vae.decoder.stages[2].op_list[5]

        del dc_vae.decoder.stages[3].op_list[2]

    elif args.prune_version == "w3-v2":
        del dc_vae.decoder.stages[3].op_list[2]
        
        del dc_vae.decoder.stages[4].op_list[2]

    elif args.prune_version == "w4-v1":
        del dc_vae.decoder.stages[1].op_list[4]
        del dc_vae.decoder.stages[1].op_list[3]
        
        del dc_vae.decoder.stages[2].op_list[9]
        del dc_vae.decoder.stages[2].op_list[8]
        del dc_vae.decoder.stages[2].op_list[7]
        del dc_vae.decoder.stages[2].op_list[6]
        del dc_vae.decoder.stages[2].op_list[5]

    elif args.prune_version == "w4-v2":
        del dc_vae.decoder.stages[1].op_list[4]
        del dc_vae.decoder.stages[1].op_list[3]
        
        del dc_vae.decoder.stages[2].op_list[9]
        del dc_vae.decoder.stages[2].op_list[8]
        del dc_vae.decoder.stages[2].op_list[7]
        del dc_vae.decoder.stages[2].op_list[6]
        del dc_vae.decoder.stages[2].op_list[5]

        del dc_vae.decoder.stages[3].op_list[2]
        
        del dc_vae.decoder.stages[4].op_list[2]

    elif args.prune_version == "w4-v3":
        del dc_vae.decoder.stages[1].op_list[4]
        del dc_vae.decoder.stages[1].op_list[3]
        del dc_vae.decoder.stages[1].op_list[2]
        del dc_vae.decoder.stages[1].op_list[1]
        
        del dc_vae.decoder.stages[2].op_list[9]
        del dc_vae.decoder.stages[2].op_list[8]
        del dc_vae.decoder.stages[2].op_list[7]
        del dc_vae.decoder.stages[2].op_list[6]
        del dc_vae.decoder.stages[2].op_list[5]
        del dc_vae.decoder.stages[2].op_list[4]
        del dc_vae.decoder.stages[2].op_list[3]
        del dc_vae.decoder.stages[2].op_list[2]

        # del dc_vae.decoder.stages[3].op_list[2]
        
        # del dc_vae.decoder.stages[4].op_list[2]


    print(dc_vae)

    # Get the model's state_dict
    state_dict = dc_vae.state_dict()
    # state_dict = dc_vae.decoder.state_dict()
    # state_dict = dc_vae.encoder.state_dict()
    # print(dc_vae)

    # Save the modified model in safetensors format
    save_file(state_dict, args.save_path)
    print(f"Modified model saved successfully to {args.save_path}")


if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parse_args()
    modify_and_save_model(args)


