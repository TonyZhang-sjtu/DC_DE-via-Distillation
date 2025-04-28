from efficientvit.diffusion_model_zoo import DCAE_Diffusion_HF
from efficientvit.ae_model_zoo import DCAE_HF
from safetensors.torch import save_file
import argparse
import torch
from dc_diff_modify_layer import create_convolution_layer, create_norm_layer

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Modify and save a modified DC-AE model.")
    parser.add_argument("--encoder_model", type=str, required=True, default="/data2/user/jyzhang/MIT/efficientvit/applications/dc_ae/pruned_models/dc-ae-f32c32-in-1.0-v6",
                        help="Path to the pretrained DC-AE model.")
    parser.add_argument("--save_path", type=str, required=True, default="/data2/user/jyzhang/MIT/efficientvit/applications/dc_ae/pruned_models/dc-ae-f32c32-in-1.0-usit-h-in-512px-v6/model.safetensors",
                        help="Path to save the modified DC-AE model.")
    parser.add_argument("--diffusion_model", type=str, required=True, default="/data2/user/jyzhang/MIT/efficientvit/applications/dc_ae/pruned_models/dc-ae-f32c32-in-1.0-usit-h-in-512px",
                        help="Path to Diffusion Model")
    parser.add_argument("--prune_method", type=str, required=False, default="direct",
                        help="Method to prune the model, including 'random', 'direct', 'gap'")
    parser.add_argument("--prune_version", type=str, required=True, default="v1",
                        help="Method to prune the model, including 'v1', 'v2', 'v3'")
    return parser.parse_args()

def main(args):
    # Load Layer Modified Diffusion Model
    diff_model = DCAE_Diffusion_HF.from_pretrained(args.diffusion_model).eval()

    if args.prune_version == "v1":
        
        diff_model.autoencoder.decoder.stages[3].op_list[0].main.conv.conv = create_convolution_layer(diff_model.autoencoder.decoder.stages[3].op_list[0].main.conv.conv, 512, 2048, 3, method=args.prune_method)
        diff_model.autoencoder.decoder.stages[4].op_list[0].main.conv.conv = create_convolution_layer(diff_model.autoencoder.decoder.stages[4].op_list[0].main.conv.conv, 1024, 2048, 3, True, method=args.prune_method)

        diff_model.autoencoder.decoder.stages[4].op_list[1].context_module.main.qkv.conv = create_convolution_layer(diff_model.autoencoder.decoder.stages[4].op_list[1].context_module.main.qkv.conv, 512, 1536, method=args.prune_method)
        diff_model.autoencoder.decoder.stages[4].op_list[2].context_module.main.qkv.conv = create_convolution_layer(diff_model.autoencoder.decoder.stages[4].op_list[2].context_module.main.qkv.conv, 512, 1536, method=args.prune_method)

        diff_model.autoencoder.decoder.stages[4].op_list[1].context_module.main.proj.conv = create_convolution_layer(diff_model.autoencoder.decoder.stages[4].op_list[1].context_module.main.proj.conv, 512, 512, method=args.prune_method)
        diff_model.autoencoder.decoder.stages[4].op_list[2].context_module.main.proj.conv = create_convolution_layer(diff_model.autoencoder.decoder.stages[4].op_list[2].context_module.main.proj.conv, 512, 512, method=args.prune_method)
        diff_model.autoencoder.decoder.stages[4].op_list[1].context_module.main.proj.norm = create_norm_layer(diff_model.autoencoder.decoder.stages[4].op_list[1].context_module.main.proj.norm, 512, norm_type="trms2d", eps=1e-05, method=args.prune_method)
        diff_model.autoencoder.decoder.stages[4].op_list[2].context_module.main.proj.norm = create_norm_layer(diff_model.autoencoder.decoder.stages[4].op_list[2].context_module.main.proj.norm, 512, norm_type="trms2d", eps=1e-05, method=args.prune_method)

        diff_model.autoencoder.decoder.stages[4].op_list[1].local_module.main.inverted_conv.conv = create_convolution_layer(diff_model.autoencoder.decoder.stages[4].op_list[1].local_module.main.inverted_conv.conv, 512, 4096, method=args.prune_method)
        diff_model.autoencoder.decoder.stages[4].op_list[2].local_module.main.inverted_conv.conv = create_convolution_layer(diff_model.autoencoder.decoder.stages[4].op_list[2].local_module.main.inverted_conv.conv, 512, 4096, method=args.prune_method)

        diff_model.autoencoder.decoder.stages[4].op_list[1].local_module.main.depth_conv.conv = create_convolution_layer(diff_model.autoencoder.decoder.stages[4].op_list[1].local_module.main.depth_conv.conv, 4096, 4096, 3, 4096, method=args.prune_method)
        diff_model.autoencoder.decoder.stages[4].op_list[2].local_module.main.depth_conv.conv = create_convolution_layer(diff_model.autoencoder.decoder.stages[4].op_list[2].local_module.main.depth_conv.conv, 4096, 4096, 3, 4096, method=args.prune_method)

        diff_model.autoencoder.decoder.stages[4].op_list[1].local_module.main.point_conv.conv = create_convolution_layer(diff_model.autoencoder.decoder.stages[4].op_list[1].local_module.main.point_conv.conv, 2048, 512, method=args.prune_method)
        diff_model.autoencoder.decoder.stages[4].op_list[2].local_module.main.point_conv.conv = create_convolution_layer(diff_model.autoencoder.decoder.stages[4].op_list[2].local_module.main.point_conv.conv, 2048, 512, method=args.prune_method)
        diff_model.autoencoder.decoder.stages[4].op_list[1].local_module.main.point_conv.norm = create_norm_layer(diff_model.autoencoder.decoder.stages[4].op_list[1].local_module.main.point_conv.norm, 512, norm_type="trms2d", eps=1e-05, method=args.prune_method)
        diff_model.autoencoder.decoder.stages[4].op_list[2].local_module.main.point_conv.norm = create_norm_layer(diff_model.autoencoder.decoder.stages[4].op_list[2].local_module.main.point_conv.norm, 512, norm_type="trms2d", eps=1e-05, method=args.prune_method)

    elif args.prune_version == "v2":
        # 获取当前 decoder 的其他部分
        new_stages = diff_model.autoencoder.decoder.stages[:4] + diff_model.autoencoder.decoder.stages[5]

        # 替换原有的 decoder
        diff_model.autoencoder.decoder.stages = new_stages
        

    # Load Pruned + Distillated VAE Model
    encoder_model = DCAE_HF.from_pretrained(args.encoder_model)
    print(encoder_model)

    # Replace the autoencoder part of diffusion_model with encoder_model
    diff_model.autoencoder = encoder_model

    # Get the model's state_dict
    state_dict = diff_model.state_dict()

    # Save the modified model in safetensors format
    save_file(state_dict, args.save_path)
    print(f"Modified model saved successfully to {args.save_path}")

    # print(diffusion_model)


if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parse_args()
    main(args)





