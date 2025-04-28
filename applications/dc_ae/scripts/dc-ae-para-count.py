import argparse
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from efficientvit.ae_model_zoo import DCAE_HF
from efficientvit.apps.utils.image import DMCrop

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Image Reconstruction using DC-AE model.")
    parser.add_argument("--pretrained_model", type=str, default="/data2/user/jyzhang/MIT/efficientvit/applications/dc_ae/pretrained_models/dc-ae-f32c32-in-1.0",
                        help="Path to the pretrained DC-AE model.")
    parser.add_argument("--img_dir", type=str, default="/data2/user/jyzhang/MIT/efficientvit/assets/fig/girl.png",
                        help="Path to the input image.")
    parser.add_argument("--img_save", type=str, default="/data2/user/jyzhang/MIT/efficientvit/applications/dc_ae/reconstruction_results",
                        help="Directory to save the reconstructed image.")
    parser.add_argument("--section", type=str, default="full",
                        help="The part of paras to count")
    parser.add_argument("--benchmark", type=bool, default=False,
                        help="Whether to generate the reconstructed image of benchmark.")
    return parser.parse_args()


def count_parameters(model):
    """Count the number of parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def params_count(args):
    """Reconstruct image using the DC-AE model."""
    # Load the pretrained model
    dc_ae = DCAE_HF.from_pretrained(args.pretrained_model).to(device).eval()
    # benchmark model dc-ae 
    print("Load Model ", args.pretrained_model,  " successfully!")

    if args.section == "full":
        model = dc_ae
    elif args.section == "encoder":
        model = dc_ae.encoder
    elif args.section == "decoder":
        model = dc_ae.decoder

    num_params = count_parameters(model)
    print(f"Number of {args.section} parameters in the model {args.pretrained_model} is: {num_params}")

def main():
    """Main function to parse arguments and perform image reconstruction."""
    args = parse_args()
    params_count(args)

if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main()
