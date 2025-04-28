import torch
import argparse
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from efficientvit.ae_model_zoo import DCAE_HF
from efficientvit.apps.utils.image import DMCrop
from torchprofile import profile_macs
from thop import profile


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Image Reconstruction using DC-AE model.")
    parser.add_argument("--pretrained_model", type=str, default="/data2/user/jyzhang/MIT/efficientvit/applications/dc_ae/pretrained_models/dc-ae-f32c32-in-1.0",
                        help="Path to the pretrained DC-AE model.")
    parser.add_argument("--img_dir", type=str, default="/data2/user/jyzhang/MIT/efficientvit/assets/fig/girl.png",
                        help="Path to the input image.")
    parser.add_argument("--method", type=str, default="thop",
                        help="Which MACs calculation method to use.")
    return parser.parse_args()

def test_MAC(args):
    """Reconstruct image using the DC-AE model."""
    # Load the pretrained model
    dc_ae = DCAE_HF.from_pretrained(args.pretrained_model).to(device).eval()
    # print(dc_ae)
    # benchmark model dc-ae 
    print("Load Model ", args.pretrained_model,  " successfully!")

    # Define image transformation pipeline
    transform = transforms.Compose([
        DMCrop(512),  # resolution
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize to [-1, 1]
    ])

    # Load and preprocess the image
    image = Image.open(args.img_dir)
    x = transform(image)[None].to(device)

    # Encode the image into latent representation
    latent = dc_ae.encode(x)

    if args.method == "torch":
        macs_encoder = profile_macs(dc_ae.encoder, x)
    elif args.method == "thop":
        macs_encoder, _ = profile(dc_ae.encoder, inputs=(x,))
    print(f"The MACs of encode model {args.pretrained_model} is {macs_encoder}")
    # print(f"Encode Latent shape: {latent.shape}")

    # Decode the latent representation back into image
    y, y1, y2, y3 = dc_ae.decode(latent)

    if args.method == "torch":
        macs_decoder = profile_macs(dc_ae.decoder, latent)
    elif args.method == "thop":
        macs_decoder, _ = profile(dc_ae.decoder, inputs=(latent,))
    print(f"The MACs of decode model {args.pretrained_model} is {macs_decoder}")
    macs_vae = macs_encoder + macs_decoder
    print(f"Total MACs = {macs_vae}")


    
def main():
    """Main function to parse arguments and perform image reconstruction."""
    args = parse_args()
    test_MAC(args)

if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main()



