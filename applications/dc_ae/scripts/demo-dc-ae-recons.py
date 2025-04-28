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
    parser.add_argument("--benchmark", type=bool, default=False,
                        help="Whether to generate the reconstructed image of benchmark.")
    return parser.parse_args()

def reconstruct_image(args):
    """Reconstruct image using the DC-AE model."""
    # Load the pretrained model
    dc_ae = DCAE_HF.from_pretrained(args.pretrained_model).to(device).eval()
    print(dc_ae)
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
    image.save(f"{args.img_save}/demo_dc_ae_gth.png")
    x = transform(image)[None].to(device)

    # Encode the image into latent representation
    latent = dc_ae.encode(x)
    print(f"Encode Latent shape: {latent.shape}")

    # Decode the latent representation back into image
    y, y1, y2, y3 = dc_ae.decode(latent)
    print(f"Decode Tensor shape: {y.shape}")
    print(f"Decode 1 Tensor shape: {y1.shape}")
    print(f"Decode 2 Tensor shape: {y2.shape}")
    print(f"Decode 3 Tensor shape: {y3.shape}")

    # Save the reconstructed image
    save_image(y * 0.5 + 0.5, f"{args.img_save}/{args.pretrained_model.replace('/', '_')}.png")

    if args.benchmark:
        dc_ae_bmk = DCAE_HF.from_pretrained("/data2/user/jyzhang/MIT/efficientvit/applications/dc_ae/pretrained_models/dc-ae-f32c32-in-1.0").to(device).eval()
        latent_bmk = dc_ae_bmk.encode(x)
        y_bmk = dc_ae_bmk.decode(latent_bmk)

        criterion = nn.MSELoss()
        loss = criterion(y, y_bmk)
        print("The loss between y and y_bmk is ", loss)
        save_image(y_bmk * 0.5 + 0.5, f"{args.img_save}/demo_dc_ae_pruned_bmk.png")


def main():
    """Main function to parse arguments and perform image reconstruction."""
    args = parse_args()
    reconstruct_image(args)

if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main()
