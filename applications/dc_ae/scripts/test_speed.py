import argparse
import time
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from efficientvit.ae_model_zoo import DCAE_HF
from efficientvit.apps.utils.image import DMCrop

def parse_args():
    parser = argparse.ArgumentParser(description="Model Timing Measurement")
    parser.add_argument("--model_path", type=str, default="/data2/user/jyzhang/MIT/efficientvit/applications/dc_ae/pretrained_models/dc-ae-f32c32-in-1.0",
                       help="Path to the pretrained model")
    parser.add_argument("--dataset_path", type=str, default="/home/jyzhang/dataset/imagenet/train",
                       help="ImageNet dataset path")
    parser.add_argument("--n_iters", type=int, default=1000,
                       help="Number of iterations for timing")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size for measurement")
    parser.add_argument("--target", type=str, default="both", choices=["encoder", "decoder", "both"],
                       help="Which part to time")
    parser.add_argument("--warm_iter", type=int, default=10,
                       help="Number of warmup iterations")
    return parser.parse_args()

class Timer:
    def __init__(self, device):
        self.device = device
        if torch.cuda.is_available():
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
    
    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            self.start_event.record()
        else:
            self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        if torch.cuda.is_available():
            self.end_event.record()
            torch.cuda.synchronize()
            self.duration = self.start_event.elapsed_time(self.end_event)
        else:
            self.duration = (time.time() - self.start_time) * 1000  # ms

def prepare_dataloader(args):
    transform = transforms.Compose([
        DMCrop(512),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    dataset = datasets.ImageFolder(args.dataset_path, transform=transform)
    subset = Subset(dataset, indices=range(args.n_iters))
    return DataLoader(subset, batch_size=args.batch_size, num_workers=4)

def measure_throughput(args, model, dataloader):
    device = next(model.parameters()).device
    timer = Timer(device)
    time_records = []
    
    count_warm = 0
    # Warmup phase
    for images, _ in dataloader:
        if count_warm >= args.warm_iter:
            break

        images = images.to(device)
        
        with torch.no_grad():
            if args.target in ["encoder", "both"]:
                with timer:
                    latent = model.encode(images)
            
            if args.target in ["decoder", "both"]:
                if "latent" not in locals():  # For decoder-only timing
                    latent = model.encode(images)
                with timer:
                    _ = model.decode(latent)
        
        print(f"Warmup Iteration {count_warm}")
        count_warm += 1
    
    # Measurement phase
    for images, _ in tqdm(dataloader, desc="Timing"):
        images = images.to(device)
        
        with torch.no_grad():
            if args.target == "encoder":
                with timer:
                    latent = model.encode(images)

            elif args.target == "decoder":
                latent = model.encode(images)
                with timer:
                    _ = model.decode(latent)

            elif args.target == "both":
                with timer:
                    latent = model.encode(images)
                    _ = model.decode(latent)

            # if args.target in ["encoder", "both"]:
            #     with timer:
            #         latent = model.encode(images)
            
            # if args.target in ["decoder", "both"]:
            #     if "latent" not in locals():  # For decoder-only timing
            #         latent = model.encode(images)
            #     with timer:
            #         _ = model.decode(latent)
        
        time_records.append(timer.duration)
    
    avg_time = sum(time_records) / len(time_records)
    print(f"Test model {args.model_path}")
    print(f"\nAverage {'+'.join(args.target.split(','))} time: {avg_time:.2f}ms")

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = DCAE_HF.from_pretrained(args.model_path).to(device).eval()
    
    # Prepare dataset
    dataloader = prepare_dataloader(args)
    
    # Run measurement
    measure_throughput(args, model, dataloader)

if __name__ == "__main__":
    main()