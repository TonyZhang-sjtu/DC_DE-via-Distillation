"""
Training with Distillation Loss at 3 Alignment Point
and L1 + LPIPS + PatchGAN Loss between original images and student model
"""

import argparse
from tqdm import tqdm
import lpips
import math
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, Subset
from torchvision import datasets, transforms
from torchvision.utils import save_image
from safetensors.torch import save_file
from efficientvit.ae_model_zoo import DCAE_HF
from efficientvit.apps.utils.image import DMCrop
from loss_func import PatchDiscriminator
from itertools import chain

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Distillation training for student model.")
    parser.add_argument("--teacher_model_path", type=str, default="/data2/user/jyzhang/MIT/efficientvit/applications/dc_ae/pretrained_models/dc-ae-f32c32-in-1.0", required=False, help="Path to the teacher model.")
    parser.add_argument("--student_model_path", type=str, default="/data2/user/jyzhang/MIT/efficientvit/applications/dc_ae/pruned_models/dc-ae-f32c32-in-1.0-v1" ,required=False, help="Path to the student model.")
    parser.add_argument("--model_config", type=str, default="dc-ae-f32c32-in-1.0-pruned-w4-v3" ,required=True, help="Config name of the model.")
    parser.add_argument("--dataset_path", type=str, default="/home/jyzhang/dataset/imagenet/train", required=False, help="Path to the dataset (e.g., ImageNet).")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training.")
    parser.add_argument("--learning_rate_G", type=float, default=1e-4, help="Learning rate for training Generator (student model).")
    parser.add_argument("--alpha_disti", type=float, default=1.0, help="Weight for L1 Loss.")
    parser.add_argument("--alpha_img", type=float, default=0.8, help="Weight for L1 Loss.")
    parser.add_argument("--beta", type=float, default=0.1, help="Weight for LPIPS Loss.")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs for training.")
    parser.add_argument("--shallow_train", type=bool, default=False, required=False, help="Whether to train shallow layers first and full layers later.")
    parser.add_argument("--shallow_training_epochs", type=int, default=5, help="Number of epochs for shallow layers training.")

    parser.add_argument("--align", type=int, default=0, required=False, help="Latent to align with, 0 for final feature after project_out, 1 for feature before project_out, 2 for feature after Norm, 3 for feature after ReLu.")
    parser.add_argument("--train_samples", type=int, default=1281167, help="Number of image samples for training. 1281167 is the whole num of samples in imagenet")
    parser.add_argument("--pic_save_dir", type=str, default="/data2/user/jyzhang/MIT/efficientvit/applications/dc_ae/reconstruction_results", required=False, help="Path to the save sampled image.")
    parser.add_argument("--model_save_dir", type=str, default="/data2/user/jyzhang/MIT/efficientvit/applications/dc_ae/pruned_models", required=False, help="Path to the save distillated model.")
    parser.add_argument("--freeze_proj_out", type=bool, default=True, required=False, help="Whether to freeze the proj_out layer during training. It should be freezed for distillation training.")
    parser.add_argument("--freeze_encoder", type=bool, default=True, required=False, help="Whether to freeze the encoder layer during training. It should be freezed for distillation training.")
    parser.add_argument("--weight_decay_G", type=float, default=0.01, help="Weight decay for Generater AdamW optimizer.")
    parser.add_argument("--cosine_T_0_G", type=int, default=10, help="Number of iterations for the first restart for Generator.")
    parser.add_argument("--cosine_T_mult_G", type=int, default=1, help="A factor to increase T_i after each restart for Generator.")
    parser.add_argument("--eta_min_G", type=float, default=1e-6, help="Minimum learning rate for Generator.")
    
    parser.add_argument("--dynamic_loss", type=bool, default=False, required=False, help="Whether to use dynamic loss adatation strategy.")
    parser.add_argument("--division_epoch", type=int, default=10, required=False, help="Before division epoch, focus more on distillation loss, After that, focus more on image results.")
    
    parser.add_argument("--accumulate_batch", type=bool, default=False, required=False, help="Whether to batch accumulation training strategy.")
    parser.add_argument("--accumulation_steps", type=int, default=4, required=False, help="For how much batches, update optimizer once.")
    
    return parser.parse_args()


def shallow_training_fix(args, student_model):
    if args.model_config == "dc-ae-f32c32-in-1.0-pruned-w4-v3":
        for param in chain( 
                student_model.decoder.stages[2].op_list[1].parameters(), student_model.decoder.stages[3].op_list[1].parameters(), 
                student_model.decoder.stages[4].op_list[1].parameters(), student_model.decoder.stages[5].op_list[1].parameters()):
            
            param.requires_grad = False


def train_distillation(args):
    """Train the student model with distillation and L1 + LPIPS + PatchGAN Loss."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_list = []

    # Load Teacher and Student models
    teacher_model = DCAE_HF.from_pretrained(args.teacher_model_path).to(device)
    student_model = DCAE_HF.from_pretrained(args.student_model_path).to(device)

    # Freeze teacher model parameters
    for param in teacher_model.parameters():
        param.requires_grad = False

    # Freeze student decoder.project_out parameters 
    if args.freeze_proj_out == True:
        for param in student_model.decoder.project_out.parameters():
            param.requires_grad = False

    if args.freeze_encoder == True:
        for param in student_model.encoder.parameters():
            param.requires_grad = False

    # Loss Function
    l1_loss_fn = nn.L1Loss()
    lpips_loss_fn = lpips.LPIPS(net='alex').to(device)

    # Get Weight for loss functions
    alpha_disti = args.alpha_disti
    alpha_img = args.alpha_img
    beta = args.beta

    # Use AdamW as Optimizer for Generator (student model) and Discriminator (GAN model)
    optimizer_G = optim.AdamW(
        student_model.parameters(),
        lr=args.learning_rate_G,
        weight_decay=args.weight_decay_G
    )

    # Apply Learning Rate Scheduler for Generator using Cosine Annealing with Warm Restarts
    scheduler_G = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer_G,
        T_0=args.cosine_T_0_G,  # Length of the first cycle
        T_mult=args.cosine_T_mult_G,  # Expansion coefficient of the subsequent cycle
        eta_min=args.eta_min_G,        # Lower limit of learning rate
        verbose=True
    )

    # Parse the Data
    transform = transforms.Compose([
        DMCrop(512),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    train_dataset = datasets.ImageFolder(args.dataset_path, transform=transform)
    sampler = RandomSampler(train_dataset, replacement=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=4)

    # Add gradient cumulative variables
    if args.accumulate_batch == True:
        accumulation_steps = args.accumulation_steps  # example: accumulate gradient every four epoch
    elif args.accumulate_batch == False:
        accumulation_steps = 1
    accumulated_batches_G = 0

    for epoch in range(args.num_epochs):

        if args.shallow_train == True:
            if epoch < args.shallow_training_epochs:
                shallow_training_fix(args, student_model)

        if args.dynamic_loss == False:
            alpha_disti = args.alpha_disti
            alpha_img = args.alpha_img
            beta = args.beta
        elif args.dynamic_loss == True:
            if epoch < args.division_epoch:
                # For the first 10 epochs, focus on teacher-student alignment
                alpha_disti = args.alpha_disti * (1 + math.cos((math.pi / 2) * (epoch / args.division_epoch))) / 2  # decreasing alpha_disti
                alpha_img = args.alpha_img
                beta = args.beta
            else:
                # For the last 5 epochs, focus on image quality
                alpha_disti = args.alpha_disti / 2  # gradually decrease alpha_disti
                alpha_img = args.alpha_img * (1 + math.cos((math.pi / 2) * (epoch - args.division_epoch) / (args.num_epochs - args.division_epoch))) / 2  # increasing alpha_img
                beta = args.beta * (1 + math.cos((math.pi / 2) * (epoch - args.division_epoch) / (args.num_epochs - args.division_epoch))) / 2  # increasing beta

        student_model.train()
        total_loss = 0
        l1_dis_loss = 0
        l1_img_loss = 0
        lpips_val_loss = 0

        i = 0
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}", total=args.train_samples/args.batch_size, unit='img') as pbar:
            for images, _ in pbar:
                if i * args.batch_size >= args.train_samples:
                    break

                images = images.to(device)
                last_images = images

                # === Forward teacher ===
                with torch.no_grad():
                    teacher_latent = teacher_model.encode(images)
                    teacher_recon, teacher_recon_1, teacher_recon_2, teacher_recon_3 = teacher_model.decode(teacher_latent)

                # === Forward student ===
                student_latent = student_model.encode(images)
                student_recon, student_recon_1, student_recon_2, student_recon_3 = student_model.decode(student_latent)
                student_img = student_recon * 0.5 + 0.5

                # ========== Train Student Model ==========
                optimizer_G.zero_grad()
                if args.align == 1:
                    l1_dis = l1_loss_fn(student_recon_1, teacher_recon_1)
                elif args.align == 2:
                    l1_dis = l1_loss_fn(student_recon_2, teacher_recon_2)
                elif args.align == 3:
                    l1_dis = l1_loss_fn(student_recon_3, teacher_recon_3)

                # Loss for Image Results
                student_img = student_recon * 0.5 + 0.5
                l1_img = l1_loss_fn(student_img, images)
                lpips_val_img = lpips_loss_fn(student_img, images).mean()

                loss_G = alpha_disti * l1_dis + alpha_img * l1_img + beta * lpips_val_img
                loss_G.backward()

                accumulated_batches_G += 1

                if accumulated_batches_G >= accumulation_steps:
                    optimizer_G.step()
                    optimizer_G.zero_grad()
                    accumulated_batches_G = 0

                total_loss += loss_G.item()
                l1_dis_loss += l1_dis.item()
                l1_img_loss += l1_img.item()
                lpips_val_loss += lpips_val_img.item()

                i += 1

                pbar.set_postfix({
                    "L1_Dis": l1_dis.item(),
                    "L1_Img": l1_img.item(),
                    "LPIPS_Img": lpips_val_img.item(),
                    "Total": loss_G.item()
                })

        avg_loss = total_loss / i
        avg_l1_dis_loss = l1_dis_loss / args.train_samples
        avg_l1_img_loss = l1_img_loss / args.train_samples
        avg_lpips_val_loss = lpips_val_loss / args.train_samples

        loss_list.append(avg_loss)
        print(f"Epoch [{epoch+1}/{args.num_epochs}], Avg Loss: {avg_loss:.4f}")

        with open(f"{args.model_save_dir}/training_losses.txt", "a") as f:
            f.write(f"Epoch {epoch+1}/{args.num_epochs}, L1_Dis: {avg_l1_dis_loss:.4f}, L1_Img: {avg_l1_img_loss:.4f}, LPIPS_Img: {avg_lpips_val_loss:.4f}, Total: {loss_G.item():.4f}\n")
            f.write(f"Epoch {epoch+1}/{args.num_epochs}, Avg Loss: {avg_loss:.4f}\n")

        # Update Learning Rate Scheduler
        scheduler_G.step()

        # Save generated images
        save_image(last_images * 0.5 + 0.5, f"{args.pic_save_dir}/ground_truth_epoch_{epoch+1}.png")
        save_image(teacher_recon * 0.5 + 0.5, f"{args.pic_save_dir}/teacher_epoch_{epoch+1}.png")
        save_image(student_recon * 0.5 + 0.5, f"{args.pic_save_dir}/reconstructed_epoch_{epoch+1}.png")

    # Save Student Model
    state_dict = student_model.state_dict()
    save_file(state_dict, f"{args.model_save_dir}/model.safetensors")
    print("Save Student Model Successfully!")


def main():
    """Main function to handle arguments and start training."""
    args = parse_args()
    train_distillation(args)

if __name__ == "__main__":
    main()



