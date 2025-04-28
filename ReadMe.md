# Deep Conpression AutoDecoder via Distillation

<center>Jingyuan Zhang</center>

<center>School of Electronic Information and Electrical Engineering</center>

<center>Shanghai Jiao Tong University</center>

## Abstract

This project constructs a pipeline to get light models with minimal loss via distillation. First, we prune model structure of decoder and arquire light student models. Then, we do distillation training with teacher model [dc-ae-f32c32-in-1.0](https://huggingface.co/mit-han-lab/dc-ae-f32c32-in-1.0) to reduce the gap as much as possible. During traing, we always freeze the encoder part and project out part of decoder. Furthermore, we involved tricks like AdamW optimizer, CosineAnnealingWarmRestarts Scheduler, Dynamic Loss Weight Adjustment Method, Batch Accumulation and Segment Training. Components of loss function include L1 distillation loss, L1 image loss, LPIPS loss, PatchGAN loss etc. After that, we choose FID, PSNR, SSIM and LPIPS as our evaluation matrice for image quality and MACs/inference time as indicators for speed. Finally, new light models outperform benchmark in PSNR and SSIM, the quality of generated images. Also, there is no significant difference in visualization between the generated image and teacher model.



## Environment Setup

1. In this folder, run the command below to create a new environment named "myenv", or it will create an environment named efficient by default.

``` bash
conda env create -f environment.yml -n myenv
```

2. Notice that the efficientvit package in path efficientvit/applications/dc_ae/scripts/efficientvit and efficientvit/efficientvit have been modified due to the changes in new model configurations.


## Usage

### Modify Model Structure

Code for modifying layers and pruning components is in dc_de_modify_layer.py, and run the command:

``` bash
CUDA_VISIBLE_DEVICES=3 python /data2/user/jyzhang/MIT/efficientvit/applications/dc_ae/scripts/dc_de_modify_layer.py \
--prune_method direct --prune_version w4-v3 \
--pretrained_model /data2/user/jyzhang/MIT/efficientvit/applications/dc_ae/pretrained_models/dc-ae-f32c32-in-1.0 \
--save_path /data2/user/jyzhang/MIT/efficientvit/applications/dc_ae/pruned_models/dc-ae-f32c32-in-1.0-w4-v3/model.safetensors
```

There are three prune_methods. "Direct" means if I want to take 15 out of 30 parameters, the first 15 parameters will be taken directly. "gap" method will take 15 at intervals and "random" method will initialize to normally distributed random numbers.

Parameter pretrained_model are path for pretrained teacher model and save_path are for target model. In model save path, there're two other files, config.json for model_name which has been registered in ae_model_zoo.py and dc_ae.py and training_loss.txt for records during training.


Remember changing file/model path to the local path on your device. If you want to create new models by modifying layers, remember adding corresponding model info in efficient/model/efficient/dc_ae.py and efficient/ae_model_zoo.py.


### Distillation Training




## Deep Compression AutoDecoder





## Appendix

Generally the parameters and corresponding description are as followed:

``` bash
parser.add_argument("--teacher_model_path", type=str, default="/data2/user/jyzhang/MIT/efficientvit/applications/dc_ae/pretrained_models/dc-ae-f32c32-in-1.0", required=False, help="Path to the teacher model.")
    parser.add_argument("--student_model_path", type=str, default="/data2/user/jyzhang/MIT/efficientvit/applications/dc_ae/pruned_models/dc-ae-f32c32-in-1.0-v1" ,required=False, help="Path to the student model.")
    parser.add_argument("--model_config", type=str, default="dc-ae-f32c32-in-1.0-pruned-w4-v3" ,required=True, help="Config name of the model.")
    parser.add_argument("--dataset_path", type=str, default="/home/jyzhang/dataset/imagenet/train", required=False, help="Path to the dataset (e.g., ImageNet).")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training.")
    parser.add_argument("--learning_rate_G", type=float, default=1e-4, help="Learning rate for training Generator (student model).")
    parser.add_argument("--learning_rate_D", type=float, default=1e-4, help="Learning rate for training Discriminator.")
    parser.add_argument("--alpha_disti", type=float, default=1.0, help="Weight for L1 Loss.")
    parser.add_argument("--alpha_img", type=float, default=0.8, help="Weight for L1 Loss.")
    parser.add_argument("--beta", type=float, default=0.1, help="Weight for LPIPS Loss.")
    parser.add_argument("--gamma", type=float, default=0.05, help="Weight for PatchGAN Loss.")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs for training.")
    parser.add_argument("--shallow_train", type=bool, default=False, required=False, help="Whether to train shallow layers first and full layers later.")
    parser.add_argument("--shallow_training_epochs", type=int, default=5, help="Number of epochs for shallow layers training.")

    parser.add_argument("--gan_ratio", type=int, default=10000, help="Number of epochs for training.")
    parser.add_argument("--align", type=int, default=0, required=False, help="Latent to align with, 0 for final feature after project_out, 1 for feature before project_out, 2 for feature after Norm, 3 for feature after ReLu.")
    parser.add_argument("--train_samples", type=int, default=1281167, help="Number of image samples for training. 1281167 is the whole num of samples in imagenet")
    parser.add_argument("--pic_save_dir", type=str, default="/data2/user/jyzhang/MIT/efficientvit/applications/dc_ae/reconstruction_results", required=False, help="Path to the save sampled image.")
    parser.add_argument("--model_save_dir", type=str, default="/data2/user/jyzhang/MIT/efficientvit/applications/dc_ae/pruned_models", required=False, help="Path to the save distillated model.")
    parser.add_argument("--freeze_proj_out", type=bool, default=True, required=False, help="Whether to freeze the proj_out layer during training. It should be freezed for distillation training.")
    parser.add_argument("--freeze_encoder", type=bool, default=True, required=False, help="Whether to freeze the encoder layer during training. It should be freezed for distillation training.")
    parser.add_argument("--weight_decay_G", type=float, default=0.01, help="Weight decay for Generater AdamW optimizer.")
    parser.add_argument("--weight_decay_D", type=float, default=0.01, help="Weight decay for Discriminator AdamW optimizer.")
    parser.add_argument("--cosine_T_0_G", type=int, default=10, help="Number of iterations for the first restart for Generator.")
    parser.add_argument("--cosine_T_0_D", type=int, default=10, help="Number of iterations for the first restart for Discriminator.")
    parser.add_argument("--cosine_T_mult_G", type=int, default=1, help="A factor to increase T_i after each restart for Generator.")
    parser.add_argument("--cosine_T_mult_D", type=int, default=1, help="A factor to increase T_i after each restart for Discriminator.")
    parser.add_argument("--eta_min_G", type=float, default=1e-6, help="Minimum learning rate for Generator.")
    parser.add_argument("--eta_min_D", type=float, default=1e-6, help="Minimum learning rate for Discriminator.")
    
    parser.add_argument("--dynamic_loss", type=bool, default=False, required=False, help="Whether to use dynamic loss adatation strategy.")
    parser.add_argument("--division_epoch", type=int, default=10, required=False, help="Before division epoch, focus more on distillation loss, After that, focus more on image results.")
    
    parser.add_argument("--accumulate_batch", type=bool, default=False, required=False, help="Whether to batch accumulation training strategy.")
    parser.add_argument("--accumulation_steps", type=int, default=4, required=False, help="For how much batches, update optimizer once.")
```

