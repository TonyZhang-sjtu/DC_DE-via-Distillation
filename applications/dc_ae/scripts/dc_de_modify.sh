export HF_ENDPOINT=https://hf-mirror.com
cd /data2/user/jyzhang/MIT/efficientvit

CUDA_VISIBLE_DEVICES=3 python /data2/user/jyzhang/MIT/efficientvit/applications/dc_ae/scripts/dc_de_modify_layer.py \
--prune_method direct --prune_version w4-v3 \
--pretrained_model /data2/user/jyzhang/MIT/efficientvit/applications/dc_ae/pretrained_models/dc-ae-f32c32-in-1.0 \
--save_path /data2/user/jyzhang/MIT/efficientvit/applications/dc_ae/pruned_models/dc-ae-f32c32-in-1.0-w4-v3/model.safetensors


