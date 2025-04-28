export HF_ENDPOINT=https://hf-mirror.com
cd /data2/user/jyzhang/MIT/efficientvit

python /data2/user/jyzhang/MIT/efficientvit/applications/dc_ae/scripts/dc_diff_modify_layer.py \
--prune_method direct \
--pretrained_model /data2/user/jyzhang/MIT/efficientvit/applications/dc_ae/pretrained_models/dc-ae-f32c32-in-1.0-usit-h-in-512px \
--save_path /data2/user/jyzhang/MIT/efficientvit/applications/dc_ae/pruned_models/dc-ae-f32c32-in-1.0-usit-h-in-512px-wo/model.safetensors