export HF_ENDPOINT=https://hf-mirror.com
cd /data2/user/jyzhang/MIT/efficientvit

CUDA_VISIBLE_DEVICES=1 python /data2/user/jyzhang/MIT/efficientvit/applications/dc_ae/scripts/dc-ae-para-count.py \
--pretrained_model /data2/user/jyzhang/MIT/efficientvit/applications/dc_ae/pretrained_models/dc-ae-f32c32-in-1.0 \
--section full

CUDA_VISIBLE_DEVICES=1 python /data2/user/jyzhang/MIT/efficientvit/applications/dc_ae/scripts/dc-ae-para-count.py \
--pretrained_model /data2/user/jyzhang/MIT/efficientvit/applications/dc_ae/pretrained_models/dc-ae-f32c32-in-1.0 \
--section encoder

CUDA_VISIBLE_DEVICES=1 python /data2/user/jyzhang/MIT/efficientvit/applications/dc_ae/scripts/dc-ae-para-count.py \
--pretrained_model /data2/user/jyzhang/MIT/efficientvit/applications/dc_ae/pretrained_models/dc-ae-f32c32-in-1.0 \
--section decoder
