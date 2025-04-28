from efficientvit.ae_model_zoo import DCAE_HF
from efficientvit.diffusion_model_zoo import DCAE_Diffusion_HF
from efficientvit.ae_model_zoo import DCAE_HF

# 加载模型
model = DCAE_HF.from_pretrained("mit-han-lab/dc-ae-f128c512-in-1.0")

# 保存模型到本地
model.save_pretrained("/data2/user/jyzhang/MIT/efficientvit/applications/dc_ae/pretrained_models/dc-ae-f128c512-in-1.0")
