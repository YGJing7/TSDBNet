from mmengine.config import Config
from mmseg.registry import MODELS

config_path = '/home/gjyang/project/code/TSBANet/semantic_segmentation/configs/tsbanet/mamba_vision_tsba_head_20k_polyp-352x352_base.py'
cfg = Config.fromfile(config_path)

# 只构建主分割头
decode_head = MODELS.build(cfg.model.decode_head)

decode_params = sum(p.numel() for p in decode_head.parameters())
decode_trainable = sum(p.numel() for p in decode_head.parameters() if p.requires_grad)

print(f'decode_head params: {decode_params / 1e6:.3f} M')
print(f'decode_head trainable params: {decode_trainable / 1e6:.3f} M')