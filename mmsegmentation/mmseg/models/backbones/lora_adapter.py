# lora_adapter.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
# from einops import rearrange

class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=8, alpha=16, dropout=0.0):
        super().__init__()
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x, original_weight, original_bias=None):
        result = F.linear(x, original_weight, original_bias)
        lora_result = F.linear(self.dropout(x), self.lora_B @ self.lora_A) * self.scaling
        return result + lora_result


class MambaVisionLoRAAdapter(nn.Module):
    def __init__(self, mamba_vision_model, rank=8, alpha=16, lora_dropout=0.0, target_modules=None):
        super().__init__()
        self.base_model = mamba_vision_model
        self.lora_layers = nn.ModuleDict()
        
        # 默认只适配线性层
        if target_modules is None:
            target_modules = ['qkv', 'proj', 'fc', 'head', 'in_proj', 'out_proj', 'dt_proj', 'x_proj']
        
        self._inject_lora_layers(rank, alpha, lora_dropout, target_modules)
        
        # 冻结基础模型参数
        self._freeze_base_model()

    def _inject_lora_layers(self, rank, alpha, dropout, target_modules):
        """为匹配的模块注入LoRA层"""
        for name, module in self.base_model.named_modules():
            if isinstance(module, nn.Linear) and any(target in name for target in target_modules):
                # 创建LoRA适配器
                lora_layer = LoRALayer(
                    module.in_features,
                    module.out_features,
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout
                )
                
                # 保存原始权重
                lora_layer.register_buffer('original_weight', module.weight.data.clone())
                if module.bias is not None:
                    lora_layer.register_buffer('original_bias', module.bias.data.clone())
                else:
                    lora_layer.register_buffer('original_bias', None)
                
                # 替换前向传播
                module.forward = self._create_lora_forward(module, lora_layer)
                
                # 存储LoRA层以便管理
                self.lora_layers[name] = lora_layer

    def _create_lora_forward(self, original_layer, lora_layer):
        """创建带有LoRA的前向传播函数"""
        def lora_forward(x):
            return lora_layer(x, original_layer.weight, original_layer.bias)
        return lora_forward

    def _freeze_base_model(self):
        """冻结基础模型的所有参数"""
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # LoRA层参数保持可训练
        for param in self.lora_layers.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.base_model(x)

    def get_trainable_parameters(self):
        """获取所有可训练参数（仅LoRA层）"""
        return list(self.lora_layers.parameters())

    def save_lora_weights(self, path):
        """保存LoRA权重"""
        state_dict = {
            'lora_layers': self.lora_layers.state_dict(),
            'config': {
                'rank': self.lora_layers.lora_A.shape[0],
                'alpha': self.lora_layers.scaling * self.lora_layers.lora_A.shape[0]
            }
        }
        torch.save(state_dict, path)

    def load_lora_weights(self, path):
        """加载LoRA权重"""
        checkpoint = torch.load(path, map_location='cpu')
        self.lora_layers.load_state_dict(checkpoint['lora_layers'])