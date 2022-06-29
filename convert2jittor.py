from jittor.utils.pytorch_converter import convert
pytorch_code = """
import torch
torch.tensor([0.40760392, 0.45795686, 0.48501961]
        ).type_as(tensor_bgr).view(1, 3, 1, 1)
"""
jittor_code = convert(pytorch_code)
print(jittor_code)
