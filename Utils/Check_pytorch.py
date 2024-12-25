"""
(utils)
这个脚本用来检测运行GCN的torch环境
"""
import torch
print(torch.__version__)                # 查看pytorch安装的版本号
print(torch.cuda.is_available())        # 查看cuda是否可用。True为可用，即是gpu版本pytorch
print(torch.cuda.get_device_name(0))    # 返回GPU型号
print(torch.cuda.device_count())        # 返回可以用的cuda（GPU）数量，0代表一个
print(torch.version.cuda)               # 查看cuda的版本