import torch

input_data = torch.randn(1, 5, 100, 100) # 批量 输入通道数 长 宽
print(input_data.shape)

c_kernel = torch.nn.Conv2d(5, 10, kernel_size=3) # 输入通道数，输出通道数，核大小
output_data = c_kernel(input_data) # 输出的长宽 ＝ 输入的长宽 - 核大小 ＋ 1
print(output_data.shape)

pool = torch.nn.MaxPool2d(kernel_size=2, stride=3) # 池化，取每个区域中的最大值
output_data = pool(output_data)
print(output_data.shape)