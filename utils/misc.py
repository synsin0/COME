import torch
def downsample_visible_mask(input_tensor, downsample_size=8):
    B, T, L, W, H = input_tensor.shape
    input_tensor = input_tensor.view(B * T, L, W, H)
    # 定义降采样后的尺寸
    output_height = int(L/downsample_size)
    output_width = int(W/downsample_size)

    # 计算每个 pillar 的尺寸
    pillar_height = input_tensor.size(1) // output_height
    pillar_width = input_tensor.size(2) // output_width
    pillar_depth = input_tensor.size(3)

    # 将输入张量重塑为 [12, 50, 4, 50, 4, 16]
    reshaped_tensor = input_tensor.view(B*T, output_height, pillar_height, output_width, pillar_width, pillar_depth)

    # 统计每个 pillar 中 True（不可见）的数量
    true_count = reshaped_tensor.sum(dim=(2, 4, 5))

    # 计算每个 pillar 中 False（可见）的数量
    false_count = (pillar_height * pillar_width * pillar_depth) - true_count

    # 判断每个 pillar 是否应该被标记为 True（不可见）
    output_tensor = true_count > false_count

    output_tensor = output_tensor.view(B, T, output_height, output_width)

    # # 输出结果
    # print(output_tensor.shape)  # 应该是 [12, 50, 50]
    return output_tensor


# def downsample_visible_mask(x, downsample_size=8):
#     """
#     缩小高维张量，0值优先
#     参数:
#         x: 输入张量，形状为 [B, T, L, W, H]，布尔类型
#     返回:
#         缩小后的张量，形状为 [B, T, L//8, W//8, 1]，布尔类型
#     """
#     # 将布尔张量转换为float32 (0.0或1.0)
#     x_float = x.float()
#     # 获取原始形状
#     B, T, L, W, H = x.shape
    
#     # 先处理H维度 (缩小到1)
#     # 使用最大池化，只要H维度上有0，结果就是0
#     # 反转->最大池化->再反转回来
#     x_reduced_h = F.max_pool3d(
#         x_float,  # 添加虚拟维度 [B,T,L,W,H,1]
#         kernel_size=(1, 1, H),
#         stride=(1, 1, H)
#     )  # 形状变为 [B,T,L,W,1]
    
#     # 处理L和W维度 (各缩小8倍)
#     # 使用最大池化的技巧
#     x_reduced_lw = F.max_pool3d(
#         x_reduced_h.permute(0, 1, 4, 2, 3),  # 调整为 [B,T,1,L,W]
#         kernel_size=(1, downsample_size, downsample_size),
#         stride=(1, downsample_size, downsample_size)
#     ).permute(0, 1, 3, 4, 2)  # 恢复为 [B,T,L//8,W//8,1]
    
#     # 转换为布尔类型
#     result = x_reduced_lw.squeeze(-1) > 0.5
    
#     return result