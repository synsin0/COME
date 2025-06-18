import torch


# def main():
#     # 加载检查点
#     checkpoint_path = "ckpts/dome_latest.pth"
#     checkpoint = torch.load(checkpoint_path)
#     # 假设 temp_embed 是你想要修改的参数
#     if "temp_embed" in checkpoint["state_dict"]:
#         temp_embed = checkpoint["state_dict"]["temp_embed"]
#         print(f"Original temp_embed shape: {temp_embed.shape}")
#         # 只保留前 10 个元素
#         temp_embed = temp_embed[:, :10, :]
#         print(f"Modified temp_embed shape: {temp_embed.shape}")
#         checkpoint["state_dict"]["temp_embed"] = temp_embed

#     if "temp_embed" in checkpoint["ema"]:
#         temp_embed = checkpoint["ema"]["temp_embed"]
#         print(f"Original temp_embed shape: {temp_embed.shape}")
#         # 只保留前 10 个元素
#         temp_embed = temp_embed[:, :10, :]
#         print(f"Modified temp_embed shape: {temp_embed.shape}")
#         checkpoint["ema"]["temp_embed"] = temp_embed

#     # 重新保存检查点
#     torch.save(checkpoint, "ckpts/dome_world_model.pth")


def main():
    # 加载检查点
    checkpoint_path = "ckpts/dome_latest.pth"
    checkpoint = torch.load(checkpoint_path)

    world_model_path = "work_dir/dome_controlnet/latest_world_model.pth"
    world_model = torch.load(world_model_path)

    controlnet_path = "work_dir/dome_controlnet/latest.pth"
    controlnet = torch.load(controlnet_path)


    ori_temp_embed = checkpoint["ema"]["temp_embed"]

    checkpoint["state_dict"]['pose_encoder.pose_encoder.0.bias'] - controlnet["state_dict"]['pose_encoder.pose_encoder.0.bias']
    torch.sum(torch.abs(checkpoint["ema"]['pose_encoder.pose_encoder.0.weight'] - controlnet["state_dict"]['pose_encoder.pose_encoder.0.weight']))
    torch.sum(torch.abs(checkpoint["ema"]['pose_encoder.pose_encoder.0.weight'] - world_model["state_dict"]['pose_encoder.pose_encoder.0.weight']))



if __name__ == "__main__":
    main()