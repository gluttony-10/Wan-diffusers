from huggingface_hub import snapshot_download, login
import os

# 1. 检查是否已登录
try:
    # 尝试无需登录的下载（如果已缓存凭据）
    snapshot_download(
        repo_id="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        local_dir="models/Wan2.2-TI2V-5B-Diffusers",
        resume_download=True
    )
    print("下载成功！")
    
except Exception as e:
    if "401" in str(e):
        # 2. 需要登录
        print("需要 Hugging Face 登录，创建新令牌（选择'Read'权限即可）")
        token = input("请输入您的 Hugging Face 访问令牌 (https://huggingface.co/settings/tokens): ")
        
        # 登录并重试
        login(token=token)
        snapshot_download(
            repo_id="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
            local_dir="models/Wan2.2-TI2V-5B-Diffusers",
            resume_download=True
        )
        print("下载成功！")
        
    else:
        print(f"下载失败: {str(e)}")