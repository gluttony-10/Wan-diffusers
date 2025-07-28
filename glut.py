import torch
import numpy as np
from diffusers import AutoencoderKLWan, WanImageToVideoPipeline, WanTransformer3DModel, GGUFQuantizationConfig, WanPipeline
from diffusers.utils import export_to_video, load_image
from transformers import CLIPVisionModel, UMT5EncoderModel
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
import argparse
import torch.nn.functional as F
from diffusers.hooks import apply_group_offloading
from typing import Callable, List, Optional, Tuple, Union
from diffusers.models.attention_processor import Attention
import gradio as gr
import psutil
import datetime
import os
import random


parser = argparse.ArgumentParser() 
parser.add_argument("--server_name", type=str, default="127.0.0.1", help="IP地址，局域网访问改为0.0.0.0")
parser.add_argument("--server_port", type=int, default=7891, help="使用端口")
parser.add_argument("--share", action="store_true", help="是否启用gradio共享")
parser.add_argument("--mcp_server", action="store_true", help="是否启用mcp服务")
parser.add_argument('--vram', type=str, default='low', choices=['low', 'high'], help='显存模式')
parser.add_argument('--lora', type=str, default="None", help='lora模型路径')
args = parser.parse_args()

print(" 启动中，请耐心等待 bilibili@十字鱼 https://space.bilibili.com/893892")
print(f'\033[32mPytorch版本：{torch.__version__}\033[0m')
if torch.cuda.is_available():
    device = "cuda" 
    print(f'\033[32m显卡型号：{torch.cuda.get_device_name()}\033[0m')
    total_vram_in_gb = torch.cuda.get_device_properties(0).total_memory / 1073741824
    print(f'\033[32m显存大小：{total_vram_in_gb:.2f}GB\033[0m')
    mem = psutil.virtual_memory()
    print(f'\033[32m内存大小：{mem.total/1073741824:.2f}GB\033[0m')
    if torch.cuda.get_device_capability()[0] >= 8:
        print(f'\033[32m支持BF16\033[0m')
        dtype = torch.bfloat16
    else:
        print(f'\033[32m不支持BF16，仅支持FP16\033[0m')
        dtype = torch.float16
else:
    print(f'\033[32mCUDA不可用，请检查\033[0m')
    device = "cpu"


MAX_SEED = np.iinfo(np.int32).max

os.makedirs("outputs", exist_ok=True)

onload_device = torch.device("cuda")
offload_device = torch.device("cpu")


def generate(
    image_input,
    prompt,
    negative_prompt,
    steps,
    nf,
    height, 
    width, 
    seed_param,
):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if seed_param<=-1:
        seed = random.randint(0, MAX_SEED)
    else:
        seed = seed_param

    if image_input is None:
        model_id = "models/Wan2.2-TI2V-5B-Diffusers"
        pipe = WanPipeline.from_pretrained(
            model_id, 
            torch_dtype=dtype
        )
        if args.lora!="None":
            pipe.load_lora_weights(args.lora)
            print(f"加载{args.lora}")
        if args.vram=="high":
            pipe.vae.enable_slicing()
            pipe.enable_model_cpu_offload()
        else:
            apply_group_offloading(pipe.text_encoder, onload_device=onload_device, offload_device=offload_device, offload_type="block_level", num_blocks_per_group=2)
            apply_group_offloading(pipe.transformer, onload_device=onload_device, offload_device=offload_device, offload_type="leaf_level")
            apply_group_offloading(pipe.vae, onload_device=onload_device, offload_device=offload_device, offload_type="leaf_level")
        output = pipe(
            prompt=prompt, 
            negative_prompt=negative_prompt, 
            height=height, 
            width=width, 
            num_frames=nf*24+1, 
            guidance_scale=5.0,
            num_inference_steps=steps,
            generator=torch.Generator().manual_seed(seed)
        ).frames[0]
    else:
        model_id = "models/Wan2.2-I2V-A14B-Diffusers"
        transformer = WanTransformer3DModel.from_single_file(
            f"{model_id}/wan2.2_i2v_high_noise_14B_Q8_0.gguf",
            config=f"{model_id}/transformer/config.json",
            quantization_config=GGUFQuantizationConfig(compute_dtype=dtype),
            torch_dtype=dtype,
        )
        transformer_2 = WanTransformer3DModel.from_single_file(
            f"{model_id}/wan2.2_i2v_low_noise_14B_Q8_0.gguf",
            config=f"{model_id}/transformer_2/config.json",
            quantization_config=GGUFQuantizationConfig(compute_dtype=dtype),
            torch_dtype=dtype,
        )
        pipe = WanImageToVideoPipeline.from_pretrained(
            model_id, 
            transformer=transformer,
            transformer_2=transformer_2,
            torch_dtype=dtype
        )
        if args.lora!="None":
            pipe.load_lora_weights(args.lora)
            print(f"加载{args.lora}")
        if args.vram=="high":
            pipe.vae.enable_slicing()
            pipe.enable_model_cpu_offload()
        else:
            apply_group_offloading(pipe.text_encoder, onload_device=onload_device, offload_device=offload_device, offload_type="block_level", num_blocks_per_group=2)
            apply_group_offloading(pipe.transformer, onload_device=onload_device, offload_device=offload_device, offload_type="leaf_level")
            apply_group_offloading(pipe.transformer_2, onload_device=onload_device, offload_device=offload_device, offload_type="leaf_level")
            apply_group_offloading(pipe.vae, onload_device=onload_device, offload_device=offload_device, offload_type="leaf_level")
        image = load_image(image_input)
        image = image.resize((width, height))
        output = pipe(
            image=image, 
            prompt=prompt, 
            negative_prompt=negative_prompt, 
            height=height, 
            width=width, 
            num_frames=nf*24+1, 
            guidance_scale=3.5,
            num_inference_steps=steps,
            generator=torch.Generator().manual_seed(seed)
        ).frames[0]
    export_to_video(output, f"outputs/{timestamp}.mp4", fps=24)
    return f"outputs/{timestamp}.mp4", seed

with gr.Blocks(theme=gr.themes.Base()) as demo:
    gr.Markdown("""
            <div>
                <h2 style="font-size: 30px;text-align: center;">Wan2.2</h2>
            </div>
            <div style="text-align: center;">
                十字鱼
                <a href="https://space.bilibili.com/893892">🌐bilibili</a> 
                |Wan2.2
                <a href="https://github.com/Wan-Video/Wan2.2">🌐github</a> 
            </div>
            <div style="text-align: center; font-weight: bold; color: red;">
                ⚠️ 该演示仅供学术研究和体验使用。
            </div>
            """)
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(label="输入图像", type="filepath", height=480)
            prompt = gr.Textbox(label="提示词（不超过200字）", value="Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.")
            negative_prompt = gr.Textbox(label="负面提示词", value="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走")
            steps = gr.Slider(label="采样步数", minimum=1, maximum=100, step=1, value=20)
            nf = gr.Slider(label="生成时长（秒）", minimum=3, maximum=10, step=1, value=5)
            height = gr.Slider(label="高度", minimum=256, maximum=2560, step=32, value=704)
            width = gr.Slider(label="宽度", minimum=256, maximum=2560, step=32, value=1280)
            seed_param = gr.Number(label="种子，请输入正整数，-1为随机", value=-1)
            generate_button = gr.Button("🎬 开始生成", variant='primary')
        with gr.Column():
            video_output = gr.Video(label="Generated Video")
            seed_output = gr.Textbox(label="种子")

    gr.on(
        triggers=[generate_button.click, prompt.submit, negative_prompt.submit],
        fn = generate,
        inputs = [
            image_input,
            prompt,
            negative_prompt,
            steps,
            nf,
            height, 
            width,
            seed_param,
        ],
        outputs = [video_output, seed_output]
    )

if __name__ == "__main__": 
    demo.launch(
        server_name=args.server_name, 
        server_port=args.server_port,
        share=args.share, 
        mcp_server=args.mcp_server,
        inbrowser=True,
    )