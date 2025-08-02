import torch
import numpy as np
from diffusers import AutoencoderKLWan, WanImageToVideoPipeline, WanTransformer3DModel, GGUFQuantizationConfig, WanPipeline, ModularPipeline
from diffusers.utils import export_to_video, load_image
from transformers import CLIPVisionModel, UMT5EncoderModel, UMT5EncoderModel
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
import argparse
import torch.nn.functional as F
from diffusers.hooks import apply_group_offloading, apply_first_block_cache, FirstBlockCacheConfig
from typing import Callable, List, Optional, Tuple, Union
from diffusers.models.attention_processor import Attention
import gradio as gr
import psutil
import datetime
import os
import random
try:
    from sageattention import sageattn
except ImportError:
    print("未安装sageattention")


parser = argparse.ArgumentParser() 
parser.add_argument("--server_name", type=str, default="127.0.0.1", help="IP地址，局域网访问改为0.0.0.0")
parser.add_argument("--server_port", type=int, default=7891, help="使用端口")
parser.add_argument("--share", action="store_true", help="是否启用gradio共享")
parser.add_argument("--mcp_server", action="store_true", help="是否启用mcp服务")
parser.add_argument('--vram', type=str, default='low', choices=['low', 'high'], help='显存模式')
parser.add_argument('--lora', type=str, default="None", help='lora模型路径')
parser.add_argument("--afba", type=float, default=0.1, help="第一块缓存加速，0就是不启用")
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
        dtype = torch.float16
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
pipe=None
model=None


def generate(
    image_input,
    prompt,
    negative_prompt,
    steps,
    nf,
    height, 
    width, 
    seed_param,
    last_image,
):
    global pipe, model
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_id = "models/Wan2.2-I2V-A14B-Diffusers"
    if seed_param<0:
        seed = random.randint(0, MAX_SEED)
    else:
        seed = seed_param

    if image_input is None:
        if pipe==None or model!="A14B_t2v":
            model="A14B_t2v"
            transformer = WanTransformer3DModel.from_single_file(
                f"{model_id}/Wan2.2-T2V-A14B-HighNoise-Q4_K_M.gguf",
                config=f"{model_id}/transformer/config2.json",
                quantization_config=GGUFQuantizationConfig(compute_dtype=dtype),
                torch_dtype=dtype,
            )
            transformer_2 = WanTransformer3DModel.from_single_file(
                f"{model_id}/Wan2.2-T2V-A14B-LowNoise-Q4_K_M.gguf",
                config=f"{model_id}/transformer_2/config2.json",
                quantization_config=GGUFQuantizationConfig(compute_dtype=dtype),
                torch_dtype=dtype,
            )
            pipe = WanPipeline.from_pretrained(
                model_id, 
                transformer=transformer,
                transformer_2=transformer_2,
                torch_dtype=dtype
            )
            pipe.load_lora_weights(f"{model_id}/lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank64_bf16.safetensors")
            if args.lora!="None":
                pipe.load_lora_weights(args.lora)
                print(f"加载{args.lora}")
            if args.vram=="high":
                pipe.vae.enable_slicing()
                pipe.enable_model_cpu_offload()
            else:
                apply_group_offloading(pipe.text_encoder, onload_device=onload_device, offload_device=offload_device, offload_type="leaf_level")
                apply_group_offloading(pipe.transformer, onload_device=onload_device, offload_device=offload_device, offload_type="leaf_level")
                apply_group_offloading(pipe.transformer_2, onload_device=onload_device, offload_device=offload_device, offload_type="leaf_level")
                apply_group_offloading(pipe.vae, onload_device=onload_device, offload_device=offload_device, offload_type="leaf_level")
            if args.afba>0:
                pipe.transformer_2.enable_cache(FirstBlockCacheConfig(threshold=args.afba))
                print("开启第一块缓存")
        output = pipe(
            prompt=prompt, 
            negative_prompt=negative_prompt, 
            height=height, 
            width=width, 
            num_frames=nf*16+1, 
            guidance_scale=4.0,
            guidance_scale_2=3.0,
            num_inference_steps=steps,
            generator=torch.Generator().manual_seed(seed)
        ).frames[0]
        export_to_video(output, f"outputs/{timestamp}.mp4", fps=16)
        return f"outputs/{timestamp}.mp4", seed
    else:
        if pipe==None or model!="A14B_i2v":
            model="A14B_i2v"
            transformer = WanTransformer3DModel.from_single_file(
                f"{model_id}/wan2.2_i2v_high_noise_14B_Q4_K_M.gguf",
                config=f"{model_id}/transformer/config.json",
                quantization_config=GGUFQuantizationConfig(compute_dtype=dtype),
                torch_dtype=dtype,
            )
            transformer_2 = WanTransformer3DModel.from_single_file(
                f"{model_id}/wan2.2_i2v_low_noise_14B_Q4_K_M.gguf",
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
            pipe.load_lora_weights(f"{model_id}/lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank64_bf16.safetensors")
            if args.lora!="None":
                pipe.load_lora_weights(args.lora)
                print(f"加载{args.lora}")
            if args.vram=="high":
                pipe.vae.enable_slicing()
                pipe.enable_model_cpu_offload()
            else:
                apply_group_offloading(pipe.text_encoder, onload_device=onload_device, offload_device=offload_device, offload_type="leaf_level")
                apply_group_offloading(pipe.transformer, onload_device=onload_device, offload_device=offload_device, offload_type="leaf_level")
                apply_group_offloading(pipe.transformer_2, onload_device=onload_device, offload_device=offload_device, offload_type="leaf_level")
                apply_group_offloading(pipe.vae, onload_device=onload_device, offload_device=offload_device, offload_type="leaf_level")
            if args.afba>0:
                pipe.transformer_2.enable_cache(FirstBlockCacheConfig(threshold=args.afba))
                print("开启第一块缓存")
        image = load_image(image_input)
        image = image.resize((width, height))
        if last_image is not None:
            last_image = load_image(last_image)
            last_image = last_image.resize((width, height))
        output = pipe(
            image=image, 
            prompt=prompt, 
            negative_prompt=negative_prompt, 
            height=height, 
            width=width, 
            num_frames=nf*16+1, 
            guidance_scale=3.5,
            num_inference_steps=steps,
            generator=torch.Generator().manual_seed(seed),
            last_image=last_image if last_image is not None else None,
        ).frames[0]
        export_to_video(output, f"outputs/{timestamp}.mp4", fps=16)
        return f"outputs/{timestamp}.mp4", seed


def generate_5b(
    image_input,
    prompt,
    negative_prompt,
    steps,
    nf,
    height, 
    width, 
    seed_param,
    last_image,
):
    global pipe, model
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_id = "models/Wan2.2-TI2V-5B-Diffusers"
    if seed_param<0:
        seed = random.randint(0, MAX_SEED)
    else:
        seed = seed_param

    if image_input is None:
        if pipe==None or model!="5b_t2v":
            model="5b_t2v"
            transformer = WanTransformer3DModel.from_single_file(
                f"{model_id}/Wan2.2-TI2V-5B-Q8_0.gguf",
                config=f"{model_id}/transformer/config.json",
                quantization_config=GGUFQuantizationConfig(compute_dtype=dtype),
                torch_dtype=dtype,
            )
            text_encoder = UMT5EncoderModel.from_pretrained(
                "models/Wan2.2-I2V-A14B-Diffusers",
                subfolder="text_encoder",
                torch_dtype=dtype,
            )
            pipe = WanPipeline.from_pretrained(
                model_id, 
                transformer=transformer,
                text_encoder=text_encoder,
                torch_dtype=dtype
            )
            if args.vram=="high":
                pipe.vae.enable_slicing()
                pipe.enable_model_cpu_offload()
            else:
                apply_group_offloading(pipe.text_encoder, onload_device=onload_device, offload_device=offload_device, offload_type="leaf_level")
                apply_group_offloading(pipe.transformer, onload_device=onload_device, offload_device=offload_device, offload_type="leaf_level")
                apply_group_offloading(pipe.vae, onload_device=onload_device, offload_device=offload_device, offload_type="leaf_level")
            if args.afba>0:
                pipe.transformer.enable_cache(FirstBlockCacheConfig(threshold=args.afba))
                print("开启第一块缓存")
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
        export_to_video(output, f"outputs/{timestamp}.mp4", fps=24)
        return f"outputs/{timestamp}.mp4", seed
    else:
        if pipe==None or model!="5b_i2v":
            model="5b_i2v"
            transformer = WanTransformer3DModel.from_single_file(
                f"{model_id}/Wan2.2-TI2V-5B-Q8_0.gguf",
                config=f"{model_id}/transformer/config.json",
                quantization_config=GGUFQuantizationConfig(compute_dtype=dtype),
                torch_dtype=dtype,
            )
            text_encoder = UMT5EncoderModel.from_pretrained(
                "models/Wan2.2-I2V-A14B-Diffusers",
                subfolder="text_encoder",
                torch_dtype=dtype,
            )
            pipe = WanImageToVideoPipeline.from_pretrained(
                model_id, 
                transformer=transformer,
                text_encoder=text_encoder,
                torch_dtype=dtype
            )
            #pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=3.0)
            if args.vram=="high":
                pipe.vae.enable_slicing()
                pipe.enable_model_cpu_offload()
            else:
                apply_group_offloading(pipe.text_encoder, onload_device=onload_device, offload_device=offload_device, offload_type="leaf_level")
                apply_group_offloading(pipe.transformer, onload_device=onload_device, offload_device=offload_device, offload_type="leaf_level")
                apply_group_offloading(pipe.vae, onload_device=onload_device, offload_device=offload_device, offload_type="leaf_level")
            if args.afba>0:
                pipe.transformer.enable_cache(FirstBlockCacheConfig(threshold=args.afba))
                print("开启第一块缓存")
            #image_processor = ModularPipeline.from_pretrained("models/WanImageProcessor", trust_remote_code=True)
        #image = image_processor(image=image_input, max_area=width*height, output="processed_image")
        image = load_image(image_input)
        image = image.resize((width, height))
        if last_image is not None:
            last_image = load_image(last_image)
            last_image = last_image.resize((width, height))
        output = pipe(
            image=image, 
            prompt=prompt, 
            negative_prompt=negative_prompt, 
            height=height, 
            width=width, 
            num_frames=nf*24+1, 
            guidance_scale=5.0,
            num_inference_steps=steps,
            generator=torch.Generator().manual_seed(seed),
            last_image=last_image if last_image is not None else None,
        ).frames[0]
        export_to_video(output, f"outputs/{timestamp}.mp4", fps=24)
        return f"outputs/{timestamp}.mp4", seed
    

with gr.Blocks(theme=gr.themes.Base()) as demo:
    gr.Markdown("""
            <div>
                <h2 style="font-size: 30px;text-align: center;">Wan-diffusers</h2>
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
    with gr.TabItem("Wan2.2 A14B"):
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(label="输入图像（上传图像是i2v模型，不上传图像是t2v模型）", type="filepath", height=400)
                with gr.Accordion("首尾帧", open=False):
                    last_image = gr.Image(label="首尾帧中的尾帧", type="filepath", height=400)
                prompt = gr.Textbox(label="提示词（不超过200字）", value="Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.")
                negative_prompt = gr.Textbox(label="负面提示词", value="")
                steps = gr.Slider(label="采样步数", minimum=1, maximum=100, step=1, value=12)
                nf = gr.Slider(label="生成时长（秒），默认帧率16", minimum=0, maximum=10, step=1, value=5)
                height = gr.Slider(label="高度", minimum=256, maximum=2560, step=32, value=480)
                width = gr.Slider(label="宽度", minimum=256, maximum=2560, step=32, value=832)
                seed_param = gr.Number(label="种子，请输入正整数，-1为随机", value=-1)
                generate_button = gr.Button("🎬 开始生成", variant='primary')
            with gr.Column():
                video_output = gr.Video(label="Generated Video")
                seed_output = gr.Textbox(label="种子")
    with gr.TabItem("Wan2.2 5B"):
        with gr.Row():
            with gr.Column():
                image_input_5b = gr.Image(label="输入图像（上传图像是i2v模式，不上传图像是t2v模式）", type="filepath", height=400)
                with gr.Accordion("首尾帧", open=False):
                    last_image_5b = gr.Image(label="首尾帧中的尾帧", type="filepath", height=400)
                prompt_5b = gr.Textbox(label="提示词（不超过200字）", value="Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.")
                negative_prompt_5b = gr.Textbox(label="负面提示词", value="")
                steps_5b = gr.Slider(label="采样步数", minimum=1, maximum=100, step=1, value=40)
                nf_5b = gr.Slider(label="生成时长（秒），默认帧率24", minimum=0, maximum=10, step=1, value=5)
                height_5b = gr.Slider(label="高度", minimum=256, maximum=2560, step=32, value=704)
                width_5b = gr.Slider(label="宽度", minimum=256, maximum=2560, step=32, value=1280)
                seed_param_5b = gr.Number(label="种子，请输入正整数，-1为随机", value=-1)
                generate_button_5b = gr.Button("🎬 开始生成", variant='primary')
            with gr.Column():
                video_output_5b = gr.Video(label="Generated Video")
                seed_output_5b = gr.Textbox(label="种子")

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
            last_image,
        ],
        outputs = [video_output, seed_output]
    )
    gr.on(
        triggers=[generate_button_5b.click, prompt_5b.submit, negative_prompt_5b.submit],
        fn = generate_5b,
        inputs = [
            image_input_5b,
            prompt_5b,
            negative_prompt_5b,
            steps_5b,
            nf_5b,
            height_5b, 
            width_5b,
            seed_param_5b,
            last_image_5b,
        ],
        outputs = [video_output_5b, seed_output_5b]
    )

if __name__ == "__main__": 
    demo.launch(
        server_name=args.server_name, 
        server_port=args.server_port,
        share=args.share, 
        mcp_server=args.mcp_server,
        inbrowser=True,
    )