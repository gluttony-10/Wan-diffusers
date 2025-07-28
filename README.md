# Wan-diffusers

基于通义万相Wan的diffusers推理项目

一键包详见 [bilibili@十字鱼](https://space.bilibili.com/893892)

## 更新内容
250728 添加5B模型
## 安装依赖
```
git clone https://github.com/gluttony-10/Wan-diffusers
cd Wan-diffusers
conda create -n Wan python=3.10
conda activate Wan
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```
## 下载模型
```
python download.py
```
## 开始运行
```
python glut.py
```
## 参考项目
https://github.com/Wan-Video/Wan2.2

