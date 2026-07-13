# msswift-docker

ms-swift 训练环境的一键 Docker 部署。**镜像只提供环境**，ms-swift 源码 clone 在宿主机、挂载进容器，
在宿主机正常路径下改代码，容器内立即生效（editable install），无需重启容器或重新 build 镜像。

## 版本说明

| 组件 | 版本 | 说明 |
|---|---|---|
| **ms-swift** | **v4.4.1** | 当前最新稳定 release（2026-07-13），`start.sh` 会自动 clone 该 tag |
| 基础镜像 | `pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel` | Python 3.11，驱动 ≥ 550（CUDA 12.8+）即可运行 |
| flash-attn | 2.8.3 | ms-swift v4.4.1 官方推荐版本，预编译 wheel，支持 H100/H800/H20 (sm_90) |
| deepspeed | <0.19 | 按 ms-swift 官方 `install_all.sh` 的约束 |
| vllm | 0.10.0（默认不装） | 仅 GRPO / vLLM 推理加速需要，见下方"可选组件" |

> 适配硬件：为 Hopper 架构（H100/H800/H20）+ 驱动 575.x / CUDA 12.9 环境配置，
> A100/A800、4090 等 Ampere/Ada 卡同样可用。

## 前置条件

目标服务器需要 Docker（≥19.03，支持 `--gpus`）+ nvidia-container-toolkit。
`start.sh` 会自动检测 docker compose v2：有则用 compose，没有则回退到原生 `docker run`，两者行为一致。

验证 GPU 容器运行时：

```bash
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

如果报错，安装 nvidia-container-toolkit（Ubuntu）：

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -sL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker
```

## 一键启动

```bash
git clone git@github.com:yanghlll/msswift-docker.git
cd msswift-docker
bash start.sh
```

`start.sh` 自动完成：clone ms-swift v4.4.1 源码到 `./ms-swift` → build 镜像 → 后台启动容器 → 进入容器 shell。

退出 shell 后容器仍在后台运行，再次进入：

```bash
docker exec -it msswift bash
```

## 目录结构与改代码

```
msswift-docker/
├── ms-swift/     # ms-swift 源码（宿主机路径，直接编辑）→ 容器内 /workspace/ms-swift
├── workdir/      # 你的训练脚本 / 输出                   → 容器内 /workspace/workdir
├── Dockerfile
├── docker-compose.yml
├── entrypoint.sh # 容器启动时自动 pip install -e /workspace/ms-swift --no-deps
└── start.sh
```

容器启动时 entrypoint 会把挂载进来的源码以 **editable** 方式安装，
所以在宿主机上编辑 `./ms-swift/` 下的任何 Python 文件，容器内下一次运行即生效。
模型缓存挂载在宿主机 `~/.cache/huggingface` 和 `~/.cache/modelscope`，删除容器不丢模型。

数据盘按需在 `docker-compose.yml` 的 `volumes` 中添加挂载。

## 最简测试

进入容器后，按顺序执行以下两步验证环境。

**第 1 步：验证安装与 GPU**

```bash
python -c "import swift, torch; print('swift', swift.__version__); print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

预期输出 `swift 4.4.1`、`cuda True NVIDIA H100 ...`。

**第 2 步：最小 LoRA 微调（0.5B 模型 + 500 条样本，单卡几分钟跑完）**

```bash
CUDA_VISIBLE_DEVICES=0 swift sft \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --train_type lora \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#500' \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --learning_rate 1e-4 \
    --max_length 1024 \
    --attn_impl flash_attn \
    --output_dir /workspace/workdir/output/test-sft
```

看到 loss 正常下降、训练结束后在 `workdir/output/test-sft/` 生成 checkpoint，即环境完全正常。
训练完可以直接对话验证（把 `--adapters` 指向上一步输出的 checkpoint 路径）：

```bash
swift infer \
    --adapters /workspace/workdir/output/test-sft/vX-xxx/checkpoint-xxx \
    --stream true
```

> 模型和数据集默认从 ModelScope 下载；想改用 HuggingFace Hub，在命令前加 `USE_HF=1`。

## 可选组件

**vLLM**（GRPO 训练 / 推理加速需要，镜像增大约 10GB）：取消 `Dockerfile` 中这行的注释后重新 build：

```dockerfile
# RUN pip install --no-cache-dir vllm==0.10.0
```

```bash
docker compose build && docker compose up -d
```

## 升级 ms-swift 版本

```bash
cd ms-swift && git fetch --tags && git checkout v4.x.x && cd ..
# 同步修改 Dockerfile 中 ms-swift[llm]==4.x.x 后重新 build（保证依赖一致）
docker compose build && docker compose up -d
```
