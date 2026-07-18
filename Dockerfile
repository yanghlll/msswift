# syntax=docker/dockerfile:1.4
# 构建命令: DOCKER_BUILDKIT=1 docker build -t your-image .

FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_RETRIES=10 \
    PIP_TIMEOUT=120 \
    PIP_RESUME_RETRIES=5
# 说明:
# - 不设 PIP_INDEX_URL, 使用官方 PyPI (实测本机访问官方源最快最稳)
# - PIP_RETRIES: 网络瞬断时 pip 自动重试, 不至于整层失败
# - PIP_RESUME_RETRIES: pip 25.1+ 支持断点续传, 大包下到一半断了从断点继续

RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl wget vim tmux htop openssh-client \
        build-essential ninja-build ca-certificates \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*
# ffmpeg 提供 ffprobe: streaming 数据 preprocessor 用它拿视频时长
# (否则每行时长=0 -> 数据集全空 -> ZeroDivisionError)

# 先升级 pip 以获得断点续传能力 (PIP_RESUME_RETRIES 需要 pip >= 25.1)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -U pip setuptools wheel packaging ninja

# ── 关于缓存策略 ──────────────────────────────────────────────
# 全部改用 BuildKit cache mount 替代 --no-cache-dir:
# 下载缓存保存在宿主机构建缓存中, 不进镜像层 (镜像不会变大),
# 但任何包只要成功下载过一次, 之后重新 build 直接本地秒装, 不再依赖网络.
# ─────────────────────────────────────────────────────────────

# flash-attn: 自动匹配 torch2.7 + cu12 的预编译 wheel (H100/sm_90 可用), 无需本地编译
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install flash-attn==2.8.3 --no-build-isolation

# 先把 ms-swift 的完整依赖装进镜像; 容器启动时会被挂载进来的源码以 editable 方式覆盖
# 版本与 start.sh 中 clone 的源码 tag (v4.4.1) 保持一致
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install 'ms-swift[llm]==4.4.1' 'deepspeed<0.19'

# 视频处理依赖: av(PyAV) / decord / cv2. 放在靠后的层, 改这里重新 build 走增量缓存
# opencv 用 headless 版: 服务器无显示环境, 避免依赖 libGL; import cv2 用法完全一致
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install av decord opencv-python-headless

# 可选: GRPO 训练 / vLLM 推理加速需要 vllm (镜像会大 ~10GB, 不需要可注释掉)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install vllm==0.10.0

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

WORKDIR /workspace
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["sleep", "infinity"]