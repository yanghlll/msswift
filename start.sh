#!/usr/bin/env bash
# 一键启动：clone 源码（如缺）→ build 镜像 → 后台起容器 → 进入 shell
# 自动检测 docker compose v2；没有则回退到原生 docker build/run，不依赖 compose
set -euo pipefail
cd "$(dirname "$0")"

MSSWIFT_VERSION=v4.4.1   # 与 Dockerfile 中的依赖版本保持一致
IMAGE=msswift:torch2.7.1-cu128
NAME=msswift

# 额外挂载的宿主机目录（逗号分隔），以相同路径出现在容器内，方便跑其他项目的代码：
#   EXTRA_MOUNT=/root/zlx_workspace bash start.sh
#   EXTRA_MOUNT=/root/zlx_workspace,/data bash start.sh
EXTRA_ARGS=()
if [ -n "${EXTRA_MOUNT:-}" ]; then
    IFS=',' read -ra _dirs <<< "${EXTRA_MOUNT}"
    for d in "${_dirs[@]}"; do
        EXTRA_ARGS+=(-v "${d}:${d}")
        echo ">>> Extra mount: ${d} -> ${d}"
    done
fi

if [ ! -d ms-swift ]; then
    echo ">>> Cloning ms-swift ${MSSWIFT_VERSION} source to ./ms-swift ..."
    git clone -b "${MSSWIFT_VERSION}" https://github.com/modelscope/ms-swift.git
fi

mkdir -p workdir "${HOME}/.cache/huggingface" "${HOME}/.cache/modelscope"

if docker compose version >/dev/null 2>&1 && [ ${#EXTRA_ARGS[@]} -eq 0 ]; then
    echo ">>> Building image & starting container (docker compose) ..."
    echo ">>> （需要额外挂载目录时，直接编辑 docker-compose.yml 的 volumes，或用 EXTRA_MOUNT=... 走 docker run 路径）"
    docker compose up -d --build
else
    echo ">>> docker compose v2 不可用，使用原生 docker build/run ..."
    docker build -t "${IMAGE}" .
    docker rm -f "${NAME}" >/dev/null 2>&1 || true
    docker run -d --name "${NAME}" \
        --gpus all \
        --ipc host \
        --network host \
        --ulimit memlock=-1 --ulimit stack=67108864 \
        --restart unless-stopped \
        -e HF_HOME=/root/.cache/huggingface \
        -v "$(pwd)/ms-swift:/workspace/ms-swift" \
        -v "$(pwd)/workdir:/workspace/workdir" \
        -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
        -v "${HOME}/.cache/modelscope:/root/.cache/modelscope" \
        ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
        "${IMAGE}"
fi

echo ">>> Entering container (再次进入可执行: docker exec -it ${NAME} bash)"
docker exec -it "${NAME}" bash
