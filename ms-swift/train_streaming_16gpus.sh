#!/usr/bin/env bash
# =============================================================================
# 16 GPU 版启动脚本 —— train_streaming.sh 的多节点/多卡预设包装。
# 只设「节点/通信/grad_accum」这几个多卡相关项, 其余全部复用 train_streaming.sh
# (改主脚本的配置这里自动跟着变, 不用维护两份)。
#
# 两种 16 卡拓扑:
#   A) 2 节点 × 8 卡(H20 常见)—— 每个节点各跑一次本脚本, 只改 NODE_RANK:
#        节点0(主): NODE_RANK=0 MASTER_ADDR=<节点0的IP> bash train_streaming_16gpus.sh
#        节点1:     NODE_RANK=1 MASTER_ADDR=<节点0的IP> bash train_streaming_16gpus.sh
#      两个节点的 MASTER_ADDR 都填「节点0的实际IP」(不能是 127.0.0.1), MASTER_PORT 一致。
#   B) 单机 16 卡 —— 一条命令:
#        NNODES=1 NPROC=16 bash train_streaming_16gpus.sh
#
# 透传: DEBUG=1 / STREAM_PROFILE=1 / MAX_PIXELS=... 等所有 train_streaming.sh 的开关
#       照常前置即可(会被继承)。
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

# --------------------------- 多节点/多卡参数 --------------------------------
export NNODES=${NNODES:-2}                     # 节点数(2 节点各 8 卡 = 16)
export NPROC=${NPROC:-8}                        # 每节点 GPU 数(单机16卡时设 16)
export NODE_RANK=${NODE_RANK:-0}                # 本节点序号: 主节点=0, 其余 1,2,...
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}    # 主节点(rank0)IP; 多节点必须填真实IP
export MASTER_PORT=${MASTER_PORT:-29500}        # 所有节点用同一端口
# -----------------------------------------------------------------------------

TOTAL_GPUS=$(( NNODES * NPROC ))
BS=${BS:-1}
# 保持全局 batch = 64 不变(和单机 8 卡一致): grad_accum = 64 / 总卡数 / BS
export GA=${GA:-$(( 64 / TOTAL_GPUS / BS > 0 ? 64 / TOTAL_GPUS / BS : 1 ))}

echo "[16gpu] NNODES=${NNODES} 每节点=${NPROC}卡 总卡=${TOTAL_GPUS} NODE_RANK=${NODE_RANK} " \
     "MASTER=${MASTER_ADDR}:${MASTER_PORT} BS=${BS} GA=${GA} " \
     "全局batch=$(( TOTAL_GPUS * BS * GA ))"

if [[ "${NNODES}" -gt 1 && "${MASTER_ADDR}" == "127.0.0.1" ]]; then
  echo "[16gpu][WARN] 多节点但 MASTER_ADDR=127.0.0.1 —— 从节点连不上主节点!" \
       "请设成节点0的真实IP。" >&2
fi

# 复用主脚本的全部配置与逻辑(DS/profile/env/swift 命令都在里面)
exec bash train_streaming.sh "$@"
