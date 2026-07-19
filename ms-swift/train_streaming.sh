#!/usr/bin/env bash
# =============================================================================
# LLaVA-OneVision-2-8B streaming video understanding (JoyAI 格式) SFT 启动脚本
# 目标机: 8 × H20 96GB。全参 SFT + DeepSpeed ZeRO-2。
#
# 前提(务必确认):
#   1) 训练机上用的是**我们改过的这份 ms-swift**(含 streaming 模板 + preprocessor),
#      不是 pip 装的原版。即: cd 到我们的 ms-swift 目录, 或 pip install -e . 装它。
#      改动文件: swift/template/templates/llava.py, swift/template/constant.py,
#                swift/dataset/preprocessor/streaming_video.py
#   2) reg_joy.py 里的 DATA_PATH 指向训练机上的标注 jsonl。
#   3) 环境: transformers>=5.7, dill>=0.3.8, decord(可选), flash-attn。
#
# 用法:  bash train_streaming.sh
#        DEBUG=1 bash train_streaming.sh      # 只跑 ~20 步测 step 时间和显存
# =============================================================================
set -euo pipefail

# --------------------------- 改这里 ------------------------------------------
# ms-swift 源码根(我们改过的那份): 让 `swift` 命令走这份代码
SWIFT_ROOT=${SWIFT_ROOT:-/home/yifan.lu/msswift/ms-swift}
# 模型: 本地目录 或 HF repo id
MODEL=${MODEL:-/data1/zlx/cache/huggingface/hub/model/llavaov2}
# 数据注册脚本(改它里面的 DATA_PATH)
REG=${REG:-/root/zlx_workspace/test/msswift-docker/ms-swift/reg_joy.py}
# 输出
OUTPUT=${OUTPUT:-/root/zlx_workspace/test/msswift-docker/ms-swift/output/joy_streaming}
# GPU
NPROC=${NPROC:-8}

# ROOT_IMAGE_DIR=/data/TimeLens 可能要写这个



# token 预算(每帧 token = MAX_PIXELS/784; 见 README 公式)。
# max_duration=230 + tail_margin=10 时平均 n_seconds ~60s, 但 worst case 仍可能
# 撞 max_length -> 用 truncation_strategy=raise 让超长样本被跳过而非砍尾。
MAX_PIXELS=${MAX_PIXELS:-50176}      # 128 tok/frame
MAX_LENGTH=${MAX_LENGTH:-32768}
# -----------------------------------------------------------------------------

export PYTHONPATH=${SWIFT_ROOT}:${PYTHONPATH:-}
export TOKENIZERS_PARALLELISM=false
# 模型/tokenizer 全部从本地目录加载 -> 强制离线, 免去 transformers 每次去 huggingface.co
# 做版本校验的联网等待(外网不通时每个 rank 干等数分钟, 即 "tokenizer 加载慢" 的元凶)。
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export HF_HUB_ETAG_TIMEOUT=${HF_HUB_ETAG_TIMEOUT:-5}
export NPROC_PER_NODE=${NPROC}

# streaming 后端 + loss_scale 权重(joy_streaming 读这三个 env)
export VIDEO_BACKEND=frames
export W_SILENCE_FIRST=${W_SILENCE_FIRST:-1.0}
export W_SILENCE_REPEATED=${W_SILENCE_REPEATED:-0.4}
export W_RESPONSE=${W_RESPONSE:-1.5}
export MAX_PIXELS=${MAX_PIXELS}
# 计时开关: 默认关(0)。要看各段耗时: STREAM_PROFILE=1 bash train_streaming.sh
# 开销本身极小且自限(每进程只测前 STREAM_PROFILE_MAX 个样本, decode/encode 只是
# perf_counter, vision_fwd 的 cuda.synchronize 也只在前几步), 测满即零开销。
export STREAM_PROFILE=${STREAM_PROFILE:-0}
export STREAM_PROFILE_MAX=${STREAM_PROFILE_MAX:-6}
# STAGE_PROFILE=1: GPU 侧分段计时(ViT/aligner-MLP/LLM 前反向 + fwd/bwd/gap 步级分解),
# 前 STAGE_PROFILE_STEPS(默认8)步测完写 ${OUTPUT}/profile/stage_profile_rank*.log 后自动
# 卸 hook。worker 侧(decord/encode)汇总也写同目录 worker.log。全链路瓶颈分析:
#   STAGE_PROFILE=1 STREAM_PROFILE=1 DEBUG=1 bash train_streaming.sh
export STAGE_PROFILE=${STAGE_PROFILE:-0}
# 解码端降采样: 解码时就把帧缩到短边 N(而非 1080p 全解再 smart_resize), decode 段
# 提速数倍。取值须 ≥ smart_resize 最终分辨率(MAX_PIXELS=100352 时 ~317px, 448 有余量)。
# 0 = 关。日志看 [STREAM_PIXELS] 确认每帧 token 数不因此变化。
export DECORD_SHORT_SIDE=${DECORD_SHORT_SIDE:-448}
# 每个 VideoReader 解码线程上限: 64 个 worker 同时解码, auto 线程会过订阅抢 CPU。
# 0 = 不干预。
export DECORD_NUM_THREADS=${DECORD_NUM_THREADS:-2}
# 显存碎片整理: reserved-but-unallocated 不再浪费(torch 官方建议)
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# DeepSpeed stage: 全参 8B + 32k 序列在 ZeRO-2 下每卡 参数16G+梯度bucket~16G+优化器分片12G
# ≈44G, 加激活直接顶满 96G(实测 91.7G OOM 在 decoder MLP 前向)。ZeRO-3 把参数+梯度也
# 分片(16+16G -> ~4G), 每卡省 ~28G。H20 有 NVLink, zero3 的 allgather 开销可接受。
# DeepSpeed 配置: 默认调优版 zero2(zero2_tuned.json)。与官方 zero2.json 的差别:
#   overlap_comm=true  —— 梯度 reduce 与反向计算重叠(官方是 false, 反向每层算完都
#                         干等通信, 是 GPU 功率只有 200W/500W 的主因)
#   桶 2e8 -> 4e8      —— 通信次数减半(约多占 ~1.5G 显存, 若紧改回 2e8)
# 回官方: DS=zero2; 显存不够换 zero3: DS=${SWIFT_ROOT}/zero3_tuned.json(同样已调优)
DS=${DS:-${SWIFT_ROOT}/zero2_tuned.json}
# 自防御: 配置文件不存在(如没同步到训练机)或 DS 意外为空 -> 回退官方 zero2,
# 绝不把空串/坏路径传给 --deepspeed(否则 swift 报 "Unable to parse json string: ''")
if [[ -z "${DS}" ]]; then
  echo "[train_streaming][WARN] DS 为空, 回退 zero2"; DS=zero2
elif [[ "${DS}" == *.json && ! -f "${DS}" ]]; then
  echo "[train_streaming][WARN] deepspeed 配置不存在: ${DS}, 回退官方 zero2"; DS=zero2
fi
# DS_PROFILE=1: 生成临时配置开 wall_clock_breakdown, DeepSpeed 每步打印 fwd/bwd/step
# 各段耗时 —— 定位 20s/it 到底花在哪(配 DEBUG=1 跑 20 步看)。
if [[ "${DS_PROFILE:-0}" == "1" && -f "${DS}" ]]; then
  DS_TMP=$(mktemp /tmp/ds_profile_XXXX.json)
  python -c "import json,sys; c=json.load(open('${DS}')); c['wall_clock_breakdown']=True; json.dump(c, open('${DS_TMP}','w'), indent=2)"
  DS=${DS_TMP}
fi
echo "[train_streaming] deepspeed = ${DS}"

# DEBUG: 少量步数快速验证能否跑通 + 看 step 时间/显存
EXTRA=()
if [[ "${DEBUG:-0}" == "1" ]]; then
  EXTRA+=(--max_steps 20 --save_steps 1000000 --logging_steps 1)
  OUTPUT=${OUTPUT}_debug
fi

mkdir -p "${OUTPUT}"
if [[ "${STAGE_PROFILE}" == "1" || "${STREAM_PROFILE}" == "1" ]]; then
  export STREAM_PROFILE_DIR=${STREAM_PROFILE_DIR:-${OUTPUT}/profile}
  mkdir -p "${STREAM_PROFILE_DIR}"
fi

# swift 命令: 优先用 SWIFT_ROOT 里的入口
SWIFT_BIN=${SWIFT_BIN:-swift}

${SWIFT_BIN} sft \
  --model "${MODEL}" \
  --custom_register_path "${REG}" \
  --dataset joy_streaming_video \
  --template llava_onevision2_streaming \
  --loss_scale joy_streaming \
  --new_special_tokens '</silence>,</response>' \
  --tuner_type full \
  --deepspeed "${DS}" \
  --freeze_vit true \
  --freeze_aligner false \
  --torch_dtype bfloat16 \
  --attn_impl flash_attn \
  --gradient_checkpointing "${GC:-true}" \
  --lazy_tokenize true \
  --group_by_length "${GBL:-true}" \
  --load_from_cache_file "${LOAD_CACHE:-true}" \
  --dataloader_num_workers 8 \
  ${PF:+--padding_free true} \
  --per_device_train_batch_size "${BS:-1}" \
  --gradient_accumulation_steps "$(( 64 / NPROC / ${BS:-1} > 0 ? 64 / NPROC / ${BS:-1} : 1 ))" \
  --max_length "${MAX_LENGTH}" \
  --truncation_strategy delete \
  --use_logits_to_keep true \
  --num_train_epochs 1 \
  --learning_rate 2e-5 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --save_steps 500 \
  --save_total_limit 3 \
  --logging_steps 5 \
  --dataset_num_proc 8 \
  --use_liger_kernel true \
  --output_dir "${OUTPUT}" \
  "${EXTRA[@]}"



# =============================================================================
# 调参提示(按 8×H20 的"算力受限"特点):
#   - 跑起来先看两行日志:
#       [STREAM_PIXELS] 每帧 HxW -> N tokens/帧   ← 验证 MAX_PIXELS 真吃进 video_processor
#       use_logits_to_keep: True                  ← 只算被监督位置的 logits(LOV2 forward 已确认支持)
#   - use_logits_to_keep=true 是显存最大杠杆: loss_scale 路径会 materialize 完整 logits
#     并 fp32 upcast(32768×~152k 词表 ≈ 10G bf16 + 20G fp32); 开掩码后只剩被监督的
#     几百个位置, 省 50-100x。若 loss 异常再关掉排查。
#   - 想提速 -> 降 max_duration(reg_joy.py) 或 MAX_PIXELS; DECORD_SHORT_SIDE 已开(解码提速)。
#     token 数 ~线性决定 step 时间。先 DEBUG=1 测实际 s/step 再定。
#   - group_by_length=true(GBL=false 关): 按预估长度分组, 同一步 8 卡拿到等长样本,
#     消掉 bwd 里 ~5s 的 straggler 等待(bwd/fwd 应从 6.4x 回落到 ~3-4x)。
#     首次启用或改过 MAX_PIXELS 后, 先 LOAD_CACHE=false 跑一次让数据集缓存重建出
#     lengths 列(旧缓存没有该列会 KeyError), 之后恢复默认 true。
#   - OOM: zero3 已是默认(zero2 实测 91.7G 顶满)。仍 OOM 顺序试: 降 MAX_LENGTH
#     (32768->24576/16384, 激活/logits 线性降) -> 降 MAX_PIXELS -> reg_joy.py 降
#     max_duration -> zero3 + offload_optimizer。
#   - 全局 batch = NPROC × per_device_bs × grad_accum = 8×1×4 = 32。
#   - 新 token(</silence></response>)是随机初始化的; 全参训练会自然学到它们的
#     embedding。**若改用 LoRA, 必须加 --modules_to_save embed_tokens,lm_head**,
#     否则这两个控制 token 永远学不会(见 README)。
#   - 断点续训: --resume_from_checkpoint ${OUTPUT}/checkpoint-xxx
# =============================================================================






