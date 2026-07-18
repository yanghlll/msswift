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
MAX_PIXELS=${MAX_PIXELS:-100352}      # 128 tok/frame
MAX_LENGTH=${MAX_LENGTH:-32768}
# -----------------------------------------------------------------------------

export PYTHONPATH=${SWIFT_ROOT}:${PYTHONPATH:-}
export TOKENIZERS_PARALLELISM=false
export NPROC_PER_NODE=${NPROC}

# streaming 后端 + loss_scale 权重(joy_streaming 读这三个 env)
export VIDEO_BACKEND=frames
export W_SILENCE_FIRST=${W_SILENCE_FIRST:-1.0}
export W_SILENCE_REPEATED=${W_SILENCE_REPEATED:-0.4}
export W_RESPONSE=${W_RESPONSE:-1.5}
export MAX_PIXELS=${MAX_PIXELS}

# DEBUG: 少量步数快速验证能否跑通 + 看 step 时间/显存
EXTRA=()
if [[ "${DEBUG:-0}" == "1" ]]; then
  EXTRA+=(--max_steps 20 --save_steps 1000000 --logging_steps 1)
  OUTPUT=${OUTPUT}_debug
fi

mkdir -p "${OUTPUT}"

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
  --deepspeed zero2 \
  --freeze_vit true \
  --freeze_aligner false \
  --torch_dtype bfloat16 \
  --attn_impl flash_attn \
  --gradient_checkpointing true \
  --lazy_tokenize true \
  --load_from_cache_file True \
  --dataloader_num_workers 8 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps $((16 / NPROC)) \
  --max_length "${MAX_LENGTH}" \
  --truncation_strategy delete \
  --num_train_epochs 1 \
  --learning_rate 1e-5 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --save_steps 500 \
  --save_total_limit 3 \
  --logging_steps 5 \
  --dataset_num_proc 8 \
  --output_dir "${OUTPUT}" \
  "${EXTRA[@]}"



# =============================================================================
# 调参提示(按 8×H20 的"算力受限"特点):
#   - 显存不是瓶颈, 算力是。想提速 -> 降 max_duration(reg_joy.py) 或 MAX_PIXELS。
#     token 数 ~线性决定 step 时间。先 DEBUG=1 测实际 s/step 再定。
#   - OOM(不太可能): 降 MAX_PIXELS 或 max_length; ZeRO-2 已把优化器状态分片。
#   - 全局 batch = NPROC × per_device_bs × grad_accum = 8×1×4 = 32。
#   - 新 token(</silence></response>)是随机初始化的; 全参训练会自然学到它们的
#     embedding。**若改用 LoRA, 必须加 --modules_to_save embed_tokens,lm_head**,
#     否则这两个控制 token 永远学不会(见 README)。
#   - 断点续训: --resume_from_checkpoint ${OUTPUT}/checkpoint-xxx
# =============================================================================





