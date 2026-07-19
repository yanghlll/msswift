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
export STREAM_PROFILE=1
export STREAM_PROFILE_MAX=6
# 解码端降采样: 解码时就把帧缩到短边 N(而非 1080p 全解再 smart_resize), decode 段
# 提速数倍。取值须 ≥ smart_resize 最终分辨率(MAX_PIXELS=100352 时 ~317px, 448 有余量)。
# 0 = 关。日志看 [STREAM_PIXELS] 确认每帧 token 数不因此变化。
export DECORD_SHORT_SIDE=${DECORD_SHORT_SIDE:-448}

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
  --load_from_cache_file true \
  --dataloader_num_workers 8 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps $((16 / NPROC)) \
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
#   - OOM: 顺序试 use_logits_to_keep(已开) -> 降 MAX_PIXELS -> 降 MAX_LENGTH -> zero3。
#   - 全局 batch = NPROC × per_device_bs × grad_accum = 8×1×4 = 32。
#   - 新 token(</silence></response>)是随机初始化的; 全参训练会自然学到它们的
#     embedding。**若改用 LoRA, 必须加 --modules_to_save embed_tokens,lm_head**,
#     否则这两个控制 token 永远学不会(见 README)。
#   - 断点续训: --resume_from_checkpoint ${OUTPUT}/checkpoint-xxx
# =============================================================================





