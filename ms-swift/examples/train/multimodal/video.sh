# 2*24GB
# You can refer to `https://github.com/QwenLM/Qwen2.5-VL` for the meaning of the `VIDEO_MAX_PIXELS` parameter.
nproc_per_node=8

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=$nproc_per_node \
NUM_FRAMES=12 \
MAX_PIXELS=50176 \
ROOT_IMAGE_DIR=/data/TimeLens \
swift sft \
    --model /data1/zlx/cache/huggingface/hub/model/llavaov2 \
    --model_type llava_onevision2 \
    --dataset /data1/zlx/cache/huggingface/hub/video_data/test/chat_TimeLens-100K_000_1fps_abs.jsonl \
    --load_from_cache_file True \
    --split_dataset_ratio 0.01 \
    --tuner_type lora \
    --torch_dtype bfloat16 \
    --num_train_epochs 10 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --freeze_vit true \
    --freeze_aligner true \
    --gradient_accumulation_steps $(expr 16 / $nproc_per_node) \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 8192 \
    --output_dir output \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --deepspeed zero2 \
    --dataset_num_proc 4


