NUM_FRAMES=8 MAX_PIXELS=1003520 swift sft \
    --model /data1/zlx/cache/huggingface/hub/model/llavaov2 \
    --model_type llava_onevision2 \
    --dataset swift/VideoChatGPT:all \
    --tuner_type lora \
    --freeze_vit true \
    --output_dir output \
    --max_steps 10