# JoyAI streaming 数据集注册。训练时 `swift sft --custom_register_path reg_joy.py`
# 会在启动时 import 本文件, 把下面的数据集注册进 ms-swift, 之后即可用
# `--dataset joy_streaming_video` 引用。
#
# 改这里 3 个东西即可:
#   DATA_PATH     指向你的标注 (目录 / 通配 / 单文件, 见下)
#   max_duration  样本时间轴上限(秒); 配合 filter_joyai.py 砍掉的长尾, 建议 230
#   tail_margin   末个事件之后只再保留几秒(其余 </silence> 尾部不看); 建议 10
#                 None=不裁(JoyAI 原版), 0=裁到末事件(不推荐)
from swift.dataset.preprocessor.streaming_video import register_joy_streaming_dataset

# --- 改这一行为你的实际路径 ---
# 目录(自动递归所有 jsonl): '/data/joyai/annotations_filtered'
# 或通配: '/data/joyai/**/*.jsonl'   或单文件: '/data/joyai/chat/ActivityNet.jsonl'
DATA_PATH = '/data1/zlx/cache/huggingface/hub/annotations_en_raw_jsonl_short'

register_joy_streaming_dataset(
    DATA_PATH,
    name='joy_streaming_video',   # --dataset 用这个名字引用
    max_duration=230,             # 配 filter_joyai.py 砍掉的 1% 长尾 (t_max>230)
    tail_margin=10,               # 砍末事件之后的长尾 (留 10s 尾部静默做负样本)
    video_root='/data1/zlx/cache/huggingface/hub/vl-interaction'
)
