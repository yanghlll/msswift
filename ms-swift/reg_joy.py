# JoyAI streaming 数据集注册。训练时 `swift sft --custom_register_path reg_joy.py`
# 会在启动时 import 本文件, 把下面的数据集注册进 ms-swift, 之后即可用
# `--dataset joy_streaming_video` 引用。
#
# 改这里几个东西即可:
#   DATA_PATH     指向你的标注 (目录 / 通配 / 单文件, 见下)
#   max_duration  样本时间轴上限(秒); 配合 filter_joyai.py 砍掉的长尾, 建议 230
#   head_margin   事件窗口截断: 首事件【之前】保留几秒上下文。砍掉问题前的长 lead-in
#                 silence(不平衡元凶), 又给"刚才发生了什么"类问题留依据。
#   tail_margin   末事件【之后】保留几秒(尾部静默负样本)。
#                 三形态: None=0; int=固定; (lo,hi)=每样本随机(打散位置防周期性开口)。
from swift.dataset.preprocessor.streaming_video import register_joy_streaming_dataset

# --- 改这一行为你的实际路径 ---
# 目录(自动递归所有 jsonl): '/data/joyai/annotations_filtered'
# 或通配: '/data/joyai/**/*.jsonl'   或单文件: '/data/joyai/chat/ActivityNet.jsonl'
DATA_PATH = '/data1/zlx/cache/huggingface/hub/annotations_filtered_96'

register_joy_streaming_dataset(
    DATA_PATH,
    name='joy_streaming_video',   # --dataset 用这个名字引用
    max_duration=230,             # 配 filter_joyai.py 砍掉的 1% 长尾 (t_max>230)
    head_margin=(5, 15),          # 首事件前保留 5~15s 随机上下文 (砍问题前的长 lead-in)
    tail_margin=(0, 10),          # 末事件后保留 0~10s 随机 (尾部静默负样本)
    video_root='/data1/zlx/cache/huggingface/hub/vl-interaction-reenc'
)
