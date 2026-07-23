# Copyright (c) ModelScope Contributors. All rights reserved.
"""Preprocessor for JoyAI-VL-Interaction streaming-video-understanding data.

Converts a raw JoyAI row::

    {"video_name", "video_path", "task_type", "source",
     "question": [{"content", "time"}], "response": [{"content", "time"}]}

into per-second interleaved turns **without extracting frames to disk**. The
video is left as a single ``videos: [path]`` entry and decoded on the fly by
``LLavaOneVision2StreamingTemplate`` (frames-sample backend). This mirrors
``JoyAI-VL-Interaction/datasets/convert_data.py::convert_sample`` but keeps the
video instead of writing ``frame_XXXXXX.jpg`` + ``images: [...]``.

Per second ``sec`` in ``[0, n_seconds)``:
  - user turn: ``"[<question>\\n]<sec.0 seconds>\\n<|video_pad|>"`` — exactly one
    ``<|video_pad|>`` sentinel; the template splices that second's decoded frame
    tokens in its place.
  - assistant turn: ``"</response> <answer>"`` if a response fires at ``sec``,
    else ``"</silence>"``.

fps is duration-adaptive (>=160s -> 1, >=64s -> 2, else 4), matching JoyAI.
Metadata the template needs (fps / n_seconds / frames_per_sec) rides in
``chat_template_kwargs`` — the only non-media row field that survives
``RowPreprocessor.remove_useless_columns``.
"""
import hashlib
import os
import random
import subprocess
from typing import Any, Dict, List, Optional, Tuple, Union

from swift.utils import get_logger
from .core import RowPreprocessor

logger = get_logger()

# per-second visual placeholder: reuse the model's own <|video_pad|> token, which
# is already a single vocab token AND a truncation-protected placeholder_token.
STREAM_FRAME_TAG = '<|video_pad|>'


def _ffprobe_duration(video_path: str) -> float:
    """视频时长(秒)。优先 ffprobe(镜像 convert_data.get_video_duration), 失败/没装
    ffprobe 时回退 opencv(frames 后端本就依赖它) —— 避免硬依赖系统 ffprobe 二进制。
    两者都拿不到返回 0.0。"""
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
             '-of', 'default=noprint_wrappers=1:nokey=1', video_path],
            capture_output=True, text=True)
        d = float(result.stdout.strip())
        if d > 0:
            return d
    except (ValueError, AttributeError, OSError):
        pass
    # 回退: opencv 用 帧数/fps 估时长
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        try:
            frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
            fps = cap.get(cv2.CAP_PROP_FPS) or 0
        finally:
            cap.release()
        if frames > 0 and fps > 0:
            return float(frames) / float(fps)
    except Exception:
        pass
    return 0.0


def _parse_times(time_str) -> List[int]:
    """'8' or '5,6,7' -> [8] / [5,6,7] (mirrors convert_data.parse_times)."""
    if not time_str and time_str != 0:
        return []
    return [int(float(t.strip())) for t in str(time_str).split(',') if str(t).strip()]


class JoyStreamingVideoPreprocessor(RowPreprocessor):

    def __init__(self, *, max_duration: int = 320,
                 head_margin: Union[int, Tuple[int, int], None] = (5, 15),
                 tail_margin: Union[int, Tuple[int, int], None] = (0, 10),
                 video_root: Optional[str] = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.max_duration = max_duration
        # video_root: 数据里 video_path 是相对路径(如 JoyAI 的 'videos_pool/xxx.mp4')时的
        # 根目录。preprocess 阶段就要 ffprobe, 所以必须在这里解析成绝对路径 —— 模板阶段的
        # ROOT_IMAGE_DIR 解析发生得太晚, 救不了这里。未显式传时回退读 ROOT_IMAGE_DIR env。
        self.video_root = video_root or os.environ.get('ROOT_IMAGE_DIR') or None
        # ---- 事件窗口截断: 只保留 [首事件-head, 末事件+tail] 这段, 时间轴归零从 0 秒开始 ----
        # 动机: 问题/首事件【之前】的长 lead-in 全是 </silence>(数据 90% 是 silence,
        # 元凶就是这段无问题的空转), 且答完之后的长尾也全 silence。两头一裁, 既大幅缓解
        # silence:response 不平衡, 又缩短序列(省算力), 还把时间轴对齐到推理的 <0.0 seconds>。
        #
        # head_margin: 首事件【之前】保留几秒上下文(供"刚才发生了什么"类问题看到问题前的画面)。
        # tail_margin: 末事件【之后】保留几秒(尾部静默负样本, 教模型"答完继续闭嘴")。
        #   取值三形态: None -> 0(不留); int -> 固定; (lo,hi) -> 每样本在 [lo,hi] 随机。
        #   随机窗口(默认 head=(5,15) tail=(0,10))能打散 response 在 clip 内的位置, 防止
        #   模型学成"沉默 N 秒必开口"的周期性病(见讨论)。时间不足时自动裁到可用范围。
        #   随机用【每样本确定性种子】(video_path 的 md5), 缓存重建/多 worker 都稳定一致。
        self.head_margin = head_margin
        self.tail_margin = tail_margin

    @staticmethod
    def _adaptive_fps(duration: float) -> float:
        if duration >= 160:
            return 1.0
        elif duration >= 64:
            return 2.0
        return 4.0

    @staticmethod
    def _sample_margin(rng: random.Random, spec: Union[int, Tuple[int, int], None]) -> int:
        """把 head/tail_margin 规格解析成一个具体秒数: None->0; int->固定; (lo,hi)->随机。"""
        if spec is None:
            return 0
        if isinstance(spec, (tuple, list)):
            lo, hi = int(spec[0]), int(spec[1])
            return lo if hi <= lo else rng.randint(lo, hi)
        return int(spec)

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        video_path = row.get('video_path') or (row.get('videos') or [None])[0]
        if not video_path:
            return None
        # 相对路径 -> 用 video_root 拼成绝对路径(ffprobe 和后续解码都要能找到文件)
        if self.video_root and not os.path.isabs(video_path) and not video_path.startswith('http'):
            video_path = os.path.join(self.video_root, video_path)
        duration = _ffprobe_duration(video_path)
        if duration <= 0:
            logger.warning_once(
                f'ffprobe duration<=0, skipping: {video_path}. '
                f'若是相对路径找不到文件, 传 video_root=(或设 ROOT_IMAGE_DIR)指向视频根目录。',
                hash_id='joy_stream_dur')
            return None

        fps = self._adaptive_fps(duration)
        frames_per_sec = max(int(fps), 1)
        if self.max_duration and self.max_duration > 0:
            truncated = duration > self.max_duration
            effective_duration = min(duration, self.max_duration)
        else:
            truncated = False
            effective_duration = duration
        n_seconds = max(int(effective_duration), 1)

        # second_idx -> text
        question_map: Dict[int, str] = {}
        for q in row.get('question') or []:
            for t in _parse_times(q.get('time')):
                if t > effective_duration:
                    if truncated:
                        continue
                    return None  # question past end of an un-truncated video -> bad row
                question_map[min(t, n_seconds - 1)] = q['content']

        response_map: Dict[int, str] = {}
        raw_responses = row.get('response') or []
        flat: List[Dict[str, Any]] = []
        for item in raw_responses:
            flat.extend(item) if isinstance(item, list) else flat.append(item)
        for r in flat:
            for t in _parse_times(r.get('time')):
                if t > effective_duration:
                    if truncated:
                        continue
                    return None
                response_map[min(t, n_seconds - 1)] = r['content']

        # 所有 response 都落在 max_duration 之外 -> 保留下来会退化成全 </silence>,
        # 等于教模型"该说话时保持沉默"(静默的数据污染)。直接丢弃这条样本。
        # 注: JoyAI convert_data.py 此处是 continue(保留), 我们刻意更严格 ——
        # max_duration=320 时仅影响 1/180k 条; 但调小 max_duration 时(128->4.2%,
        # 64->11.9%)这个保护就很关键。
        if raw_responses and not response_map:
            logger.warning_once(
                f'所有 response 都超出 max_duration={self.max_duration}s, 丢弃该样本: {video_path}',
                hash_id='joy_stream_all_resp_cut')
            return None

        # ---- 事件窗口截断 + 时间轴归零 ----------------------------------------
        # 窗口 = [首事件 - head, 末事件 + tail], 只保留这段, 秒号重编从 0 开始。
        #   · 砍掉首事件【之前】的长 lead-in silence(问题还没来, 全是空转) —— 不平衡元凶。
        #   · 砍掉末事件【之后】的长尾 silence。
        #   · head/tail 每样本随机(确定性种子), 打散 response 在窗口内的位置, 防周期性开口。
        #   · 时间不足自动裁到可用范围(window_start>=0, window_end<=n_seconds)。
        # 必须放在两个 map 建好之后。窗口按绝对秒算, 之后把所有 event 平移 -window_start 重编号。
        window_start = 0
        events = sorted(set(question_map) | set(response_map))
        if events and (self.head_margin is not None or self.tail_margin is not None):
            seed = int(hashlib.md5(str(video_path).encode('utf-8')).hexdigest()[:8], 16)
            rng = random.Random(seed)
            head = self._sample_margin(rng, self.head_margin)
            tail = self._sample_margin(rng, self.tail_margin)
            first_ev, last_ev = events[0], events[-1]
            window_start = max(0, first_ev - head)
            window_end = min(n_seconds, last_ev + tail + 1)   # 绝对秒; 时间不足则被 n_seconds 夹住
            # 平移重编号: 绝对秒 t -> 相对秒 t - window_start (窗口按构造覆盖了所有 event)
            question_map = {t - window_start: c for t, c in question_map.items()
                            if window_start <= t < window_end}
            response_map = {t - window_start: c for t, c in response_map.items()
                            if window_start <= t < window_end}
            n_seconds = max(window_end - window_start, 1)

        messages: List[Dict[str, str]] = []
        for sec in range(n_seconds):
            parts = []
            if sec in question_map:
                parts.append(question_map[sec])
            parts.append(f'<{sec:.1f} seconds>')
            parts.append(STREAM_FRAME_TAG)  # exactly one sentinel per second
            messages.append({'role': 'user', 'content': '\n'.join(parts)})
            if sec in response_map:
                messages.append({'role': 'assistant', 'content': f'</response> {response_map[sec]}'})
            else:
                messages.append({'role': 'assistant', 'content': '</silence>'})

        # token 长度估算(零解码, 供 --group_by_length 消 straggler / _stat_dataset 统计):
        # 视觉 = 每秒帧数 × (每帧 token + vision_start/end + 换行), 每帧 token ≈ MAX_PIXELS/784
        # (patch14 × merge2 -> (28px)^2=784 px/token, 帧通常打满 smart_resize 预算);
        # 文本按 ~3 字符/token 粗估。分组只需相对大小正确, n_seconds 主导, 精度足够。
        n_per_frame = int(os.environ.get('MAX_PIXELS', '100352')) // 784
        visual_est = n_seconds * frames_per_sec * (n_per_frame + 3)
        text_est = sum(len(m['content']) for m in messages) // 3 + 5 * len(messages)
        return {
            'messages': messages,
            'videos': [video_path],
            'lengths': int(visual_est + text_est + 32),
            'chat_template_kwargs': {
                'stream_fps': fps,
                'stream_n_seconds': n_seconds,
                'stream_frames_per_sec': frames_per_sec,
                'stream_max_duration': self.max_duration,
                # 模板解码要按此偏移分桶: 绝对秒 int(t) - window_start -> 窗口内相对秒。
                'stream_window_start': window_start,
            },
        }


def register_joy_streaming_dataset(dataset_path: str, *, name: str = 'joy_streaming_video',
                                   max_duration: int = 320,
                                   head_margin: Union[int, Tuple[int, int], None] = (5, 15),
                                   tail_margin: Union[int, Tuple[int, int], None] = (0, 10),
                                   video_root: Optional[str] = None,
                                   pattern: str = '**/*.jsonl') -> None:
    """把本地 JoyAI 原始数据注册成可用 `--dataset {name}` 引用的**单个**数据集。

    `dataset_path` 支持三种形态（都汇成一个数据集，preprocessor 逐行应用）：
      - 单个文件:   '/path/a.jsonl'
      - 通配符:     '/path/*.jsonl' 或 '/path/**/*.jsonl'（HF 递归 glob）
      - 目录:       '/path/'  → 自动展开为 '/path/{pattern}'（默认递归所有 .jsonl）

    `max_duration` 时间轴上限(秒); `head_margin`/`tail_margin` 事件窗口截断的前/后保留秒数
    (None=0; int=固定; (lo,hi)=每样本随机; 见 JoyStreamingVideoPreprocessor)。

    在 `--custom_register_path your_reg.py` 里调用即可，例如::

        from swift.dataset.preprocessor.streaming_video import register_joy_streaming_dataset
        register_joy_streaming_dataset('/data/joyai/annotations', max_duration=230,
                                       head_margin=(5, 15), tail_margin=(0, 10))
    """
    import glob as _glob
    from ..dataset_meta import DatasetMeta
    from ..register import register_dataset
    if os.path.isdir(dataset_path):
        # fsspec 的 '*.jsonl' 只匹配顶层, '**/*.jsonl' 只匹配子目录, 无单一 .jsonl-glob
        # 同时匹配两者。按实际布局选: 纯顶层(flat) -> *.jsonl; 有子目录(JoyAI 的
        # task/source 结构) -> **/*.jsonl。混放时给出告警。
        has_flat = bool(_glob.glob(os.path.join(dataset_path, '*.jsonl')))
        has_nested = bool(_glob.glob(os.path.join(dataset_path, '*', '**', '*.jsonl'), recursive=True))
        pat = '*.jsonl' if (has_flat and not has_nested) else pattern
        if has_flat and has_nested:
            logger.warning(f'{dataset_path} 顶层与子目录都有 jsonl; 用 {pat!r} 只会匹配子目录, '
                           '顶层文件请单独注册或移入子目录')
        dataset_path = os.path.join(dataset_path, pat)
    register_dataset(
        DatasetMeta(
            dataset_path=dataset_path,
            dataset_name=name,
            preprocess_func=JoyStreamingVideoPreprocessor(
                max_duration=max_duration, head_margin=head_margin, tail_margin=tail_margin,
                video_root=video_root),
            tags=['video', 'streaming'],
        ),
        exist_ok=True)
