# Copyright (c) ModelScope Contributors. All rights reserved.
from typing import List, Literal, Optional, Tuple

from swift.template import ContextType, Messages
from swift.utils import get_env_args
from .base import LossScale


class JoyStreamingLossScale(LossScale):
    """Role-aware token weighting for time-aligned (streaming) interaction data.

    On time-aligned data the assistant emits one turn per second and the vast majority
    of those turns are `</silence>`, so a standard SFT loss is dominated by the silence
    token: the gradient is pushed toward continued silence and the signal for responding
    gets diluted. Following the streaming-native recipe, only the *control* token of each
    turn is reweighted:

    - `</response>`                   -> ``w_response`` (> 1, up-weight response onsets)
    - first `</silence>` of a run     -> ``w_silence_first``
    - continued `</silence>`          -> ``w_silence_repeated`` (< 1, down-weight silence runs)
    - every other supervised position -> 1.0

    A run of silence starts at the beginning of the sample or right after a `</response>`,
    so the leading `</silence>` of a video counts as "first". The response body (including
    any delegation) rides inside the `</response>` turn and keeps weight 1.0.

    Conventional turn-based data contains neither control token, so every context falls
    through to weight 1.0 and the objective reduces exactly to standard SFT. No dataset
    flag is needed to tell the two apart.

    Combined with the trainer's ``sum(w * ce) / (labels != -100).sum()``, this yields the
    normalized weighted cross-entropy ``-1/|A| * sum_{j in A} w_j * log p(y_j | y_<j)``,
    where ``A`` is the set of supervised assistant-token positions.

    The weights are read from the environment so they can be swept without editing code:
    ``W_SILENCE_FIRST``, ``W_SILENCE_REPEATED``, ``W_RESPONSE``.
    """
    is_binary = False

    silence_token = '</silence>'
    response_token = '</response>'

    def __init__(self, base_strategy: Literal['default', 'last_round', 'all'] = 'default'):
        super().__init__(base_strategy)
        self.w_silence_first = get_env_args('w_silence_first', float, 1.)
        self.w_silence_repeated = get_env_args('w_silence_repeated', float, 0.4)
        self.w_response = get_env_args('w_response', float, 1.5)
        self._prev_control = None

    def __call__(self, context_list: List[str], context_types: List[ContextType], messages: Messages,
                 **kwargs) -> Tuple[List[str], List[float]]:
        # A silence run spans rounds, so the state has to be reset per sample rather than
        # per round. `__call__` is invoked once per sample, `get_loss_scale` once per round.
        self._prev_control = None
        return super().__call__(context_list, context_types, messages, **kwargs)

    def _split_control(self, context: str, control_token: str, weight: float) -> Tuple[List[str], List[float]]:
        body = context[len(control_token):]
        if body:
            return [control_token, body], [weight, 1.]
        return [control_token], [weight]

    def get_loss_scale(self, context: str, *, query: Optional[str] = None, **kwargs):
        if not isinstance(context, str):
            return super().get_loss_scale(context, query=query, **kwargs)
        if context.startswith(self.silence_token):
            weight = self.w_silence_repeated if self._prev_control == 'silence' else self.w_silence_first
            self._prev_control = 'silence'
            return self._split_control(context, self.silence_token, weight)
        if context.startswith(self.response_token):
            self._prev_control = 'response'
            return self._split_control(context, self.response_token, self.w_response)
        # Turn-based data, and the template suffix of every streaming round, land here. The
        # suffix must not touch `_prev_control`: it is emitted between control tokens.
        return [context], [1.]
