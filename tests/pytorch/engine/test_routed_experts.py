# yapf: disable
import numpy as np

from lmdeploy.messages import GenerationConfig
from lmdeploy.pytorch.engine.engine_instance import EngineInstance
from lmdeploy.pytorch.engine.request import ResponseType

# yapf: enable

EOS = 151645


class _FakeResp:
    """Minimal stand-in for engine ``Response`` (only ``.type`` / ``.data``)."""

    def __init__(self, resp_type, routed_experts=None, token_ids=None):
        self.type = resp_type
        data = {}
        if routed_experts is not None:
            data['routed_experts'] = routed_experts
        if token_ids is not None:
            data['token_ids'] = token_ids
        self.data = data


class _FakeInstance:
    """Carries only what ``_get_extra_outputs`` reads (no engine construction)."""

    _enable_transfer_obj_ref = False
    _is_trailing_stop_token_excluded = staticmethod(EngineInstance._is_trailing_stop_token_excluded)


def _gen_config(include_stop_str_in_output=True, ignore_eos=False, stop_token_ids=(EOS, )):
    return GenerationConfig(
        include_stop_str_in_output=include_stop_str_in_output,
        ignore_eos=ignore_eos,
        stop_token_ids=list(stop_token_ids),
    )


class TestTrailingStopTokenExcluded:
    """``_is_trailing_stop_token_excluded`` must agree with AsyncEngine's gen_len."""

    def test_cancel_with_trailing_eos_is_excluded(self):
        resp = _FakeResp(ResponseType.CANCEL)
        assert EngineInstance._is_trailing_stop_token_excluded(resp, _gen_config(), [1, 2, EOS]) is True

    def test_cancel_without_trailing_stop_is_kept(self):
        resp = _FakeResp(ResponseType.CANCEL)
        assert EngineInstance._is_trailing_stop_token_excluded(resp, _gen_config(), [1, 2, 999]) is False

    def test_finish_stop_kept_when_include_stop_str(self):
        resp = _FakeResp(ResponseType.FINISH)
        cfg = _gen_config(include_stop_str_in_output=True)
        assert EngineInstance._is_trailing_stop_token_excluded(resp, cfg, [1, 2, EOS]) is False

    def test_finish_stop_excluded_when_not_include_stop_str(self):
        resp = _FakeResp(ResponseType.FINISH)
        cfg = _gen_config(include_stop_str_in_output=False)
        assert EngineInstance._is_trailing_stop_token_excluded(resp, cfg, [1, 2, EOS]) is True

    def test_ignore_eos_never_excluded(self):
        resp = _FakeResp(ResponseType.CANCEL)
        cfg = _gen_config(ignore_eos=True)
        assert EngineInstance._is_trailing_stop_token_excluded(resp, cfg, [1, 2, EOS]) is False

    def test_empty_or_missing_tokens(self):
        resp = _FakeResp(ResponseType.CANCEL)
        assert EngineInstance._is_trailing_stop_token_excluded(resp, _gen_config(), []) is False
        assert EngineInstance._is_trailing_stop_token_excluded(resp, _gen_config(), None) is False


class TestGetExtraOutputsLength:
    """``len(routed_experts)`` must equal ``prompt + completion - 1`` after the fix."""

    def _routed_len(self, resp_type, raw_len, num_all_ids, last_token, gen_config):
        routed = np.arange(raw_len * 2, dtype=np.int64).reshape(raw_len, 2)
        resp = _FakeResp(resp_type, routed_experts=routed, token_ids=[1, 2, last_token])
        out = EngineInstance._get_extra_outputs(_FakeInstance(), resp, num_all_ids, gen_config)
        return out['routed_experts'].shape[0]

    def test_partial_rollout_abort_eos_offbyone_is_trimmed(self):
        # Regression: prompt=53, engine generated 427 tokens (last is EOS) so the
        # engine reports raw routed_experts length 479, but completion_tokens is
        # 426 (the aborted EOS is dropped). Must be trimmed to 478.
        prompt, generated = 53, 427
        num_all_ids = prompt + generated
        n = self._routed_len(ResponseType.CANCEL, num_all_ids - 1, num_all_ids, EOS, _gen_config())
        assert n == 478

    def test_normal_completion_keeps_full_length(self):
        # A clean stop with include_stop_str_in_output keeps the EOS, so no trim.
        prompt, generated = 53, 427
        num_all_ids = prompt + generated
        cfg = _gen_config(include_stop_str_in_output=True)
        n = self._routed_len(ResponseType.FINISH, num_all_ids - 1, num_all_ids, EOS, cfg)
        assert n == 479

    def test_overlong_routed_experts_is_clamped(self):
        # Defensive clamp: never return more than the expected number of entries
        # (last token is not a stop token, so nothing is dropped for that reason).
        prompt, generated = 53, 427
        num_all_ids = prompt + generated
        n = self._routed_len(ResponseType.CANCEL, num_all_ids + 5, num_all_ids, 999, _gen_config())
        assert n == num_all_ids - 1
