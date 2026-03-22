
import pytest

from lmdeploy import GenerationConfig, Tokenizer
from lmdeploy.messages import LinearPrefixCacheStats, ScheduleMetrics
from lmdeploy.metrics.stats import SchedulerStats
from lmdeploy.utils import get_hf_gen_cfg


def test_engine_generation_config():
    tokenizer = Tokenizer('internlm/internlm-chat-7b')
    config = GenerationConfig(n=3, stop_words=['<eoa>'])
    stop_token_ids = tokenizer.encode('<eoa>', add_bos=False)
    config.convert_stop_bad_words_to_ids(tokenizer)
    assert stop_token_ids == config.stop_token_ids
    assert isinstance(config.stop_token_ids, list) and \
        isinstance(config.stop_token_ids[0], int)


@pytest.mark.parametrize('model_path', [
    'deepseek-ai/DeepSeek-V3',
    'Qwen/Qwen2.5-32B-Instruct',
    'internlm/internlm3-8b-instruct',
])
def test_update_from_hf_gen_cfg(model_path):
    tokenizer = Tokenizer(model_path)
    model_cfg = get_hf_gen_cfg(model_path)

    generation_config = GenerationConfig()
    generation_config.update_from_hf_gen_cfg(model_cfg, tokenizer.eos_token_id)
    assert generation_config.stop_token_ids is not None


def test_scheduler_stats_linear_prefix_merge():
    s = SchedulerStats()
    s.update_from_schedule_metrics(
        ScheduleMetrics(active_seqs=1,
                        waiting_seqs=0,
                        total_blocks=8,
                        active_blocks=2,
                        cached_blocks=0,
                        free_blocks=4,
                        prefix_cache_hit_rate=0.0))
    s.update_linear_prefix_cache_stats(
        LinearPrefixCacheStats(publish_ok=3,
                               publish_miss=1,
                               publish_pool_exhausted=0,
                               prefix_match_skipped_alpha=2,
                               linear_restore=5))
    assert s.linear_prefix_publish_ok == 3
    assert s.linear_prefix_publish_miss == 1
    assert s.linear_prefix_publish_pool_exhausted == 0
    assert s.linear_prefix_match_skipped_alpha == 2
    assert s.linear_prefix_match_restored == 5
    assert s.gpu_cache_usage == pytest.approx(0.5)
