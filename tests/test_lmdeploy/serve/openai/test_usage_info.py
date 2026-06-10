from lmdeploy.serve.openai.protocol import PromptTokensDetails, build_usage_info


def test_build_usage_info_includes_cached_tokens():
    usage = build_usage_info(prompt_tokens=100, completion_tokens=20, cached_tokens=64)
    assert usage.prompt_tokens == 100
    assert usage.completion_tokens == 20
    assert usage.total_tokens == 120
    assert usage.prompt_tokens_details == PromptTokensDetails(cached_tokens=64)


def test_build_usage_info_defaults_cached_tokens_to_zero():
    usage = build_usage_info(prompt_tokens=50, completion_tokens=10)
    assert usage.prompt_tokens_details == PromptTokensDetails(cached_tokens=0)


def test_usage_info_model_dump():
    usage = build_usage_info(prompt_tokens=10, completion_tokens=5, cached_tokens=3)
    payload = usage.model_dump()
    assert payload['prompt_tokens_details'] == {'cached_tokens': 3}
