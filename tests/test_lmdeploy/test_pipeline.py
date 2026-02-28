import gc

import pytest
import torch

from lmdeploy import GenerationConfig, PytorchEngineConfig, TurbomindEngineConfig, pipeline
from lmdeploy.messages import Response

MODEL_ID = 'Qwen/Qwen3-8B'


@pytest.mark.parametrize('backend', ['pytorch', 'turbomind'], scope='class')
class TestBackendInference:
    """Test class grouping all tests for each backend."""

    @pytest.fixture(scope='class', autouse=True)
    def backend_config(self, backend):
        """Parametrized backend configuration for all tests."""

        if backend == 'pytorch':
            return PytorchEngineConfig(session_len=4096, max_batch_size=4, tp=1)
        elif backend == 'turbomind':
            return TurbomindEngineConfig(session_len=4096, max_batch_size=4, tp=1)
        else:
            raise ValueError(f'Unknown backend type: {backend}')

    @pytest.fixture(scope='class', autouse=True)
    def pipe(self, backend_config):
        """Shared pipeline instance across all tests in class."""
        pipe = pipeline(MODEL_ID, backend_config=backend_config)
        yield pipe
        pipe.close()
        del pipe
        gc.collect()
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

    def test_infer_single_string(self, pipe):
        """Test infer with single string prompt."""
        prompt = 'Hello, how are you?'
        response = pipe.infer(prompt)

        assert isinstance(response, Response)
        assert hasattr(response, 'text')
        assert hasattr(response, 'generate_token_len')
        assert hasattr(response, 'input_token_len')
        assert len(response.text) > 0

    def test_infer_batch_strings(self, pipe):
        """Test infer with batch of string prompts."""
        prompts = ['What is AI?', 'Explain quantum computing', 'Tell me a joke']
        responses = pipe.infer(prompts)

        assert isinstance(responses, list)
        assert len(responses) == len(prompts)
        for resp in responses:
            assert isinstance(resp, Response)
            assert len(resp.text) > 0

    def test_infer_openai_format(self, pipe):
        """Test infer with OpenAI-style message format."""
        prompts = [[{
            'role': 'user',
            'content': 'What is machine learning?'
        }], [{
            'role': 'user',
            'content': 'Define deep learning'
        }]]
        responses = pipe.infer(prompts)

        assert len(responses) == 2
        for resp in responses:
            assert isinstance(resp, Response)

    def test_infer_with_generation_config(self, pipe):
        """Test infer with custom GenerationConfig."""
        gen_config = GenerationConfig(max_new_tokens=50, temperature=0.5, top_p=0.9, top_k=40, do_sample=True)
        prompt = 'Write a haiku about nature'
        response = pipe.infer(prompt, gen_config=gen_config)

        assert isinstance(response, Response)
        assert response.generate_token_len <= 50

    def test_call_method(self, pipe):
        """Test __call__ method as shortcut for infer."""
        prompt = 'What is Python?'
        response = pipe(prompt)

        assert isinstance(response, Response)
        assert len(response.text) > 0

    def test_stream_infer_single(self, pipe):
        """Test stream_infer with single prompt."""
        prompt = 'Count from 1 to 5'
        generator = pipe.stream_infer(prompt)

        chunks = []
        for chunk in generator:
            chunks.append(chunk)
            assert isinstance(chunk, Response)

        assert len(chunks) > 0
        full_text = ''.join([c.text for c in chunks])
        assert len(full_text) > 0

    def test_stream_infer_batch(self, pipe):
        """Test stream_infer with batch prompts."""
        prompts = ['First prompt', 'Second prompt']
        generator = pipe.stream_infer(prompts)

        responses = {}
        for chunk in generator:
            chunks = responses.setdefault(chunk.index, [])
            chunks.append(chunk)
            assert isinstance(chunk, Response)

        assert len(responses) == len(prompts)
        for chunks in responses.values():
            full_text = ''.join([c.text for c in chunks])
            assert len(full_text) > 0

    def test_stream_infer_with_session(self, pipe):
        """Test stream_infer with session for multi-turn context."""
        session = pipe.session()
        prompt1 = 'Hello! My name is Alice.'
        step = 0

        # First turn
        generator = pipe.stream_infer(prompts=prompt1,
                                      sessions=session,
                                      gen_config=GenerationConfig(max_new_tokens=30),
                                      sequence_start=True,
                                      sequence_end=False,
                                      enable_thinking=False)
        resp = None
        for out in generator:
            resp = resp.extend(out) if resp else out

        step += resp.generate_token_len + resp.input_token_len

        response1 = resp.text

        assert response1

        # Second turn should remember context
        prompt2 = 'What is my name?'
        session.step = step
        generator = pipe.stream_infer(prompts=prompt2,
                                      sessions=session,
                                      gen_config=GenerationConfig(max_new_tokens=30),
                                      sequence_start=False,
                                      sequence_end=False,
                                      enable_thinking=False)

        resp = None
        for out in generator:
            resp = resp.extend(out) if resp else out

        step += out.generate_token_len + out.input_token_len

        response2 = resp.text

        assert 'alice' in response2.lower()

    def test_chat_streaming(self, pipe):
        """Test chat method with streaming output."""
        prompt = 'Tell me a short story'
        session = pipe.session()

        generator = pipe.chat(prompt=prompt,
                              session=session,
                              stream_response=True,
                              gen_config=GenerationConfig(max_new_tokens=50))

        chunks = []
        for chunk in generator:
            chunks.append(chunk)
            assert isinstance(chunk, Response)

        assert len(chunks) > 0
        assert session.response is not None
        assert session.step > 0

    def test_chat_non_streaming(self, pipe):
        """Test chat method with non-streaming output."""
        prompt = 'What is 2+2?'
        session = pipe.chat(prompt=prompt,
                            stream_response=False,
                            gen_config=GenerationConfig(max_new_tokens=20),
                            enable_thinking=False)

        assert session is not None
        assert hasattr(session, 'response')
        assert hasattr(session, 'history')
        assert len(session.history) == 1
        assert '4' in session.response.text or 'four' in session.response.text.lower()

    def test_chat_multi_turn(self, pipe):
        """Test chat method with multi-turn conversation."""
        # First turn
        session = pipe.chat(prompt='My favorite color is blue.',
                            stream_response=False,
                            gen_config=GenerationConfig(max_new_tokens=30),
                            enable_thinking=False)

        # Second turn should remember context
        session = pipe.chat(prompt='What is my favorite color?',
                            session=session,
                            stream_response=False,
                            gen_config=GenerationConfig(max_new_tokens=30),
                            enable_thinking=False)

        assert 'blue' in session.response.text.lower()
        assert len(session.history) == 2

    def test_session_creation(self, pipe):
        """Test session method to create new sessions."""
        session1 = pipe.session()
        session2 = pipe.session()

        assert session1 is not None
        assert session2 is not None
        assert session1 != session2

    def test_get_ppl_single(self, pipe):
        """Test get_ppl with single input."""
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

        text = 'This is a test sentence.'
        input_ids = tokenizer.encode(text, return_tensors='pt')[0].tolist()

        ppl = pipe.get_ppl(input_ids)

        assert isinstance(ppl, list)
        assert len(ppl) == 1
        assert isinstance(ppl[0], float)
        assert ppl[0] > 0

    def test_get_ppl_batch(self, pipe):
        """Test get_ppl with batch inputs."""
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

        texts = ['First text.', 'Second text.']
        input_ids_list = [tokenizer.encode(text, return_tensors='pt')[0].tolist() for text in texts]

        ppl = pipe.get_ppl(input_ids_list)

        assert isinstance(ppl, list)
        assert len(ppl) == len(texts)
        for score in ppl:
            assert isinstance(score, float)
            assert score > 0

    def test_stream_infer_stream_response_parameter(self, pipe):
        """Test stream_infer stream_response parameter."""
        prompt = 'Test'
        gen = pipe.stream_infer(prompt, stream_response=True)
        assert hasattr(gen, '__iter__')

        results = list(gen)
        assert len(results) > 0

    @pytest.mark.parametrize('max_new_tokens', [10, 50, 100])
    def test_infer_different_max_tokens(self, pipe, max_new_tokens):
        """Parametrized test for different max_new_tokens values."""
        gen_config = GenerationConfig(max_new_tokens=max_new_tokens)
        prompt = 'Continue: Once upon a time'
        response = pipe.infer(prompt, gen_config=gen_config)

        assert response.generate_token_len <= max_new_tokens + 5

    def test_batch_infer_different_gen_configs(self, pipe):
        """Test batch infer with different GenerationConfig per prompt."""
        prompts = ['Short answer: What is AI?', 'Long answer: Explain ML']
        gen_configs = [GenerationConfig(max_new_tokens=20), GenerationConfig(max_new_tokens=50)]

        responses = pipe.infer(prompts, gen_config=gen_configs)

        assert len(responses) == 2
        assert responses[0].generate_token_len <= responses[1].generate_token_len + 10

    def test_infer_zero_tokens(self, pipe):
        """Test infer with max_new_tokens=0 to end generation immediately
        without producing tokens."""
        gen_config = GenerationConfig(max_new_tokens=0)
        prompt = 'This prompt should not generate any response'
        response = pipe.infer(prompt, gen_config=gen_config, enable_thinking=False)
        assert isinstance(response, Response)
        assert response.generate_token_len == 0
