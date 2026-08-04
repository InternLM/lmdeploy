import torch

from lmdeploy.pytorch.model_inputs import DPMeta, ModelInputs, ModelInputsDelta, StepContext, VisionModelInputs
from lmdeploy.pytorch.multimodal.data_type import MultiModalData


def _make_model_inputs(is_decoding: bool, dp_is_decoding: bool | None = None):
    dp_meta = None
    if dp_is_decoding is not None:
        dp_meta = DPMeta(dp_is_decoding=dp_is_decoding)
    return ModelInputs(
        input_ids=torch.zeros((1, 1), dtype=torch.long),
        seq_length=torch.ones(1, dtype=torch.long),
        history_lengths=torch.zeros(1, dtype=torch.long),
        block_offsets=torch.zeros((1, 1), dtype=torch.long),
        is_decoding=is_decoding,
        num_ignored_history=torch.zeros(1, dtype=torch.long),
        max_q_seqlen=1,
        max_kv_seqlen=1,
        sum_kv_seqlen=1,
        dp_meta=dp_meta,
    )


def test_model_inputs_global_is_decoding_uses_local_without_dp_meta():
    assert _make_model_inputs(is_decoding=True).global_is_decoding()
    assert not _make_model_inputs(is_decoding=False).global_is_decoding()


def test_model_inputs_global_is_decoding_uses_dp_global_state():
    assert not _make_model_inputs(is_decoding=True, dp_is_decoding=False).global_is_decoding()
    assert _make_model_inputs(is_decoding=False, dp_is_decoding=True).global_is_decoding()


def test_step_context_global_is_decoding_uses_dp_global_state():
    step_ctx = StepContext(
        input_ids=torch.zeros((1, 1), dtype=torch.long),
        model_config=None,
        cache_config=None,
        block_offsets=torch.zeros((1, 1), dtype=torch.long),
        position_ids=torch.zeros((1, 1), dtype=torch.long),
        attention_mask=None,
        q_seqlens=torch.ones(1, dtype=torch.long),
        kv_seqlens=torch.ones(1, dtype=torch.long),
        q_start_loc=torch.zeros(1, dtype=torch.long),
        kv_caches=[],
        is_decoding=True,
        sum_kv_seqlen=1,
        dp_meta=DPMeta(dp_is_decoding=False),
    )
    assert not step_ctx.global_is_decoding()


def test_model_inputs_logprob_metadata_clone_device_and_step_lifecycle():
    inputs = _make_model_inputs(is_decoding=True)
    inputs.logits_indices = torch.tensor([0])
    inputs.seq_logit_length = torch.tensor([1])

    clone = inputs.clone()
    assert clone.logits_indices.tolist() == [0]
    assert clone.seq_logit_length.tolist() == [1]
    moved = clone.to_device('cpu')
    assert moved.logits_indices.tolist() == [0]
    assert moved.seq_logit_length.device.type == 'cpu'

    stepped = inputs.step(torch.tensor([[4]]))
    assert stepped.logits_indices is None
    assert stepped.seq_logit_length is None


def test_model_inputs_record_stream_matches_device_owned_fields():
    recorded = []

    class _CudaTensor(torch.Tensor):

        @staticmethod
        def __new__(cls):
            return torch.Tensor._make_subclass(cls, torch.empty(1), False)

        @property
        def is_cuda(self):
            return True

        def record_stream(self, stream):
            recorded.append((id(self), stream))

    stream = object()
    input_ids = _CudaTensor()
    embedding = _CudaTensor()
    embedding_range = _CudaTensor()
    multimodal_data = _CudaTensor()
    multimodal_meta = _CudaTensor()
    untouched_model_meta = _CudaTensor()
    vision_inputs = VisionModelInputs(
        input_embeddings=[[embedding]],
        input_embedding_ranges=[embedding_range],
        input_multimodals=[{
            'image': [MultiModalData(data=multimodal_data, start=0, meta={'shape': multimodal_meta})]
        }],
    )
    inputs = _make_model_inputs(is_decoding=False)
    inputs.input_ids = input_ids
    inputs.vision_inputs = vision_inputs
    inputs.model_metas = [{'persistent': untouched_model_meta}]

    inputs.record_stream(stream)

    assert set(recorded) == {
        (id(input_ids), stream),
        (id(embedding), stream),
        (id(embedding_range), stream),
        (id(multimodal_data), stream),
        (id(multimodal_meta), stream),
    }


def test_model_inputs_delta_record_stream_records_tensor_fields():
    recorded = []

    class _CudaTensor(torch.Tensor):

        @staticmethod
        def __new__(cls):
            return torch.Tensor._make_subclass(cls, torch.empty(1), False)

        @property
        def is_cuda(self):
            return True

        def record_stream(self, stream):
            recorded.append((id(self), stream))

    stream = object()
    indices = _CudaTensor()
    block_offsets = _CudaTensor()
    delta = ModelInputsDelta(indices=indices,
                             block_offsets=block_offsets,
                             indice_cpu=None,
                             max_q_seqlen=1,
                             max_kv_seqlen=1,
                             sum_kv_seqlen=1)

    delta.record_stream(stream)

    assert recorded == [
        (id(indices), stream),
        (id(block_offsets), stream),
    ]
