# Copyright (c) OpenMMLab. All rights reserved.
"""Tests for lmdeploy.pytorch.engine.input_process module."""

import pytest

from lmdeploy.pytorch.engine.input_process import (
    BaseModelInputProcessor,
    DefaultModelInputProcessor,
    PreprocessInputResult,
)


class TestPreprocessInputResult:
    """Test PreprocessInputResult dataclass."""

    def test_basic_creation_with_input_ids(self):
        """Test basic creation with input_ids only."""
        result = PreprocessInputResult(input_ids=[1, 2, 3])
        assert result.input_ids == [1, 2, 3]
        assert result.input_multimodals is None
        assert result.model_metas is None

    def test_creation_with_multimodals(self):
        """Test creation with multimodal inputs."""
        mm_inputs = {'images': ['image1.jpg']}
        result = PreprocessInputResult(
            input_ids=[1, 2, 3],
            input_multimodals=mm_inputs,
        )
        assert result.input_ids == [1, 2, 3]
        assert result.input_multimodals == mm_inputs

    def test_creation_with_model_metas(self):
        """Test creation with model metadata."""
        metas = {'layer_count': 32, 'hidden_size': 4096}
        result = PreprocessInputResult(
            input_ids=[1, 2, 3],
            model_metas=metas,
        )
        assert result.model_metas == metas

    def test_creation_with_all_fields(self):
        """Test creation with all fields populated."""
        result = PreprocessInputResult(
            input_ids=[1, 2, 3],
            input_multimodals={'images': ['img.jpg']},
            model_metas={'meta': 'data'},
        )
        assert result.input_ids == [1, 2, 3]
        assert result.input_multimodals is not None
        assert result.model_metas is not None


class TestBaseModelInputProcessor:
    """Test BaseModelInputProcessor abstract class."""

    def test_cannot_instantiate_abstract_class(self):
        """Test that abstract class cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseModelInputProcessor()

    def test_abstract_class_requires_implementation(self):
        """Test that abstract class requires implementation of preprocess_input."""
        # Trying to instantiate without implementing abstract method should fail
        with pytest.raises(TypeError):
            class IncompleteProcessor(BaseModelInputProcessor):
                pass
            IncompleteProcessor()

    def test_subclass_must_implement_preprocess_input(self):
        """Test that subclasses must implement preprocess_input."""
        class ValidProcessor(BaseModelInputProcessor):
            def preprocess_input(self, input_ids, input_mms=None, **kwargs):
                return PreprocessInputResult(input_ids=input_ids)

        processor = ValidProcessor()
        result = processor.preprocess_input([1, 2, 3])
        assert isinstance(result, PreprocessInputResult)
        assert result.input_ids == [1, 2, 3]


class TestDefaultModelInputProcessor:
    """Test DefaultModelInputProcessor class."""

    def test_basic_preprocess_input(self):
        """Test basic input preprocessing."""
        processor = DefaultModelInputProcessor()
        input_ids = [1, 2, 3, 4, 5]
        result = processor.preprocess_input(input_ids)

        assert isinstance(result, PreprocessInputResult)
        assert result.input_ids == input_ids
        assert result.input_multimodals is None

    def test_preprocess_with_multimodals(self):
        """Test preprocessing with multimodal inputs."""
        processor = DefaultModelInputProcessor()
        input_ids = [1, 2, 3]
        mm_inputs = {'images': ['image1.jpg', 'image2.jpg']}
        result = processor.preprocess_input(input_ids, input_mms=mm_inputs)

        assert result.input_ids == input_ids
        assert result.input_multimodals == mm_inputs

    def test_preprocess_preserves_input_order(self):
        """Test that input order is preserved."""
        processor = DefaultModelInputProcessor()
        input_ids = list(range(100))
        result = processor.preprocess_input(input_ids)

        assert result.input_ids == input_ids
        assert len(result.input_ids) == 100

    def test_preprocess_empty_input(self):
        """Test preprocessing empty input."""
        processor = DefaultModelInputProcessor()
        result = processor.preprocess_input([])

        assert result.input_ids == []
        assert isinstance(result, PreprocessInputResult)

    def test_preprocess_with_kwargs(self):
        """Test preprocessing with additional kwargs."""
        processor = DefaultModelInputProcessor()
        input_ids = [1, 2, 3]
        # Should accept extra kwargs without error
        result = processor.preprocess_input(
            input_ids,
            extra_param='value',
            another_param=123,
        )
        assert result.input_ids == input_ids

    def test_processor_is_instance_of_base(self):
        """Test that DefaultModelInputProcessor is instance of base class."""
        processor = DefaultModelInputProcessor()
        assert isinstance(processor, BaseModelInputProcessor)

    def test_multiple_calls_independence(self):
        """Test that multiple calls are independent."""
        processor = DefaultModelInputProcessor()

        result1 = processor.preprocess_input([1, 2, 3])
        result2 = processor.preprocess_input([4, 5, 6])

        assert result1.input_ids == [1, 2, 3]
        assert result2.input_ids == [4, 5, 6]
        # Results should be independent
        assert result1.input_ids != result2.input_ids


class TestInputProcessorEdgeCases:
    """Test edge cases for input processors."""

    def test_large_input_ids(self):
        """Test processing large input sequences."""
        processor = DefaultModelInputProcessor()
        large_input = list(range(10000))
        result = processor.preprocess_input(large_input)

        assert len(result.input_ids) == 10000
        assert result.input_ids == large_input

    def test_special_token_ids(self):
        """Test processing with special token IDs."""
        processor = DefaultModelInputProcessor()
        # Common special tokens: BOS=1, EOS=2, PAD=0
        input_ids = [1, 100, 200, 2, 0, 0]
        result = processor.preprocess_input(input_ids)

        assert result.input_ids == input_ids

    def test_negative_token_ids(self):
        """Test processing with negative token IDs (edge case)."""
        processor = DefaultModelInputProcessor()
        input_ids = [-1, -2, -3]
        result = processor.preprocess_input(input_ids)

        # Processor should pass through without validation
        assert result.input_ids == input_ids

    def test_duplicate_token_ids(self):
        """Test processing with duplicate token IDs."""
        processor = DefaultModelInputProcessor()
        input_ids = [1, 1, 2, 2, 3, 3]
        result = processor.preprocess_input(input_ids)

        assert result.input_ids == input_ids
