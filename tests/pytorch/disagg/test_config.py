# Copyright (c) OpenMMLab. All rights reserved.
"""Unit tests for Disaggregated Serving configuration."""

import enum

import pytest

from lmdeploy.pytorch.disagg.config import (
    DistServeEngineConfig,
    DistServeRDMAConfig,
    DistServeTCPConfig,
    DistServeNVLinkConfig,
    EngineRole,
    MigrationBackend,
    MooncakeEngineConfig,
    RDMALinkType,
    ServingStrategy,
)


class TestServingStrategy:
    """Tests for ServingStrategy enum."""

    def test_serving_strategy_has_hybrid(self):
        """Test that Hybrid strategy exists."""
        assert hasattr(ServingStrategy, 'Hybrid')
        assert ServingStrategy.Hybrid.name == 'Hybrid'

    def test_serving_strategy_has_distserve(self):
        """Test that DistServe strategy exists."""
        assert hasattr(ServingStrategy, 'DistServe')
        assert ServingStrategy.DistServe.name == 'DistServe'

    def test_serving_strategy_values_are_unique(self):
        """Test that enum values are unique."""
        values = [strategy.value for strategy in ServingStrategy]
        assert len(values) == len(set(values))

    def test_serving_strategy_count(self):
        """Test that there are exactly 2 strategies."""
        assert len(list(ServingStrategy)) == 2


class TestEngineRole:
    """Tests for EngineRole enum."""

    def test_engine_role_has_hybrid(self):
        """Test that Hybrid role exists."""
        assert hasattr(EngineRole, 'Hybrid')
        assert EngineRole.Hybrid.name == 'Hybrid'

    def test_engine_role_has_prefill(self):
        """Test that Prefill role exists."""
        assert hasattr(EngineRole, 'Prefill')
        assert EngineRole.Prefill.name == 'Prefill'

    def test_engine_role_has_decode(self):
        """Test that Decode role exists."""
        assert hasattr(EngineRole, 'Decode')
        assert EngineRole.Decode.name == 'Decode'

    def test_engine_role_values_are_unique(self):
        """Test that enum values are unique."""
        values = [role.value for role in EngineRole]
        assert len(values) == len(set(values))

    def test_engine_role_count(self):
        """Test that there are exactly 3 roles."""
        assert len(list(EngineRole)) == 3


class TestMigrationBackend:
    """Tests for MigrationBackend enum."""

    def test_migration_backend_has_dlslime(self):
        """Test that DLSlime backend exists."""
        assert hasattr(MigrationBackend, 'DLSlime')
        assert MigrationBackend.DLSlime.name == 'DLSlime'

    def test_migration_backend_has_mooncake(self):
        """Test that Mooncake backend exists."""
        assert hasattr(MigrationBackend, 'Mooncake')
        assert MigrationBackend.Mooncake.name == 'Mooncake'

    def test_migration_backend_values_are_unique(self):
        """Test that enum values are unique."""
        values = [backend.value for backend in MigrationBackend]
        assert len(values) == len(set(values))

    def test_migration_backend_count(self):
        """Test that there are exactly 2 backends."""
        assert len(list(MigrationBackend)) == 2


class TestRDMALinkType:
    """Tests for RDMALinkType enum."""

    def test_rdma_link_type_has_ib(self):
        """Test that IB link type exists."""
        assert hasattr(RDMALinkType, 'IB')
        assert RDMALinkType.IB.name == 'IB'

    def test_rdma_link_type_has_roce(self):
        """Test that RoCE link type exists."""
        assert hasattr(RDMALinkType, 'RoCE')
        assert RDMALinkType.RoCE.name == 'RoCE'

    def test_rdma_link_type_values_are_unique(self):
        """Test that enum values are unique."""
        values = [link_type.value for link_type in RDMALinkType]
        assert len(values) == len(set(values))

    def test_rdma_link_type_count(self):
        """Test that there are exactly 2 link types."""
        assert len(list(RDMALinkType)) == 2


class TestDistServeRDMAConfig:
    """Tests for DistServeRDMAConfig."""

    def test_default_with_gdr_is_true(self):
        """Test that with_gdr defaults to True."""
        config = DistServeRDMAConfig()
        assert config.with_gdr is True

    def test_default_link_type_is_roce(self):
        """Test that link_type defaults to RoCE."""
        config = DistServeRDMAConfig()
        assert config.link_type == RDMALinkType.RoCE

    def test_can_set_with_gdr_false(self):
        """Test that with_gdr can be set to False."""
        config = DistServeRDMAConfig(with_gdr=False)
        assert config.with_gdr is False

    def test_can_set_link_type_ib(self):
        """Test that link_type can be set to IB."""
        config = DistServeRDMAConfig(link_type=RDMALinkType.IB)
        assert config.link_type == RDMALinkType.IB

    def test_can_set_both_parameters(self):
        """Test setting both parameters."""
        config = DistServeRDMAConfig(with_gdr=False, link_type=RDMALinkType.IB)
        assert config.with_gdr is False
        assert config.link_type == RDMALinkType.IB

    def test_config_is_pydantic_model(self):
        """Test that config is a Pydantic BaseModel."""
        from pydantic import BaseModel
        assert isinstance(DistServeRDMAConfig(), BaseModel)

    def test_config_validation_rejects_invalid_types(self):
        """Test that invalid types are rejected."""
        with pytest.raises(Exception):  # Pydantic ValidationError or ValueError
            DistServeRDMAConfig(with_gdr="invalid")


class TestDistServeTCPConfig:
    """Tests for DistServeTCPConfig (placeholder)."""

    def test_tcp_config_exists(self):
        """Test that TCP config class exists."""
        assert DistServeTCPConfig is not None

    def test_tcp_config_can_be_instantiated(self):
        """Test that TCP config can be instantiated."""
        config = DistServeTCPConfig()
        assert config is not None

    def test_tcp_config_is_pydantic_model(self):
        """Test that TCP config is a Pydantic BaseModel."""
        from pydantic import BaseModel
        assert isinstance(DistServeTCPConfig(), BaseModel)


class TestDistServeNVLinkConfig:
    """Tests for DistServeNVLinkConfig (placeholder)."""

    def test_nvlink_config_exists(self):
        """Test that NVLink config class exists."""
        assert DistServeNVLinkConfig is not None

    def test_nvlink_config_can_be_instantiated(self):
        """Test that NVLink config can be instantiated."""
        config = DistServeNVLinkConfig()
        assert config is not None

    def test_nvlink_config_is_pydantic_model(self):
        """Test that NVLink config is a Pydantic BaseModel."""
        from pydantic import BaseModel
        assert isinstance(DistServeNVLinkConfig(), BaseModel)


class TestDistServeEngineConfig:
    """Tests for DistServeEngineConfig."""

    def test_basic_config_creation(self):
        """Test basic engine config creation."""
        config = DistServeEngineConfig(
            tp_size=1,
            ep_size=1,
            dp_size=1,
            pp_size=None,
            dp_rank=0,
            block_size=16,
            num_cpu_blocks=100,
            num_gpu_blocks=1000,
        )
        assert config.tp_size == 1
        assert config.ep_size == 1
        assert config.dp_size == 1
        assert config.pp_size is None
        assert config.dp_rank == 0
        assert config.block_size == 16
        assert config.num_cpu_blocks == 100
        assert config.num_gpu_blocks == 1000

    def test_config_with_pp_size(self):
        """Test config with pipeline parallel size."""
        config = DistServeEngineConfig(
            tp_size=2,
            ep_size=1,
            dp_size=2,
            pp_size=4,
            dp_rank=1,
            block_size=16,
            num_cpu_blocks=200,
            num_gpu_blocks=2000,
        )
        assert config.pp_size == 4
        assert config.dp_rank == 1

    def test_config_requires_all_fields(self):
        """Test that all required fields must be provided."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            DistServeEngineConfig(
                tp_size=1,
                # Missing other required fields
            )

    def test_config_accepts_different_parallel_sizes(self):
        """Test config with various parallel sizes."""
        configs = [
            (1, 1, 1, None),   # Single GPU
            (2, 1, 1, None),   # TP=2
            (1, 1, 2, None),   # DP=2
            (2, 1, 2, 2),      # TP=2, DP=2, PP=2
            (4, 8, 1, None),   # EP=8, TP=4
        ]

        for tp, ep, dp, pp in configs:
            config = DistServeEngineConfig(
                tp_size=tp,
                ep_size=ep,
                dp_size=dp,
                pp_size=pp,
                dp_rank=0,
                block_size=16,
                num_cpu_blocks=100,
                num_gpu_blocks=1000,
            )
            assert config.tp_size == tp
            assert config.ep_size == ep
            assert config.dp_size == dp
            assert config.pp_size == pp

    def test_config_dp_rank_can_be_nonzero(self):
        """Test that dp_rank can be non-zero."""
        for rank in [0, 1, 2, 3]:
            config = DistServeEngineConfig(
                tp_size=1,
                ep_size=1,
                dp_size=4,
                pp_size=None,
                dp_rank=rank,
                block_size=16,
                num_cpu_blocks=100,
                num_gpu_blocks=1000,
            )
            assert config.dp_rank == rank

    def test_config_cache_parameters(self):
        """Test cache-related parameters."""
        config = DistServeEngineConfig(
            tp_size=1,
            ep_size=1,
            dp_size=1,
            pp_size=None,
            dp_rank=0,
            block_size=32,
            num_cpu_blocks=500,
            num_gpu_blocks=5000,
        )
        assert config.block_size == 32
        assert config.num_cpu_blocks == 500
        assert config.num_gpu_blocks == 5000

    def test_config_is_pydantic_model(self):
        """Test that config is a Pydantic BaseModel."""
        from pydantic import BaseModel
        config = DistServeEngineConfig(
            tp_size=1,
            ep_size=1,
            dp_size=1,
            pp_size=None,
            dp_rank=0,
            block_size=16,
            num_cpu_blocks=100,
            num_gpu_blocks=1000,
        )
        assert isinstance(config, BaseModel)


class TestMooncakeEngineConfig:
    """Tests for MooncakeEngineConfig."""

    def test_mooncake_config_inherits_from_distserve(self):
        """Test that MooncakeEngineConfig inherits from DistServeEngineConfig."""
        assert issubclass(MooncakeEngineConfig, DistServeEngineConfig)

    def test_mooncake_config_creation(self):
        """Test Mooncake config creation with all parameters."""
        config = MooncakeEngineConfig(
            tp_size=2,
            ep_size=1,
            dp_size=1,
            pp_size=None,
            dp_rank=0,
            block_size=16,
            num_cpu_blocks=100,
            num_gpu_blocks=1000,
        )
        assert config.tp_size == 2
        assert config.ep_size == 1
        assert config.dp_size == 1
        assert config.block_size == 16

    def test_mooncake_config_has_same_fields_as_parent(self):
        """Test that Mooncake config has same fields as parent."""
        parent_config = DistServeEngineConfig(
            tp_size=1,
            ep_size=1,
            dp_size=1,
            pp_size=None,
            dp_rank=0,
            block_size=16,
            num_cpu_blocks=100,
            num_gpu_blocks=1000,
        )
        mooncake_config = MooncakeEngineConfig(
            tp_size=1,
            ep_size=1,
            dp_size=1,
            pp_size=None,
            dp_rank=0,
            block_size=16,
            num_cpu_blocks=100,
            num_gpu_blocks=1000,
        )

        assert mooncake_config.tp_size == parent_config.tp_size
        assert mooncake_config.ep_size == parent_config.ep_size
        assert mooncake_config.dp_size == parent_config.dp_size
        assert mooncake_config.pp_size == parent_config.pp_size
        assert mooncake_config.dp_rank == parent_config.dp_rank
        assert mooncake_config.block_size == parent_config.block_size
        assert mooncake_config.num_cpu_blocks == parent_config.num_cpu_blocks
        assert mooncake_config.num_gpu_blocks == parent_config.num_gpu_blocks

    def test_mooncake_config_is_pydantic_model(self):
        """Test that Mooncake config is a Pydantic BaseModel."""
        from pydantic import BaseModel
        config = MooncakeEngineConfig(
            tp_size=1,
            ep_size=1,
            dp_size=1,
            pp_size=None,
            dp_rank=0,
            block_size=16,
            num_cpu_blocks=100,
            num_gpu_blocks=1000,
        )
        assert isinstance(config, BaseModel)


class TestEnumSerialization:
    """Tests for enum serialization and comparison."""

    def test_serving_strategy_comparison(self):
        """Test ServingStrategy enum comparison."""
        assert ServingStrategy.Hybrid != ServingStrategy.DistServe
        assert ServingStrategy.Hybrid == ServingStrategy.Hybrid

    def test_engine_role_comparison(self):
        """Test EngineRole enum comparison."""
        roles = list(EngineRole)
        for i, role1 in enumerate(roles):
            for j, role2 in enumerate(roles):
                if i == j:
                    assert role1 == role2
                else:
                    assert role1 != role2

    def test_migration_backend_comparison(self):
        """Test MigrationBackend enum comparison."""
        assert MigrationBackend.DLSlime != MigrationBackend.Mooncake

    def test_rdma_link_type_comparison(self):
        """Test RDMALinkType enum comparison."""
        assert RDMALinkType.IB != RDMALinkType.RoCE


class TestConfigIntegration:
    """Integration tests for config combinations."""

    def test_rdma_config_with_engine_config(self):
        """Test using RDMA config with engine config."""
        rdma_config = DistServeRDMAConfig(with_gdr=True, link_type=RDMALinkType.RoCE)
        engine_config = DistServeEngineConfig(
            tp_size=2,
            ep_size=1,
            dp_size=1,
            pp_size=None,
            dp_rank=0,
            block_size=16,
            num_cpu_blocks=100,
            num_gpu_blocks=1000,
        )

        assert rdma_config.with_gdr is True
        assert rdma_config.link_type == RDMALinkType.RoCE
        assert engine_config.tp_size == 2

    def test_mooncake_config_with_rdma(self):
        """Test Mooncake config with RDMA settings."""
        mooncake_config = MooncakeEngineConfig(
            tp_size=4,
            ep_size=1,
            dp_size=1,
            pp_size=None,
            dp_rank=0,
            block_size=16,
            num_cpu_blocks=200,
            num_gpu_blocks=2000,
        )
        rdma_config = DistServeRDMAConfig(with_gdr=True)

        assert mooncake_config.tp_size == 4
        assert rdma_config.with_gdr is True

    def test_multiple_engine_configs_for_pd_pair(self):
        """Test creating configs for prefill-decode pair."""
        prefill_config = DistServeEngineConfig(
            tp_size=4,
            ep_size=1,
            dp_size=1,
            pp_size=None,
            dp_rank=0,
            block_size=16,
            num_cpu_blocks=100,
            num_gpu_blocks=1000,
        )

        decode_config = DistServeEngineConfig(
            tp_size=2,
            ep_size=1,
            dp_size=2,
            pp_size=2,
            dp_rank=0,
            block_size=16,
            num_cpu_blocks=200,
            num_gpu_blocks=2000,
        )

        # Prefill uses TP=4 for computation
        assert prefill_config.tp_size == 4
        assert prefill_config.pp_size is None

        # Decode uses TP=2, PP=2 for latency optimization
        assert decode_config.tp_size == 2
        assert decode_config.pp_size == 2
