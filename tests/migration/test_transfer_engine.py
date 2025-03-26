from lmdeploy.migration.engine import TransferEngine


def test_link():
    # Target
    target_engine = TransferEngine()
    target_engine.init_link(
        0,
        'mlx5_bond_0',
    )

    # initiator
    source_engine = TransferEngine()
