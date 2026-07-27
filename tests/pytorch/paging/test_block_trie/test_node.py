import numpy as np
import pytest

from lmdeploy.pytorch.paging.block_trie import Node
from lmdeploy.pytorch.paging.block_trie.node import NodeStateCheckpoint


def _make_node(hash_key: int):
    return Node(hash_key=hash_key,
                block=hash_key,
                tokens=np.array([hash_key], dtype=np.int64))


def test_state_checkpoint_is_allocated_lazily():
    node = _make_node(1)

    assert node.state_checkpoint is None

    node.state_checkpoint = NodeStateCheckpoint(slot=3)

    assert node.state_checkpoint.slot == 3
    assert not node.state_checkpoint.published


def test_attach_and_detach_leaf_maintain_both_sides():
    root = _make_node(0)
    child = _make_node(1)

    child.attach_to(root)

    assert child.parent is root
    assert child.is_attached()
    assert root.children == {child.hash_key: child}
    assert child.path_from_root() == [child]

    assert child.detach_leaf()

    assert child.parent is None
    assert not child.is_attached()
    assert root.children == {}


def test_attached_node_cannot_move_to_another_parent():
    old_root = _make_node(0)
    new_root = _make_node(10)
    child = _make_node(1)
    child.attach_to(old_root)

    with pytest.raises(RuntimeError, match='Cannot reattach'):
        child.attach_to(new_root)

    assert old_root.children[child.hash_key] is child
    assert new_root.children == {}
    assert child.parent is old_root


def test_attach_does_not_replace_an_existing_child():
    root = _make_node(0)
    displaced = _make_node(1)
    displaced.attach_to(root)
    replacement = _make_node(displaced.hash_key)

    with pytest.raises(RuntimeError, match='Cannot replace'):
        replacement.attach_to(root)

    assert displaced.parent is root
    assert replacement.parent is None
    assert root.children[displaced.hash_key] is displaced


def test_non_leaf_cannot_detach():
    root = _make_node(0)
    parent = _make_node(1)
    child = _make_node(2)
    parent.attach_to(root)
    child.attach_to(parent)

    with pytest.raises(RuntimeError, match='Cannot detach a non-leaf'):
        parent.detach_leaf()
