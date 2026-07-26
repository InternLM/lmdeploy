import numpy as np

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


def test_parent_assignment_maintains_both_sides():
    root = _make_node(0)
    child = _make_node(1)

    child.parent = root

    assert child.parent is root
    assert child.is_attached()
    assert root.children == {child.hash_key: child}
    assert child.path_from_root() == [child]

    child.parent = None

    assert child.parent is None
    assert not child.is_attached()
    assert root.children == {}


def test_reparent_invalidates_checkpoint_paths_in_subtree():
    old_root = _make_node(0)
    new_root = _make_node(10)
    child = _make_node(1)
    grandchild = _make_node(2)
    child.parent = old_root
    grandchild.parent = child
    child.state_checkpoint = NodeStateCheckpoint(slot=0, exact_match_data=object())
    grandchild.state_checkpoint = NodeStateCheckpoint(slot=1, exact_match_data=object())
    child_epoch = child._topology_epoch
    grandchild_epoch = grandchild._topology_epoch

    child.parent = new_root

    assert child.hash_key not in old_root.children
    assert new_root.children[child.hash_key] is child
    assert child.state_checkpoint.exact_match_data is None
    assert grandchild.state_checkpoint.exact_match_data is None
    assert child._topology_epoch == child_epoch + 1
    assert grandchild._topology_epoch == grandchild_epoch + 1


def test_replacing_child_detaches_and_invalidates_displaced_subtree():
    root = _make_node(0)
    displaced = _make_node(1)
    descendant = _make_node(2)
    displaced.parent = root
    descendant.parent = displaced
    displaced.state_checkpoint = NodeStateCheckpoint(slot=0, exact_match_data=object())
    descendant.state_checkpoint = NodeStateCheckpoint(slot=1, exact_match_data=object())
    displaced_epoch = displaced._topology_epoch
    descendant_epoch = descendant._topology_epoch
    replacement = _make_node(displaced.hash_key)

    replacement.parent = root

    assert displaced.parent is None
    assert root.children[replacement.hash_key] is replacement
    assert displaced.state_checkpoint.exact_match_data is None
    assert descendant.state_checkpoint.exact_match_data is None
    assert displaced._topology_epoch == displaced_epoch + 1
    assert descendant._topology_epoch == descendant_epoch + 1
