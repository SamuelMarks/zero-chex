import zero_chex as chex


def test_tree_all():
    assert chex._src.tree_util.tree_all({"a": True, "b": True})
    assert not chex._src.tree_util.tree_all({"a": True, "b": False})


def test_tree_flatten_with_path():
    leaves, treedef = chex._src.tree_util.tree_flatten_with_path({"a": 1, "b": 2})
    assert len(leaves) == 2
    assert treedef is None


def test_tree_structure():
    assert chex._src.tree_util.tree_structure({"a": 1}) is not None
