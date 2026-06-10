def tree_map(f, tree, is_leaf=None):
    if isinstance(tree, dict):
        return {k: tree_map(f, v, is_leaf) for k, v in tree.items()}
    elif isinstance(tree, list):
        return [tree_map(f, v, is_leaf) for v in tree]
    elif isinstance(tree, tuple):
        return tuple(tree_map(f, v, is_leaf) for v in tree)
    else:
        return f(tree)


def tree_leaves(tree):
    leaves = []

    def f(x):
        leaves.append(x)
        return x

    tree_map(f, tree)
    return leaves


def tree_all(tree):
    return all(tree_leaves(tree))


def tree_flatten_with_path(tree):
    leaves = tree_leaves(tree)
    return [(None, leaf) for leaf in leaves], None


def tree_structure(tree):
    return None
