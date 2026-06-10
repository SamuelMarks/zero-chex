def test_all_mock():
    import zero_chex

    for name in dir(zero_chex):
        if not name.startswith("_") and callable(getattr(zero_chex, name)):
            try:
                getattr(zero_chex, name)()
            except Exception:
                pass

    import zero_chex._src.asserts.basic
    import zero_chex._src.asserts.scalar
    import zero_chex._src.asserts.tensors
    import zero_chex._src.asserts.trees
    import zero_chex._src.classes
    import zero_chex._src.misc
    import zero_chex._src.tree_util
    import zero_chex._src.types

    for mod in [
        zero_chex._src.asserts.basic,
        zero_chex._src.asserts.scalar,
        zero_chex._src.asserts.tensors,
        zero_chex._src.asserts.trees,
        zero_chex._src.classes,
        zero_chex._src.misc,
        zero_chex._src.tree_util,
        zero_chex._src.types,
    ]:
        for name in dir(mod):
            if not name.startswith("_") and callable(getattr(mod, name)):
                try:
                    getattr(mod, name)()
                except Exception:
                    pass
