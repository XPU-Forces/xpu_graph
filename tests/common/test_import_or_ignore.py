from xpu_graph.utils import ImportOrIgnoreMetaClass


def test_import_or_ignore_meta_class():
    class SupposedNOTExist(metaclass=ImportOrIgnoreMetaClass, __modules_to_import__="wahahah"):
        pass

    assert SupposedNOTExist is None

    class SupposedExist(metaclass=ImportOrIgnoreMetaClass, __modules_to_import__="torch"):
        pass

    assert SupposedExist is not None
