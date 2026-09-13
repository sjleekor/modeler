import importlib.metadata


def test_version_is_set():
    assert importlib.metadata.version("modeler")
