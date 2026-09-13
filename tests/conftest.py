import pytest


@pytest.fixture(autouse=True)
def _default_stock_data_root(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Most tests never read/write real data — they fake or tmp_path the pieces
    that do. But a few call CLI entrypoints that resolve ``DataRoot`` before any
    mock kicks in, so it must resolve to *something* even when the real
    ``STOCK_DATA_ROOT`` (set locally via .envrc) is not present, e.g. in CI."""
    monkeypatch.setenv("STOCK_DATA_ROOT", str(tmp_path))
