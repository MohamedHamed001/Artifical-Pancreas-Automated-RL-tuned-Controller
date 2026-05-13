"""Mocked HTTP smoke test for the checkpoint downloader."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pytest

from ap_rl.scripts import download_checkpoints as mod
from ap_rl.utils.checkpoint_filenames import ACTOR_BEST


class _FakeResponse:
    def __init__(self, status_code: int, payload: bytes) -> None:
        self.status_code = status_code
        self._payload = payload

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *exc) -> None:
        return None

    def iter_content(self, chunk_size: int = 1 << 16) -> Iterable[bytes]:
        for offset in range(0, len(self._payload), chunk_size):
            yield self._payload[offset : offset + chunk_size]


def test_download_succeeds_with_mocked_http(monkeypatch, tmp_path: Path) -> None:
    payload = b"\x00" * 4096

    def fake_get(url: str, stream: bool = True, timeout: int = 60):
        return _FakeResponse(200, payload)

    monkeypatch.setattr("requests.get", fake_get, raising=False)

    rc = mod.download(
        base_url="https://example.invalid/releases/v0.1.0",
        files=(ACTOR_BEST,),
        dest_dir=tmp_path,
        force=True,
    )
    assert rc == 0
    out = tmp_path / ACTOR_BEST
    assert out.exists()
    assert out.read_bytes() == payload


def test_download_errors_when_no_url_configured(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("AP_RL_CHECKPOINT_URL", raising=False)
    with pytest.raises(SystemExit) as exc:
        mod.download(base_url=None, dest_dir=tmp_path)
    assert "no checkpoint URL configured" in str(exc.value)


def test_download_returns_nonzero_on_http_failure(monkeypatch, tmp_path: Path) -> None:
    def fake_get(url: str, stream: bool = True, timeout: int = 60):
        return _FakeResponse(404, b"")

    monkeypatch.setattr("requests.get", fake_get, raising=False)
    rc = mod.download(
        base_url="https://example.invalid",
        files=(ACTOR_BEST,),
        dest_dir=tmp_path,
        force=True,
    )
    assert rc == 1


def test_download_skips_existing_unless_forced(monkeypatch, tmp_path: Path) -> None:
    target = tmp_path / ACTOR_BEST
    target.write_bytes(b"existing")

    calls: list[str] = []

    def fake_get(url: str, stream: bool = True, timeout: int = 60):
        calls.append(url)
        return _FakeResponse(200, b"new")

    monkeypatch.setattr("requests.get", fake_get, raising=False)
    rc = mod.download(
        base_url="https://example.invalid",
        files=(ACTOR_BEST,),
        dest_dir=tmp_path,
        force=False,
    )
    assert rc == 0
    assert calls == []  # no HTTP calls because file already exists
    assert target.read_bytes() == b"existing"


def test_download_legacy_h5_url_saves_as_weights_h5(monkeypatch, tmp_path: Path) -> None:
    """Release hosts only legacy *.h5; local file should still be *.weights.h5."""

    def fake_get(url: str, stream: bool = True, timeout: int = 60):
        if url.endswith("diabetes_actor_best.weights.h5"):
            return _FakeResponse(404, b"")
        if url.endswith("diabetes_actor_best.h5"):
            return _FakeResponse(200, b"LEGACY")
        return _FakeResponse(404, b"")

    monkeypatch.setattr("requests.get", fake_get, raising=False)
    rc = mod.download(
        base_url="https://example.invalid/r",
        files=(ACTOR_BEST,),
        dest_dir=tmp_path,
        force=True,
    )
    assert rc == 0
    out = tmp_path / ACTOR_BEST
    assert out.read_bytes() == b"LEGACY"
