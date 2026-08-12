"""複本完整性檢查。

把「app/vendor/ 底下的複本不可修改」這條規則變成會紅燈的東西。
口頭約定在三週的專案裡一定會失守,通常是為了修某個命中率問題順手改一行,
然後兩個 repo 就分岔,而且分岔看不見。

紅燈時的處理方式:
  a) 不小心改到 —— git checkout 還原
  b) 有意識地重新同步上游 —— 更新 app/vendor/README.md 的 commit hash,
     再執行 python scripts/vendor_pin.py
  c) 面試側需要調整 —— 不要改複本,在 app/pipeline/ 寫擴充層
"""

from __future__ import annotations

import hashlib
import json
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
VENDOR = ROOT / "app" / "vendor"
MANIFEST = VENDOR / "MANIFEST.json"

SKIP = {"__init__.py", "MANIFEST.json", "README.md"}


def _vendored_files() -> list[pathlib.Path]:
    if not VENDOR.exists():
        return []
    return sorted(
        p for p in VENDOR.glob("*.py") if p.name not in SKIP and not p.name.startswith("_")
    )


def _load_manifest() -> dict[str, str]:
    if not MANIFEST.exists():
        return {}
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_every_vendored_file_is_pinned() -> None:
    """複製進來的檔案必須登記指紋,否則等於沒有保護。"""
    files = _vendored_files()
    if not files:
        pytest.skip("app/vendor/ 目前沒有複本")

    manifest = _load_manifest()
    unpinned = [p.name for p in files if p.name not in manifest]
    assert not unpinned, (
        f"以下複本尚未登記:{unpinned}。"
        "執行 python scripts/vendor_pin.py"
    )


def test_vendored_files_unmodified() -> None:
    """複本內容不可改動。改動請寫在 app/pipeline/ 的擴充層。"""
    files = _vendored_files()
    if not files:
        pytest.skip("app/vendor/ 目前沒有複本")

    manifest = _load_manifest()
    changed = []
    for path in files:
        expected = manifest.get(path.name)
        if expected is None:
            continue  # 由上一個測試負責
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            changed.append(path.name)

    assert not changed, (
        f"以下複本被修改:{changed}。"
        "面試側的調整請寫在 app/pipeline/ 的擴充層,不要改複本。"
        "若這是有意識的上游同步,更新 app/vendor/README.md 後執行 "
        "python scripts/vendor_pin.py"
    )


def test_manifest_has_no_orphans() -> None:
    """manifest 裡不該有已經刪掉的檔案。"""
    manifest = _load_manifest()
    if not manifest:
        pytest.skip("尚無 manifest")

    present = {p.name for p in _vendored_files()}
    orphans = [name for name in manifest if name not in present]
    assert not orphans, (
        f"manifest 記錄了不存在的檔案:{orphans}。"
        "執行 python scripts/vendor_pin.py 重新產生"
    )


def test_vendored_files_carry_source_header() -> None:
    """每個複本第一行必須標明來源與 commit hash,否則日後無從追溯。"""
    files = _vendored_files()
    if not files:
        pytest.skip("app/vendor/ 目前沒有複本")

    missing = []
    for path in files:
        lines = path.read_text(encoding="utf-8").splitlines()
        if not lines or "VENDORED from" not in lines[0]:
            missing.append(path.name)

    assert not missing, (
        f"以下複本缺少來源標頭:{missing}。"
        "第一行格式:# VENDORED from CareerSandboxModule @ <hash> on <日期>. DO NOT MODIFY."
    )
