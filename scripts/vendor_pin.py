"""記錄 app/vendor/ 底下複本的指紋。

用途:把「複本不可修改」這條規則從口頭約定變成可檢查的東西。
複製新檔案進來、或有意識地重新同步時執行一次,其餘時間不要跑。

執行:
  python scripts/vendor_pin.py

產出:
  app/vendor/MANIFEST.json
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
VENDOR = ROOT / "app" / "vendor"
MANIFEST = VENDOR / "MANIFEST.json"

SKIP = {"__init__.py", "MANIFEST.json", "README.md"}


def fingerprint(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def has_source_header(path: pathlib.Path) -> bool:
    """檢查第一行有沒有來源標頭。沒有標頭的複本日後無從追溯。"""
    try:
        first = path.read_text(encoding="utf-8").splitlines()[0]
    except (UnicodeDecodeError, IndexError):
        return False
    return "VENDORED from" in first


def main() -> int:
    if not VENDOR.exists():
        print("app/vendor/ 不存在")
        return 1

    files = sorted(
        p for p in VENDOR.glob("*.py") if p.name not in SKIP and not p.name.startswith("_")
    )

    missing_header = [p.name for p in files if not has_source_header(p)]
    if missing_header:
        print("以下檔案缺少來源標頭,補上後再執行一次:")
        for name in missing_header:
            print(f"  {name}")
        print("\n標頭格式:")
        print("  # VENDORED from CareerSandboxModule @ <hash> on <日期>. DO NOT MODIFY.")
        return 1

    manifest = {p.name: fingerprint(p) for p in files}
    MANIFEST.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if not manifest:
        print("app/vendor/ 目前沒有複本,已寫入空的 MANIFEST.json")
    else:
        print(f"已記錄 {len(manifest)} 個複本的指紋:")
        for name in manifest:
            print(f"  {name}")
    print(f"\n輸出:{MANIFEST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
