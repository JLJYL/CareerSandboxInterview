#!/usr/bin/env python3
"""重算 fixtures/scoring/answer_levels.json 裡的確定性統計。

text_stats 的規則改了就跑這一支。不跑的話 fixture 裡的數字會跟上線的
分岔,而分岔的症狀是「測試還是綠的」——測試吃的是 fixture 不是程式。

    python3 tools/refresh_answer_levels.py

【為什麼不在測試裡算】
第一版在測試裡用正則重寫了一份量化詞計數,題目 2 的「好」算出 8 處而
正式是 3 處。測試量的跟上線的不是同一件事,而那個差距讓分數看起來比
實際高了一整個帶。計數只能有一個來源。

【為什麼不在測試裡呼叫 text_stats】
TranscriptAnalyzer 會載 bge-m3,約 2.3GB。測試不能碰。
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "fixtures" / "scoring" / "answer_levels.json"


def main() -> int:
    from app.api.deps import load_stt_aliases, load_vocab
    from app.pipeline.transcript import TranscriptAnalyzer

    analyzer = TranscriptAnalyzer(
        load_vocab(),
        enable_semantic=False,
        engine_filler_policy="partial",
        stt_aliases=load_stt_aliases(),
    )

    fx = json.loads(FIXTURE.read_text(encoding="utf-8"))
    changed = []
    for case in fx["cases"]:
        for level in fx["levels"]:
            answer = case["answers"][level]
            stats = analyzer.text_stats(answer["text"])
            before = (answer.get("char_count"), answer.get("quantifier_count"))
            after = (stats.char_count, stats.quantifier_count)
            if before != after:
                changed.append((case["question"][:12], level, before, after))
            answer["char_count"], answer["quantifier_count"] = after

    fx["_stats_provenance"]["produced_at"] = datetime.date.today().isoformat()
    FIXTURE.write_text(
        json.dumps(fx, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    for question, level, before, after in changed:
        print(f"  {question}… / {level}: {before} → {after}")
    print(f"{len(changed)} 筆變動,已寫回 {FIXTURE.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
