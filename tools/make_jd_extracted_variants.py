"""產生黃金集 10 格的 extracted 變體,交給成員 A 做雙變體評測。

【這支在做什麼】
成員 A 的黃金集,JD 需求全部來自型錄的結構化欄位(source="catalog")。
但正式環境沒有型錄——使用者手貼一段散文,required_skills 由抽取器產生
(source="extracted")。她在 catalog 分佈上校準的門檻,不保證轉移到
extracted 分佈。

解法是同一組人工標記跑兩種輸入:wants 是人對 JD 的判斷,跟 required_skills
怎麼產生的無關,所以同一份標記可以當兩種輸入的答案卡。兩邊跑完的差額
就是抽取損失的直接測量,零額外標記成本。

這支負責產出「輸入 B」。

執行:
    python tools/make_jd_extracted_variants.py
    python tools/make_jd_extracted_variants.py --dry-run    # 不打 API,看會送什麼

產出:
    data/jd_variants_extracted.json

環境變數(讀 .env):
    OPENAI_API_KEY   必要
    OPENAI_MODEL     預設 gpt-4o-mini
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from app.pipeline.jd_extract import extract_jd  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
CASES = ROOT / "fixtures" / "golden" / "interview"
OUT = ROOT / "data" / "jd_variants_extracted.json"


# ---------------------------------------------------------------------------
# LLM 呼叫:注入式,所以這支是唯一 import SDK 的地方
# ---------------------------------------------------------------------------


def make_llm(model: str):
    """回傳 (system, user) -> str。

    抽取器本身不 import 任何 SDK,注入點只有這裡——測試傳假的,
    這支傳真的。
    """
    from openai import OpenAI

    client = OpenAI()

    def call(system: str, user: str) -> str:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0,  # 抽取要可重現,不要創意
        )
        return resp.choices[0].message.content or ""

    return call


def fake_llm(system: str, user: str) -> str:
    """--dry-run 用。不打 API,回一個固定的合法輸出。"""
    return '["示範技能A", "示範技能B"]'


# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="不打 API,只檢查流程與輸入")
    ap.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    ap.add_argument("--sleep", type=float, default=0.5, help="每次呼叫之間的間隔秒數")
    args = ap.parse_args()

    if args.dry_run:
        llm = fake_llm
        print("[dry-run] 不會呼叫 API\n")
    else:
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except ImportError:
            pass
        if not os.getenv("OPENAI_API_KEY"):
            print("錯誤:找不到 OPENAI_API_KEY。請確認 .env 已設定。")
            return 1
        llm = make_llm(args.model)
        print(f"模型:{args.model}\n")

    cases = sorted(CASES.glob("ivw-*.json"))
    if not cases:
        print(f"錯誤:{CASES} 底下找不到 case 檔")
        return 1

    variants: dict[str, dict] = {}
    total_catalog = 0
    total_extracted = 0

    for i, path in enumerate(cases):
        case = json.loads(path.read_text(encoding="utf-8"))
        cid = case["case_id"]
        jd = case["jd"]
        desc = jd.get("description", "") or ""
        catalog = jd.get("required_skills") or jd.get("requiredSkills") or []

        result = extract_jd(
            desc,
            llm,
            title=jd.get("title", "") or jd.get("job_title", ""),
            company=jd.get("company", "") or "",
        )
        got = result.jd.required_skills

        # 純字面重疊,只當粗略觀察值。真正的比對是 A 那邊經過正規化的。
        cat_norm = {c.casefold().replace(" ", "") for c in catalog}
        overlap = sum(1 for g in got if g.casefold().replace(" ", "") in cat_norm)

        variants[cid] = {
            "required_skills": got,
            "source": "extracted",
            "_catalog_count": len(catalog),
            "_extracted_count": len(got),
            "_literal_overlap": overlap,
            "_notices": result.notices,
        }
        total_catalog += len(catalog)
        total_extracted += len(got)

        print(f"{cid}  型錄 {len(catalog):>2} 條 → 抽出 {len(got):>2} 條(字面重疊 {overlap})")
        if result.notices:
            for n in result.notices:
                print(f"          ⚠ {n}")
        print(f"          {'、'.join(got[:8])}{' …' if len(got) > 8 else ''}")

        if not args.dry_run and i < len(cases) - 1:
            time.sleep(args.sleep)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "_meta": {
            "generated_by": "tools/make_jd_extracted_variants.py",
            "model": "fake" if args.dry_run else args.model,
            "temperature": 0,
            "note": (
                "輸入是各 case 的 jd.description(散文),輸出供 interview_eval "
                "的 --jd-source extracted 使用。字面重疊只是粗略觀察值,"
                "真正的比對請用正規化後的結果。"
            ),
        },
        "variants": variants,
    }
    OUT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )

    print(f"\n{'─' * 60}")
    print(f"型錄合計 {total_catalog} 條,抽出合計 {total_extracted} 條")
    print(f"輸出:{OUT}")
    if args.dry_run:
        print("\n這是 dry-run,內容是假的。確認流程沒問題後拿掉 --dry-run 再跑一次。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
