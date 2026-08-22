"""A5 漏講點管線測試。不打真 LLM。

重點在**否決路徑**與**機械攔截**,不在 happy path。
實測漏講 precision 約 0.47,超過一半是假指控,而 why 是 LLM 寫的——
它會替假指控寫出有說服力的理由。所以擋得住比生得出更重要。
"""

from __future__ import annotations

import json

import pytest

from app.contracts.interview_protocols import GapCandidate
from app.pipeline.missing_points import (
    MAX_WHY_CHARS,
    check_why,
    generate_missing_points,
    parse_verdicts,
)


def cand(display="SQL(sk:sql)", kind="hard", jd_evidence="需具備 SQL 撰寫能力") -> GapCandidate:
    return GapCandidate(
        skill_id="sk:sql",
        display=display,
        kind=kind,
        resume_experience_ids=["e1"],
        resume_evidence="SQL",
        jd_evidence=jd_evidence,
        weight=0.9,
    )


def with_ids(items: list[dict]) -> list[dict]:
    return [{"id": i, **x} for i, x in enumerate(items, 1)]


def llm_returning(payload) -> object:
    def _call(system: str, user: str) -> str:
        return payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)

    return _call


GOOD_WHY = "這份職缺明列需要撰寫查詢與處理資料,是日常工作的主要內容之一。"


# ---------------------------------------------------------------------------
# 解析
# ---------------------------------------------------------------------------


def test_parses_clean_json() -> None:
    v, n = parse_verdicts('[{"id": 1, "why": "x"}]', 1)
    assert v[1]["why"] == "x"
    assert n == []


def test_strips_markdown_fence() -> None:
    v, _ = parse_verdicts('```json\n[{"id": 1, "why": "x"}]\n```', 1)
    assert v[1]["why"] == "x"


@pytest.mark.parametrize("bad", ["", "抱歉我無法處理", "{}", "null", "[[]]"])
def test_bad_output_vetoes_everything_not_raises(bad: str) -> None:
    """解析失敗要全部否決,不是拋例外也不是放行。"""
    v, n = parse_verdicts(bad, 3)
    assert v == {}


def test_missing_ids_are_recorded_as_model_failure() -> None:
    """模型漏回要記成明確的失誤,不要說成「否決」——那會掩蓋問題。"""
    v, n = parse_verdicts('[{"id": 1, "why": "x"}]', 3)
    assert set(v) == {1}
    assert any("沒有回覆理由" in x for x in n)


def test_out_of_range_id_dropped() -> None:
    v, n = parse_verdicts('[{"id": 9, "why": "x"}]', 2)
    assert v == {}
    assert any("不在候選範圍" in x for x in n)


def test_verdict_without_id_dropped() -> None:
    """沒有 id 的裁決無法對齊,只能丟。"""
    v, n = parse_verdicts('[{"why": "x"}]', 1)
    assert v == {}
    assert any("沒有 id" in x for x in n)


def test_extra_element_does_not_shift_alignment() -> None:
    """這是第一版的真實 bug:模型多回一個元素,整批位移一格,
    保留下來的理由被安到別的技能上。id 對齊之後不可能發生。"""
    raw = '[{"note":"以下是我的分析"},{"id":2,"why":"b"}]'
    v, _ = parse_verdicts(raw, 3)
    assert set(v) == {2}
    assert v[2]["why"] == "b"


# ---------------------------------------------------------------------------
# 機械攔截:這是防假指控的最後一道
# ---------------------------------------------------------------------------


def test_rejects_resume_narration() -> None:
    """why 不可敘述使用者的經歷——標籤與經歷的對應不可靠,會錯誤歸因。"""
    assert check_why("你在圖書館志工經歷中展現了很好的進度控管能力,值得一提。")


def test_rejects_advice_sentence() -> None:
    """why 回答為什麼重要,不是該怎麼做。建議句屬於 improvements。"""
    assert check_why("你應該多提到自己使用資料庫查詢的實際經驗與成果表現。")


@pytest.mark.parametrize("why", ["", "太短", "很重要"])
def test_rejects_too_short(why: str) -> None:
    assert check_why(why)


def test_rejects_too_long() -> None:
    assert check_why("要" * (MAX_WHY_CHARS + 1))


def test_accepts_jd_side_reasoning() -> None:
    assert check_why(GOOD_WHY) is None


# ---------------------------------------------------------------------------
# 端到端
# ---------------------------------------------------------------------------


def test_empty_candidates_skips_the_call() -> None:
    """沒有候選就不呼叫 LLM。沒履歷或沒交集是正常情況,不該浪費呼叫。"""
    called = []

    def spy(s, u):
        called.append(1)
        return "[]"

    out, notices = generate_missing_points([], "逐字稿", "JD", spy)
    assert out == []
    assert called == []


def test_llm_failure_degrades_not_raises() -> None:
    """一份少了漏講點區塊的報告仍然可用,500 則完全不可用。"""

    def boom(s, u):
        raise RuntimeError("timeout")

    out, notices = generate_missing_points([cand()], "t", "jd", boom)
    assert out == []
    assert any("生成失敗" in n for n in notices)


def test_bad_why_is_dropped_not_rewritten() -> None:
    """違規的 why 是丟掉不是改寫。

    改寫等於我們自己編一個理由,而問題本來就是理由編得太好。
    """
    out, notices = generate_missing_points(
        [cand()],
        "t",
        "jd",
        llm_returning(with_ids([{"why": "你在社團經歷中展現了很強的資料處理能力表現。"}])),
    )
    assert out == []
    assert any("已丟棄" in n for n in notices)


def test_happy_path() -> None:
    out, notices = generate_missing_points(
        [cand()], "t", "jd", llm_returning(with_ids([{"why": GOOD_WHY}]))
    )
    assert len(out) == 1
    assert out[0].point == "SQL(sk:sql)"
    assert out[0].why == GOOD_WHY


def test_truncates_to_max_points() -> None:
    cands = [cand(display=f"技能{i}") for i in range(8)]
    verdicts = [{"why": GOOD_WHY} for _ in cands]
    out, notices = generate_missing_points(
        cands, "t", "jd", llm_returning(with_ids(verdicts)), max_points=5
    )
    assert len(out) == 5
    assert any("取前 5 條" in n for n in notices)


def test_order_follows_candidate_weight_order() -> None:
    """候選已依 weight 排序,輸出順序必須跟著,不可重排。"""
    cands = [cand(display=f"技能{i}") for i in range(3)]
    out, _ = generate_missing_points(
        cands, "t", "jd", llm_returning(with_ids([{"why": GOOD_WHY}] * 3))
    )
    assert [p.point for p in out] == ["技能0", "技能1", "技能2"]


def test_user_message_never_carries_resume_evidence() -> None:
    """履歷證據不進 prompt——它只是一個標籤,撐不起敘述,
    放進去只會誘使模型拿它敘事。"""
    from app.pipeline.missing_points import build_user_message

    msg = build_user_message([cand()], "逐字稿內容", "JD 內容")
    assert "SQL" in msg
    assert "履歷" not in msg


# ---------------------------------------------------------------------------
# 實測回饋:第一版否決機制有害,預設關閉
# ---------------------------------------------------------------------------


def test_rejects_narration_with_any_subject() -> None:
    """第一版漏掉「候選人在專題中展現了…」——主詞不是「你」、名詞不是「專案」。"""
    for why in (
        "這份 JD 要求專案管理,候選人在專題中展現了有效的時間管理與溝通技巧。",
        "應徵者在實習經驗裡累積了相關能力,因此這項需求特別重要值得一提。",
        "你在社團經歷中展現了很強的資料處理能力表現值得補充說明。",
    ):
        assert check_why(why), f"沒攔到:{why}"


def test_accepts_pure_jd_side_reasoning() -> None:
    """不提使用者做過什麼的理由要放行,否則機械攔截會把好的也殺掉。"""
    for why in (
        "這份職缺把日常帳務與憑證核對列為主要工作,沒有這項能力就無法承擔核心職責。",
        "職缺描述把版本控制放在團隊協作的段落,顯示這是日常開發流程的一部分。",
    ):
        assert check_why(why) is None, f"誤攔:{why}"


# ---------------------------------------------------------------------------
# 第二輪實測回饋:否決整個移除
# ---------------------------------------------------------------------------


def test_candidate_without_jd_evidence_is_filtered_in_code() -> None:
    """「有沒有材料」是確定性事實,程式判就好,不要問模型。

    交給模型會讓「材料不足」與「模型自己想砍」混在一起,之後查不出是哪一種。
    """
    called = []

    def spy(sys_, user):
        called.append(user)
        return "[]"

    out, notices = generate_missing_points(
        [cand(jd_evidence="")], "t", "jd", spy
    )
    assert out == []
    assert called == [], "沒有材料的候選不該送去呼叫 LLM"
    assert any("沒有 JD 依據" in n for n in notices)


def test_llm_cannot_filter_candidates() -> None:
    """這一層不做過濾。實測兩輪都是虧的:

        不經 LLM   17 條候選,7 命中   precision 0.412   recall 7/8
        經過 LLM   10 條保留,4 命中   precision 0.400   recall 4/8

    否決掉的裡面有 3 條是真漏講,而且集中在候選清單後段——
    那是注意力衰減,不是判斷。
    """
    out, _ = generate_missing_points(
        [cand()], "t", "jd", llm_returning(with_ids([{"why": GOOD_WHY}]))
    )
    assert len(out) == 1, "keep=false 不該讓候選消失,過濾不是這一層的工作"


def test_non_taiwan_terms_are_replaced_not_dropped() -> None:
    """用詞正確與否跟這條漏講點對不對無關,丟掉等於為了措辭殺掉一個可能正確的點。"""
    why = "這份 JD 要求熟悉前端框架,這對實現用戶界面與交互功能相當關鍵。"
    out, notices = generate_missing_points(
        [cand()], "t", "jd", llm_returning(with_ids([{"why": why}]))
    )
    assert len(out) == 1
    assert "使用者介面" in out[0].why
    assert "用戶" not in out[0].why
    assert any("正規化" in n for n in notices)


def test_narration_still_dropped() -> None:
    """敘事違規仍然丟棄——那是安全性問題,會產生看起來很具體但可能全錯的歸因。"""
    out, notices = generate_missing_points(
        [cand()],
        "t",
        "jd",
        llm_returning(with_ids([{"why": "候選人在專題中展現了很強的資料處理能力表現。"}])),
    )
    assert out == []
    assert any("已丟棄" in n for n in notices)


def test_empty_why_cannot_be_shown() -> None:
    """沒有理由的漏講點不能顯示,但要記進 notices 讓人看得見。"""
    out, notices = generate_missing_points(
        [cand()], "t", "jd", llm_returning(with_ids([{"why": ""}]))
    )
    assert out == []
    assert any("漏寫理由" in n for n in notices)
