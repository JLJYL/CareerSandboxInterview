"""事實判讀測試。不打真 LLM。

重點在**三件事**:

    形狀與標準答案對得上   as_label() 要能被 accepts() 吃,跟 fixture 同一個形狀
    失敗看得見             呼叫掛掉要標在 read_by 上,不能悄悄變成「沒有」
    引文沒落地就不採用     宣稱有但抄不出原文,判讀改成「沒有」

準確率不在這裡量。那要打真 API,在 `tools/star_read_eval.py`。
這裡驗的是「工具量到的那組設定有沒有被原封不動搬進來」。
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from app.pipeline.star_read import (
    ASSIGN,
    CORE_FACTS,
    ASSIGN_MIN_SENTENCES,
    FAILED,
    QUESTION,
    STAR_READ_SOURCES,
    AnswerFacts,
    FactRead,
    grounded,
    parse_assign,
    read_facts,
    read_one,
)
from app.pipeline.interviewer_spec import ACCEPT_RULES, DIM_STRUGGLE, STAR_PART_KEYS, STRUGGLE_FACT
from app.pipeline.turn_decision import struggle_dimension
from app.pipeline.transcript import SEGMENTATION_SOURCES, SEGMENT_LEVELS, segment_text
from app.prompts.star_read import (
    ASSIGN_SYSTEM,
    FACT_NAME,
    FACT_QUESTION,
    FACTS,
    LABEL_MEANING,
    STAR_LABELS,
    compose_fact_prompt,
    render_labels,
)

ANSWER = (
    "那個專案是公司要做一個新的銷售報表,原本都是人工整理。"
    "我把訂單資料接進資料庫,寫 SQL 把欄位對起來。"
    "上線之後每週整理時間從三小時降到二十分鐘。"
)
NO_PUNCT = "那時候公司要做一個報表然後我把資料接進資料庫後來每週快很多"


def obj(**kw) -> str:
    return json.dumps(kw, ensure_ascii=False)


def llm_of(script: dict[str, str], *, calls: list | None = None):
    """照 system prompt 的內容派回應。script 的鍵是要比對的片段。"""

    def call(system: str, user: str) -> str:
        if calls is not None:
            calls.append((system, user))
        for mark, out in script.items():
            if mark in system:
                return out
        return "{}"

    return call


def assign_all(label: str) -> str:
    """把每一句都掛同一個標籤。句數由 user 訊息決定,這裡給足夠多。"""
    return obj(**{str(i): label for i in range(1, 12)})


# ---------------------------------------------------------------------------
# 宣告一致性
# ---------------------------------------------------------------------------


def test_facts_names_and_questions_declared_once():
    """六件事的清單、中文名、問句必須對齊——缺一個就會印出空白的 prompt。"""
    assert set(FACTS) == set(FACT_NAME) == set(FACT_QUESTION)
    assert FACTS[: len(STAR_PART_KEYS)] == STAR_PART_KEYS


def test_every_label_has_a_meaning_and_reaches_the_prompt():
    """加標籤要同時加說明,而且說明要真的被寫進 prompt。

    取捨那個標籤是把任務從 0.86 推到 1.00 的原因。它只在 prompt 裡出現
    才有作用,寫在常數裡但沒 render 出去等於沒加。
    """
    assert set(STAR_LABELS) == set(LABEL_MEANING)
    block = render_labels()
    for label in STAR_LABELS:
        assert label in block
        assert LABEL_MEANING[label] in block
        assert label in ASSIGN_SYSTEM


def test_star_labels_cover_the_four_parts_plus_tradeoff():
    """分派的標籤要蓋住 STAR 四段,而且多一個取捨的位置。"""
    assert set(STAR_PART_KEYS) < set(STAR_LABELS)
    assert "取捨" in STAR_LABELS


def test_fact_prompt_asks_exactly_one_question():
    """單題問法一次只能問一件事——問兩件就回不出 present/quote 的形狀。"""
    for fact in FACTS:
        system = compose_fact_prompt(fact)
        asked = [f for f in FACTS if FACT_QUESTION[f] in system]
        assert asked == [fact]


def test_read_by_only_carries_declared_sources():
    """read_by 的值域就是 STAR_READ_SOURCES,報表照這個列。"""
    assert len(set(STAR_READ_SOURCES)) == 3
    facts = asyncio.run(
        read_facts(answer=ANSWER, llm=llm_of({"分派": assign_all("A"), "只判斷一件事": obj(present=False, quote="")}))
    )
    assert set(facts.read_by) <= set(FACTS)
    assert set(facts.read_by.values()) <= set(STAR_READ_SOURCES)


def test_label_shape_matches_the_golden_fixture():
    """判讀結果與標準答案必須是同一個形狀。

    `accepts()` 要用同一支函式吃兩邊:fixture 算期望值、線上算真實決定。
    形狀漂掉的那一刻,量到的數字會跟線上的行為沒有關係。
    """
    data = json.loads(
        Path("fixtures/golden/decision/answers_v1.json").read_text(encoding="utf-8")
    )
    for answer in data["answers"]:
        golden = answer["label"]
        mine = AnswerFacts.from_label(golden).as_label()
        assert set(mine) == set(golden)
        assert mine == {
            "star_parts": list(golden["star_parts"]),
            "quantified": golden["quantified"],
            "tradeoff": golden["tradeoff"],
        }


def test_the_ladder_derives_the_expected_follow_up_rates():
    """標註經過 AnswerFacts 再進 accepts(),推出來的追問率要跟量測一致。

    0.143 / 0.429 / 0.857 不是寫死的數字,是從七筆標註算出來的。
    這條接住的是:標註、`AcceptRule`、`AnswerFacts` 任何一邊動了而另外兩邊
    沒跟上——那時候量測還是跑得完,只是量到的東西換了。
    """
    data = json.loads(
        Path("fixtures/golden/decision/answers_v1.json").read_text(encoding="utf-8")
    )
    facts = [AnswerFacts.from_label(a["label"]) for a in data["answers"]]
    rates = {
        diff: sum(not f.accepted_by(rule) for f in facts) / len(facts)
        for diff, rule in ACCEPT_RULES.items()
    }
    assert [round(rates[d], 3) for d in ("新手", "中等", "困難")] == [0.143, 0.429, 0.857]


def test_a_read_answer_gets_the_gap_computed_not_asked():
    """判讀完之後缺口是算的。A3 那一筆:講了結果沒數字 → 結果沒量化。

    這是 1d 指錯得最清楚的一筆,模型 20/20 次寫「缺結果」。
    """
    facts = AnswerFacts.from_label(
        {"star_parts": ["S", "A", "R"], "quantified": False, "tradeoff": False}
    )
    assert facts.gap_for(ACCEPT_RULES["中等"]) == "結果沒量化"
    assert facts.gap_for(ACCEPT_RULES["新手"]) == ""


def test_an_incomplete_read_still_answers_but_the_caller_can_tell():
    """讀失敗的時候 accepts() 照樣算得出答案,而那個答案不該被採用。

    `complete` 就是那個分界。這一條存在是為了把它釘住:
    呼叫掛掉 → 四段都沒有 → 每個難度都「該追問」,
    而那跟「模型讀出來四段都沒有」在 accepts() 眼裡一模一樣。
    """

    def boom(system: str, user: str) -> str:
        raise RuntimeError("timeout")

    facts = asyncio.run(read_facts(answer=ANSWER, llm=boom))
    assert not facts.complete
    assert not facts.accepted_by(ACCEPT_RULES["新手"])
    assert facts.gap_for(ACCEPT_RULES["新手"])


def test_label_round_trip_keeps_the_star_order():
    """from_label 之後再 as_label 要拿回同一份。順序照 STAR_PART_KEYS,不照標註的寫法。"""
    facts = AnswerFacts.from_label(
        {"star_parts": ["R", "S"], "quantified": True, "tradeoff": False}
    )
    assert facts.star_parts == ("S", "R")
    assert facts.as_label()["star_parts"] == ["S", "R"]


# ---------------------------------------------------------------------------
# 引文落地
# ---------------------------------------------------------------------------


def test_claim_without_a_grounded_quote_is_dropped():
    """說有但抄不出原文 → 判讀改成沒有。這是抄原文那條規則的執行點。"""
    got = read_one(obj(present=True, quote="我當時的目標是把營收做大"), ANSWER)
    assert got == FactRead(False, "我當時的目標是把營收做大", False)


def test_grounded_quote_is_kept():
    got = read_one(obj(present=True, quote="我把訂單資料接進資料庫"), ANSWER)
    assert got.present and got.grounded


def test_quote_ignores_punctuation_and_spacing():
    """模型重排標點不該讓證據失效——比對前先正規化。"""
    assert grounded("我把訂單資料接進資料庫,寫SQL把欄位對起來", ANSWER)


def test_too_short_a_quote_is_not_evidence():
    """「報表」在每一筆回答裡都找得到,那不是證據。"""
    assert not grounded("報表", ANSWER)


def test_unparsable_output_reads_as_absent():
    """解析失敗當成「沒有」,不是當成「有」。"""
    assert read_one("抱歉,我無法判斷。", ANSWER) == FactRead(False, "", False)
    assert read_one("", ANSWER) == FactRead(False, "", False)


def test_saying_no_needs_no_quote():
    got = read_one(obj(present=False, quote=""), ANSWER)
    assert not got.present and got.grounded


def test_a_clean_no_is_distinguishable_from_a_broken_read():
    """兩者的 present 都是 False,但一個讀到了一個沒讀到。

    `parse_obj` 解析失敗回空 dict 不回 None,所以這條分支寫錯的時候
    沒讀到會長得跟乾淨的否認一模一樣。grounded 就是分界。
    """
    assert read_one(obj(present=False, quote=""), ANSWER).grounded
    assert not read_one("這不是 JSON", ANSWER).grounded


# ---------------------------------------------------------------------------
# 句子分派
# ---------------------------------------------------------------------------


def test_one_label_per_sentence_becomes_parts():
    sentences = ["那時候要做報表", "我把資料接進資料庫", "每週快很多"]
    got = parse_assign(obj(**{"1": "S", "2": "A", "3": "R"}), sentences)
    assert [f for f in STAR_PART_KEYS if got[f].present] == ["S", "A", "R"]
    assert got["A"].quote == "我把資料接進資料庫"
    assert not got["T"].present


def test_assignment_quotes_come_from_the_program():
    """分派模式的引文是被分派的那一句,所以必然落地——那一欄不是量測。"""
    sentences = ["我把資料接進資料庫"]
    got = parse_assign(obj(**{"1": "A"}), sentences)
    assert got["A"].grounded


def test_unknown_labels_and_out_of_range_indexes_are_dropped():
    """封閉集合修不回來的值不要猜。"""
    sentences = ["那時候要做報表", "我把資料接進資料庫"]
    got = parse_assign(obj(**{"1": "情境", "2": "A", "9": "R", "x": "S"}), sentences)
    assert not got["S"].present
    assert got["A"].present
    assert not got["R"].present


def test_tradeoff_label_does_not_become_a_star_part():
    """取捨有自己的標籤,但它不是 STAR 段落——掛上去不該多出一段。"""
    sentences = ["我判斷兩個月只夠做一件事就先放掉看板"]
    got = parse_assign(obj(**{"1": "取捨"}), sentences)
    assert not any(got[f].present for f in STAR_PART_KEYS)


def test_assignment_saying_nothing_is_not_assignment_failing():
    """全部丟掉 → 四段都沒有,但來源仍是 assign。

    「分派說沒有」跟「分派失敗」在 accepts() 眼裡一樣,在 read_by 上不一樣。
    """
    facts = asyncio.run(
        read_facts(
            answer=ANSWER,
            llm=llm_of({"分派": "完全不是 JSON", "只判斷一件事": obj(present=False, quote="")}),
        )
    )
    assert facts.star_parts == ()
    assert facts.read_by["T"] == ASSIGN
    assert facts.complete


# ---------------------------------------------------------------------------
# 失敗要看得見
# ---------------------------------------------------------------------------


def test_a_failed_call_is_marked_not_silently_absent():
    """呼叫掛掉 → read_by 記 failed,complete 為 False。

    1d 最貴的一課:降級後的結果長得跟「模型判斷沒有」完全一樣,
    只看結果看不出來,多燒一輪才查到。
    """

    def boom(system: str, user: str) -> str:
        raise RuntimeError("timeout")

    facts = asyncio.run(read_facts(answer=ANSWER, llm=boom))
    assert set(facts.failed) == set(FACTS)
    assert all(facts.read_by[f] == FAILED for f in FACTS)
    assert not facts.complete
    assert facts.star_parts == ()


def test_assignment_failing_does_not_take_the_single_questions_with_it():
    """三次呼叫是獨立的。分派掛掉,量化與取捨照樣讀得到。"""

    def half(system: str, user: str) -> str:
        if "分派" in system:
            raise RuntimeError("timeout")
        return obj(present=True, quote="上線之後每週整理時間從三小時降到二十分鐘")

    facts = asyncio.run(read_facts(answer=ANSWER, llm=half))
    assert set(facts.failed) == set(STAR_PART_KEYS)
    assert facts.read_by["quantified"] == QUESTION
    assert facts.quantified
    assert not facts.complete


def test_an_empty_answer_costs_no_calls():
    """沒有東西可讀就不要花三次呼叫,也不要假裝讀過。"""
    calls: list = []
    facts = asyncio.run(read_facts(answer="  ", llm=llm_of({}, calls=calls)))
    assert calls == []
    assert facts.read_by == {}
    assert not facts.complete
    assert facts.failed == ()


# ---------------------------------------------------------------------------
# 斷句決定走哪條路
# ---------------------------------------------------------------------------


def test_real_sentences_take_the_assignment_path():
    facts = asyncio.run(
        read_facts(
            answer=ANSWER,
            llm=llm_of({"分派": assign_all("A"), "只判斷一件事": obj(present=False, quote="")}),
        )
    )
    assert facts.segmentation == "punctuation"
    assert all(facts.read_by[p] == ASSIGN for p in STAR_PART_KEYS)
    assert facts.task_reliable


def test_an_unpunctuated_answer_falls_back_and_says_so():
    """裝置端引擎沒有標點,切不出句子就退回單題——而 T 在那條路徑上不可信。

    這條路徑存在不是因為它好,是因為它誠實。
    """
    facts = asyncio.run(
        read_facts(answer=NO_PUNCT, llm=llm_of({"只判斷一件事": obj(present=False, quote="")}))
    )
    assert facts.segmentation != "punctuation"
    assert all(facts.read_by[f] == QUESTION for f in FACTS)
    assert not facts.task_reliable


def test_one_sentence_still_takes_the_assignment_path():
    """實測的 1.00 裡 A1 就是一句話、走的就是分派。門檻設成 2 會跟量過的設定不同。"""
    assert ASSIGN_MIN_SENTENCES == 1
    one = "那個專案是公司要做一個新的銷售報表,原本都是人工在整理。"
    facts = asyncio.run(
        read_facts(
            answer=one,
            llm=llm_of({"分派": assign_all("S"), "只判斷一件事": obj(present=False, quote="")}),
        )
    )
    assert facts.read_by["S"] == ASSIGN
    assert facts.star_parts == ("S",)


# ---------------------------------------------------------------------------
# 斷句本身
# ---------------------------------------------------------------------------


def test_sentence_level_does_not_split_on_commas():
    """子句界與句界是兩個不同的事實。分派要的是整句,不是半句。"""
    text = "我把資料接進資料庫,寫 SQL 把欄位對起來。每週快很多。"
    sentences, src = segment_text(text, level="sentence")
    assert src == "punctuation"
    assert len(sentences) == 2
    assert "寫 SQL" in sentences[0]


def test_segment_levels_and_sources_are_declared():
    assert set(SEGMENT_LEVELS) == {"clause", "sentence"}
    for level in SEGMENT_LEVELS:
        _, src = segment_text(ANSWER, level=level)
        assert src in SEGMENTATION_SOURCES
    assert segment_text("", level="sentence") == ([], "unavailable")


def test_an_unknown_level_is_refused():
    """認不得的 level 要炸,不要默默回子句——那會讓分派拿到半句話。"""
    with pytest.raises(ValueError):
        segment_text(ANSWER, level="sentences")


# ---------------------------------------------------------------------------
# 第七項:自述困難
# ---------------------------------------------------------------------------


def test_struggle_is_declared_once_and_shared_with_dispatch():
    """鍵名只有一個來源。派發端讀的是 `STRUGGLE_FACT`,這邊用同一個常數。

    兩邊各寫一份字串的後果是:改了一邊,派發靜默回到休眠狀態,沒有任何錯誤。
    """
    assert FACTS[-1] == STRUGGLE_FACT
    assert FACT_NAME[STRUGGLE_FACT] == DIM_STRUGGLE
    assert STRUGGLE_FACT in FACT_QUESTION
    assert STRUGGLE_FACT not in CORE_FACTS
    assert set(CORE_FACTS) | {STRUGGLE_FACT} == set(FACTS)


def test_struggle_question_rules_out_the_two_easy_misreads():
    """問句要窄。成功敘事與「別人造成的困難」是最容易被讀成「有」的兩種,問句裡直接排除。"""
    q = FACT_QUESTION[STRUGGLE_FACT]
    assert "只算他明講" in q
    assert "別人或外部條件" in q
    assert "只講做成了什麼不算" in q


def test_struggle_is_asked_alone_even_on_the_assignment_path():
    """它不是 S/T/A/R 四段裡的一段,沒有標籤可掛,所以分派路徑也要單問。"""
    calls: list = []
    facts = asyncio.run(
        read_facts(
            answer=ANSWER,
            llm=llm_of(
                {"分派": assign_all("A"), "只判斷一件事": obj(present=False, quote="")},
                calls=calls,
            ),
        )
    )
    assert len(calls) == 4  # 一次分派 + 量化 + 取捨 + 自述困難
    assert facts.read_by[STRUGGLE_FACT] == QUESTION
    assert facts.read_by["T"] == ASSIGN


def test_a_failed_struggle_call_does_not_void_the_accept_decision():
    """輔助訊號掛了,放行判斷照舊能用。

    自述困難只決定派給誰,不進 accepts()。它的呼叫失敗如果讓 `complete` 變 False,
    整個決定會退回模型——為一個輔助訊號丟掉原本算得出來的結果。
    """

    def call(system: str, user: str) -> str:
        if FACT_QUESTION[STRUGGLE_FACT] in system:
            raise RuntimeError("timeout")
        if "分派" in system:
            return assign_all("A")
        return obj(present=False, quote="")

    facts = asyncio.run(read_facts(answer=ANSWER, llm=call))
    assert facts.failed == (STRUGGLE_FACT,)
    assert facts.complete
    # 讀失敗不能被當成「他沒講」:派發端要看得出來沒讀到。
    assert struggle_dimension(facts) == ""


def test_old_labels_without_struggle_leave_it_unread():
    """舊標註沒有這一欄,不能補成 False——那等於宣稱「讀過,他沒講」。

    補了的後果:每個用 from_label 組出來的事實都會讓派發把回合給 HR。
    """
    facts = AnswerFacts.from_label(
        {"star_parts": ["S", "A", "R"], "quantified": True, "tradeoff": True}
    )
    assert STRUGGLE_FACT not in facts.read_by
    assert facts.complete
    assert struggle_dimension(facts) == ""


def test_labeled_struggle_drives_the_dispatch_hint():
    base = {"star_parts": ["S", "T", "A", "R"], "quantified": True, "tradeoff": True}
    said = AnswerFacts.from_label({**base, STRUGGLE_FACT: True})
    silent = AnswerFacts.from_label({**base, STRUGGLE_FACT: False})
    assert said.struggle and struggle_dimension(said) == ""
    assert not silent.struggle and struggle_dimension(silent) == DIM_STRUGGLE


def test_struggle_never_changes_the_accept_decision():
    """第七項不進 accepts():同一份標註,有沒有它,放行與缺口都一樣。"""
    base = {"star_parts": ["S", "A", "R"], "quantified": False, "tradeoff": False}
    for rule in ACCEPT_RULES.values():
        without = AnswerFacts.from_label(base)
        for flag in (True, False):
            with_it = AnswerFacts.from_label({**base, STRUGGLE_FACT: flag})
            assert with_it.accepted_by(rule) == without.accepted_by(rule)
            assert with_it.gap_for(rule) == without.gap_for(rule)
