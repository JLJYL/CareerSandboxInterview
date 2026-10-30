"""逐題情緒測試。不打真 LLM。

重點在**四件事**:

    平穩不標註           回傳的是稀疏清單,空陣列是正常結果不是失敗
    引文沒落地就不採用   把人標成焦慮是對他的判斷,誤報成本比漏報高
    程式判的優先         明確說不知道有現成字面集合,那一項不問模型
    過度自責不回引文     查證要引文,呈現不要——引「我真的很爛」回去不是回饋

準確率不在這裡量,那要打真 API 配一組標註。這個檔驗的是形狀與濾除規則。
"""

from __future__ import annotations

import asyncio
import json

from app.pipeline.interview_live import _ADMITS_UNKNOWN
from app.pipeline.turn_emotion import (
    CODE,
    EMOTION_SOURCES,
    LLM,
    NO_QUOTE_BACK,
    STUCK,
    TurnEmotion,
    EmotionTurn,
    admits_unknown,
    batch_input_mode,
    parse_emotions,
    read_emotions,
)
from app.prompts.turn_emotion import (
    EMOTION_LABELS,
    EMOTION_MARK,
    EMOTION_SIGNAL,
    EMOTION_WRITEUP,
    PLAIN,
    compose_emotion_prompt,
    render_labels,
)

CALM = "我是資管系大三的學生,做過一個報表自動化的專案,把整理時間從三小時降到二十分鐘。"
ANXIOUS = "那個、那個我想一下,就是我當時,嗯,我不太確定要怎麼講比較好。"
HARSH = "我覺得我真的很爛,每次都做不好,這完全是我自己的問題。"
STUCK_TEXT = "這個我沒想過耶。"

TURNS = [
    EmotionTurn("請自我介紹", CALM),
    EmotionTurn("講一個失敗的經驗", ANXIOUS),
    EmotionTurn("你最大的缺點", HARSH),
    EmotionTurn("為什麼選這個職位", STUCK_TEXT),
]


def reply(**cells) -> str:
    return json.dumps(cells, ensure_ascii=False)


def llm_of(out: str, *, seen: list | None = None):
    def call(system: str, user: str) -> str:
        if seen is not None:
            seen.append(system)
        return out

    return call


# ---------------------------------------------------------------------------
# 宣告一致性
# ---------------------------------------------------------------------------


def test_every_label_has_a_signal_and_a_writeup():
    """加一種狀態就要同時給它訊號與報告寫法,否則 prompt 印空白、報告不知道該不該標。"""
    assert set(EMOTION_LABELS) == set(EMOTION_SIGNAL) == set(EMOTION_WRITEUP)
    block = render_labels()
    for label in EMOTION_LABELS:
        assert label in block
        assert EMOTION_SIGNAL[label] in block


def test_the_labels_the_code_uses_are_declared():
    """程式直接寫死的那幾個標籤要在值域裡。拼錯的話前端會收到查不到的值。"""
    assert {PLAIN, STUCK, NO_QUOTE_BACK} <= set(EMOTION_LABELS)
    assert PLAIN == EMOTION_LABELS[0]


def test_plain_is_the_one_that_is_not_written_up():
    assert EMOTION_WRITEUP[PLAIN] == "不標註"
    assert all(EMOTION_WRITEUP[v] != "不標註" for v in EMOTION_LABELS if v != PLAIN)


def test_sources_are_declared():
    assert set(EMOTION_SOURCES) == {CODE, LLM}


def test_the_prompt_mark_is_in_the_prompt():
    """假 LLM 照這個字串派發。它不在 prompt 裡的話,那一塊會悄悄收不到回應。

    第一版的開頭含「逐題」,跟逐題點評撞號,情緒的呼叫拿到了點評的陣列。
    """
    for mode in ("voice", "typed", "unknown"):
        assert EMOTION_MARK in compose_emotion_prompt(mode)


def test_a_mixed_batch_is_marked_mixed_not_unknown():
    """同一場裡語音辨識失敗而改打字是會發生的,而那一題的碎句程度跟其他題不可比。

    拿整場的模式去套,等於用「這場大多是語音」去解釋一題打字的回答。
    原本退回 unknown,但那一段叫模型不要看停頓,而語音題的停頓照樣送進去。
    """
    assert batch_input_mode(["voice", "voice"]) == "voice"
    assert batch_input_mode(["typed"]) == "typed"
    assert batch_input_mode(["voice", "typed"]) == "mixed"
    assert batch_input_mode(["voice", "unknown"]) == "unknown", "分不出來的仍然是 unknown"
    assert batch_input_mode([]) == "unknown"
    assert batch_input_mode(["", "voice"]) == "voice"


def test_input_mode_changes_which_signals_are_allowed():
    """打字沒有停頓資料,語音的碎句可能是引擎造成的。prompt 要分開講。

    這跟表達流暢度踩過的坑是同一個:打字的答案標點乾淨就拿到高流暢度。
    """
    typed = compose_emotion_prompt("typed")
    voice = compose_emotion_prompt("voice")
    assert "沒有停頓資料" in typed
    assert "引擎的性質" in voice
    assert typed != voice
    assert compose_emotion_prompt("亂填") == compose_emotion_prompt("unknown")


# ---------------------------------------------------------------------------
# 平穩不標註
# ---------------------------------------------------------------------------


def test_plain_turns_do_not_appear():
    got = parse_emotions(reply(**{"1": {"label": PLAIN, "quote": ""}}), TURNS)
    assert got == {}


def test_an_all_plain_interview_returns_an_empty_list():
    """空陣列是正常結果,不是判讀失敗。"""
    out = asyncio.run(
        read_emotions(
            turns=TURNS[:1], llm=llm_of(reply(**{"1": {"label": PLAIN, "quote": ""}}))
        )
    )
    assert out == []


def test_no_turns_costs_no_call():
    seen: list = []
    assert asyncio.run(read_emotions(turns=[], llm=llm_of("{}", seen=seen))) == []
    assert seen == []


# ---------------------------------------------------------------------------
# 引文要落地
# ---------------------------------------------------------------------------


def test_a_label_without_a_grounded_quote_is_dropped():
    """說他焦慮但抄不出原文 → 不標註。

    這是這一塊唯一的查證手段,而且誤報的代價比漏報高:
    把人標成焦慮是對他的判斷。
    """
    got = parse_emotions(
        reply(**{"2": {"label": "焦慮", "quote": "我整個人都慌了"}}), TURNS
    )
    assert got == {}


def test_a_grounded_quote_is_kept():
    got = parse_emotions(
        reply(**{"2": {"label": "焦慮", "quote": "那個、那個我想一下"}}), TURNS
    )
    assert got[2].label == "焦慮"
    assert got[2].source == LLM


def test_a_quote_from_another_turn_does_not_count():
    """引文要在**那一題**的回答裡。整場逐字稿裡找得到不算。"""
    got = parse_emotions(
        reply(**{"1": {"label": "焦慮", "quote": "那個、那個我想一下"}}), TURNS
    )
    assert got == {}


def test_unknown_labels_and_out_of_range_turns_are_dropped():
    got = parse_emotions(
        reply(
            **{
                "1": {"label": "很緊張", "quote": CALM[:10]},
                "9": {"label": "焦慮", "quote": ANXIOUS[:8]},
                "x": {"label": "焦慮", "quote": ANXIOUS[:8]},
            }
        ),
        TURNS,
    )
    assert got == {}


def test_unparsable_output_marks_nothing():
    assert parse_emotions("抱歉,我無法判斷。", TURNS) == {}
    assert parse_emotions("", TURNS) == {}


# ---------------------------------------------------------------------------
# 停頓那一行
# ---------------------------------------------------------------------------


PAUSE_NOTE = "這一題的段落之間共停頓 2 次,最長一次 4.5 秒,合計 5.7 秒(實測值)"


def test_the_pause_note_reaches_the_model():
    seen: list = []

    def spy(system: str, user: str) -> str:
        seen.append(user)
        return reply()

    asyncio.run(
        read_emotions(
            turns=[EmotionTurn("講一個失敗的經驗", ANXIOUS, PAUSE_NOTE)],
            llm=spy,
        )
    )
    assert PAUSE_NOTE in seen[0]


def test_no_pause_note_means_the_line_is_absent():
    """沒有時間資料就整行不出現。

    印一行「停頓:無資料」會讓模型以為那是「沒有停頓」,而兩者差別很大:
    一個是我們不知道,一個是他講得很順。
    """
    seen: list = []

    def spy(system: str, user: str) -> str:
        seen.append(user)
        return reply()

    asyncio.run(read_emotions(turns=TURNS, llm=spy))
    assert "停頓" not in seen[0]


def test_the_pause_note_cannot_be_used_as_evidence():
    """停頓那一行是我們寫的,不是他講的,所以不能被引來當證據。

    這是把它跟 answer 分開放的理由。併進 answer 的話 `grounded()` 會通過,
    於是模型可以引「最長一次 4.5 秒」來證明他焦慮——那句話不是他說的。
    """
    turns = [EmotionTurn("講一個失敗的經驗", ANXIOUS, PAUSE_NOTE)]
    got = parse_emotions(
        reply(**{"1": {"label": "焦慮", "quote": "最長一次 4.5 秒"}}), turns
    )
    assert got == {}, "引文查的是 answer,不是我們附上的那一行"

    ok = parse_emotions(
        reply(**{"1": {"label": "焦慮", "quote": "那個、那個我想一下"}}), turns
    )
    assert ok[1].label == "焦慮"


# ---------------------------------------------------------------------------
# 程式判的優先
# ---------------------------------------------------------------------------


def test_admits_unknown_uses_the_shared_literals():
    """與 forced_decision 共用同一組字面。各自列一份就會出現自相矛盾的輸出。"""
    for word in _ADMITS_UNKNOWN:
        assert admits_unknown(f"這個我{word}")
    assert not admits_unknown(CALM)
    assert not admits_unknown("")


def test_code_beats_the_model():
    """模型說平穩,而字面命中「沒想過」→ 判成卡住,來源記 code。

    「能用程式算的不要問 LLM」在這裡的形式:有唯一正確答案的那一項不交給它。
    """
    out = asyncio.run(
        read_emotions(
            turns=TURNS,
            llm=llm_of(reply(**{str(i): {"label": PLAIN, "quote": ""} for i in range(1, 5)})),
        )
    )
    assert [(e.turn_index, e.label, e.source) for e in out] == [(4, STUCK, CODE)]


def test_code_overrides_a_conflicting_model_label():
    out = asyncio.run(
        read_emotions(
            turns=TURNS,
            llm=llm_of(reply(**{"4": {"label": "焦慮", "quote": STUCK_TEXT[:5]}})),
        )
    )
    assert out[0].label == STUCK
    assert out[0].source == CODE


def test_a_failed_call_still_returns_what_the_code_found():
    """呼叫掛掉不代表整塊沒有。字面判出來的那些不依賴模型。"""

    def boom(system: str, user: str) -> str:
        raise RuntimeError("timeout")

    out = asyncio.run(read_emotions(turns=TURNS, llm=boom))
    assert [(e.turn_index, e.label, e.source) for e in out] == [(4, STUCK, CODE)]


# ---------------------------------------------------------------------------
# 過度自責不回引文
# ---------------------------------------------------------------------------


def test_self_blame_is_verified_by_a_quote_but_does_not_return_one():
    """查證與呈現是兩件事。

    一定要有落地的引文才採用(不然無法確認模型不是在編),
    但把那句話引回去給使用者看不是回饋是傷害。
    """
    got = parse_emotions(
        reply(**{"3": {"label": NO_QUOTE_BACK, "quote": "我覺得我真的很爛"}}), TURNS
    )
    assert got[3].quote == "我覺得我真的很爛"
    assert got[3].shown_quote == ""


def test_self_blame_without_a_quote_is_still_dropped():
    got = parse_emotions(reply(**{"3": {"label": NO_QUOTE_BACK, "quote": ""}}), TURNS)
    assert got == {}


def test_the_other_labels_do_return_their_quote():
    for label, turn in (("焦慮", 2), ("卡住", 2)):
        e = TurnEmotion(turn, label, "那個、那個我想一下", LLM)
        assert e.shown_quote == e.quote


# ---------------------------------------------------------------------------
# 排序
# ---------------------------------------------------------------------------


def test_output_is_sorted_by_turn():
    """前端照題號掛在逐題回顧旁邊,順序亂掉會對錯題。"""
    out = asyncio.run(
        read_emotions(
            turns=TURNS,
            llm=llm_of(
                reply(
                    **{
                        "3": {"label": NO_QUOTE_BACK, "quote": "我覺得我真的很爛"},
                        "2": {"label": "焦慮", "quote": "那個、那個我想一下"},
                    }
                )
            ),
        )
    )
    assert [e.turn_index for e in out] == [2, 3, 4]


# ---------------------------------------------------------------------------
# 字面只在短回答裡算數(前端 10/29「自我貶低」那份錄音)
# ---------------------------------------------------------------------------

SELF_DOUBT = "我講得不太好,可能沒有什麼經驗 但是我之前有參加過一個比賽 然後是跟股票相關的 但我也不確定做得對不對,應該沒有很厲害"


def test_an_admission_phrase_inside_a_long_answer_is_not_a_code_verdict():
    """第 1 題命中「不確定」,原本被程式直接標成卡住,而那一題要測的是過度自責。"""
    assert not admits_unknown(SELF_DOUBT)
    assert admits_unknown(STUCK_TEXT), "短的明確承認照算"


def test_the_model_keeps_its_label_on_a_long_self_doubting_answer():
    turns = [EmotionTurn("請自我介紹", SELF_DOUBT)]
    out = asyncio.run(
        read_emotions(
            turns=turns,
            llm=llm_of(reply(**{"1": {"label": NO_QUOTE_BACK, "quote": "應該沒有很厲害"}})),
        )
    )
    assert [(e.label, e.source) for e in out] == [(NO_QUOTE_BACK, LLM)]


# ---------------------------------------------------------------------------
# 混用逐題標;雲端辨識的語音另外一段(前端 10/21 起可以改打字,9 月起用 Whisper)
# ---------------------------------------------------------------------------


def test_a_mixed_batch_labels_each_turn_and_keeps_the_voice_pauses():
    seen: list = []

    def call(system: str, user: str) -> str:
        seen.append((system, user))
        return reply(**{"1": {"label": PLAIN, "quote": ""}, "2": {"label": PLAIN, "quote": ""}})

    turns = [
        EmotionTurn("請自我介紹", CALM, "這一題的段落之間共停頓 1 次,最長一次 6.6 秒", "voice"),
        EmotionTurn("講一個失敗的經驗", ANXIOUS, "", "typed"),
    ]
    asyncio.run(read_emotions(turns=turns, llm=call, input_mode=batch_input_mode(["voice", "typed"])))
    system, user = seen[0]
    assert "有的題是講的、有的題是打的" in system
    assert "斷句與停頓都不可靠" not in system, "不要再叫模型忽略停頓"
    assert "作答方式:講的" in user and "作答方式:打的" in user
    assert "6.6 秒" in user


def test_turns_are_not_labeled_when_the_whole_batch_is_one_mode():
    from app.prompts.turn_emotion import emotion_user_message

    msg = emotion_user_message([("q", CALM, "", "voice")])
    assert "作答方式" not in msg


def test_cloud_transcription_gets_its_own_voice_guidance():
    """裝置端那一段叫模型提防「碎」是引擎造成的;Whisper 是把猶豫清乾淨。"""
    device = compose_emotion_prompt("voice", "device")
    cloud = compose_emotion_prompt("voice", "api")
    assert "比他實際講的順" in cloud
    assert "引擎的性質" in device and "引擎的性質" not in cloud
    assert compose_emotion_prompt("typed", "api") == compose_emotion_prompt("typed", "device")
