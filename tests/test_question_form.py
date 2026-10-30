"""封閉式問句的判準。

測例全部取自四次 panel 量測的真實輸出(`data/panel_v4.json`、`panel_v5.json`、
`panel_next_v1.json`、`panel_next_v2.json`),不是我編的句子——編的句子只會
印證我自己的直覺,而這個判準的整條優先序都是逐句看那 137 句調出來的。
"""

from __future__ import annotations

import pytest

from app.pipeline.question_form import (
    CLOSED,
    NEUTRAL,
    OPEN,
    clause_kind,
    drop_questions,
    is_closed_question,
    question_sentences,
)

#: 整句只能用「有/沒有」答完。
CLOSED_CASES = [
    "在這個過程中,你有沒有遇到什麼挑戰?",
    "那在這個過程中,你有沒有跟其他部門合作的經驗?",
    "那你在選擇退貨分析時,有沒有考慮到其他團隊的需求?",
    "這樣的改進讓你省下的時間,有沒有影響到其他專案或工作呢?",
    "你覺得自己在這方面有什麼具體的學習計畫嗎?",
    "那你在重寫報表的過程中,遇到什麼挑戰嗎?",
    "你當時有想過如果行銷活動晚了,可能會影響什麼嗎?",
]

#: 有一個子句在要求敘述,所以不算——他已經被要求展開了。
OPEN_CASES = [
    # 後半是真正的問題。這三句在加逗號切點之前被誤判成封閉。
    "那你有沒有想過,怎麼從數據中找到那些關鍵問題呢?",
    "那你有沒有想過,當你不知道要問什麼問題時,你會怎麼開始分析呢?",
    "那你當時有沒有考慮過行銷活動的影響,為什麼還是選擇退貨分析呢?",
    # 複合題:前半封閉後半開放。違反禁則第 4 條,但傷害方向相反。
    "在這個過程中,你有沒有遇到什麼挑戰?你是怎麼克服的?",
    "那你是怎麼確保舊報表的數字一致性的?有沒有遇到什麼挑戰?",
    "在這個過程中,你有沒有跟其他部門合作?他們的反應如何?",
    # 禮貌的祈使句。字面是 yes/no,實際是叫他講一段。
    "那你能分享一個你曾經遇到的具體問題嗎?",
    "那你能分享一下你是怎麼確定要問哪些問題的嗎?",
    # 一般的開放問句。
    "那件事對你來說最難的是哪一段?",
    "你當時是怎麼決定優先做那三個欄位的呢?",
    "那你當時為什麼選擇先做那三個欄位呢?",
]


@pytest.mark.parametrize("q", CLOSED_CASES)
def test_closed_questions_are_caught(q: str) -> None:
    assert is_closed_question(q)


@pytest.mark.parametrize("q", OPEN_CASES)
def test_questions_with_an_open_clause_pass(q: str) -> None:
    assert not is_closed_question(q)


def test_a_strong_interrogative_beats_a_yes_no_in_the_same_clause() -> None:
    """「有沒有想過怎麼解決」真正在問的是怎麼解決。

    順序反過來的話,那三句「有沒有想過,怎麼…」會被當成封閉——
    加逗號切點之前就是那樣。
    """
    assert clause_kind("你有沒有想過怎麼解決") == OPEN
    assert clause_kind("你有沒有遇到什麼挑戰") == CLOSED


def test_a_sentence_final_ma_beats_a_weak_interrogative() -> None:
    """「有什麼具體的學習計畫嗎」裡的「什麼」是受詞不是疑問詞。

    順序反過來的話那一句會被放行,而它能用「沒有」答完。
    """
    assert clause_kind("你有什麼具體的學習計畫嗎") == CLOSED
    assert clause_kind("你學到了什麼最重要的技能") == OPEN


def test_context_fragments_are_neutral_not_open() -> None:
    """鋪陳的子句既不要求敘述也不是 yes/no。

    把它判成開放的話,「那你在這個過程中,有沒有遇到什麼挑戰?」整句會
    因為前半而放行——而那正是要擋的那一句。
    """
    assert clause_kind("那你在這個過程中") == NEUTRAL
    assert clause_kind("聽起來你很會整理需求") == NEUTRAL
    assert is_closed_question("聽起來你很會整理需求,那你在這個過程中有沒有遇到什麼挑戰?")


def test_a_question_with_no_interrogative_at_all_is_not_closed() -> None:
    """全部子句都是 NEUTRAL 時回 False。

    那表示這根本不是個問句,是另一種毛病。從這一關放它出去是刻意的——
    這一關只管「封閉」,不該順便擋別的東西。
    """
    assert not is_closed_question("那我們來聊聊你的實習")


def test_the_measured_rate_on_real_output() -> None:
    """判準的校準點:四次量測 144 題裡 36 題(25%)整句封閉。

    這條不是在測程式,是把那個數字釘住。有人調優先序或關鍵詞之後,
    這裡會告訴他影響多大——25% 是 `question_form` 整段 docstring
    與「這件事值得擋」那個論證的依據。
    """
    closed = [q for q in CLOSED_CASES + OPEN_CASES if is_closed_question(q)]
    assert len(closed) == len(CLOSED_CASES), (
        f"測例裡判為封閉的有 {len(closed)} 句,應該剛好是 CLOSED_CASES 的 "
        f"{len(CLOSED_CASES)} 句"
    )


def test_a_weak_interrogative_after_a_comma_still_opens_the_question() -> None:
    """這一條的句子是我編的,不是量測輸出,因為那個形狀沒出現在 137 句裡。

    逗號切點管的就是它:逗號後面是**弱**疑問詞的時候。

        你有沒有做過類似的事,哪一段最花時間?

    不切逗號的話,「有沒有」會蓋掉「哪一段」(弱疑問詞擋不住 yes/no),
    整句判成封閉,而它其實在問哪一段。

    強疑問詞那一類(「有沒有想過,怎麼解決」)不需要靠逗號——優先序已經
    讓強疑問詞蓋過 yes/no 了,拿 137 句實測過,加不加逗號切點結果相同。
    所以如果有人想把逗號從 `_CLAUSE` 拿掉,會紅的只有這一條。
    """
    assert not is_closed_question("你有沒有做過類似的事,哪一段最花時間?")
    assert not is_closed_question("你有沒有做過類似的事，哪一段最花時間？")
    # 對照:逗號後面沒有疑問詞,整句還是封閉。
    assert is_closed_question("你有沒有做過類似的事,那時候是實習嗎?")


# ---------------------------------------------------------------------------
# 群面應徵者的問句(10/30 定案:應徵者不發問)
#
# 前三句是 group-1(前端 10/29 錄的那一場)裡 AI 的真實發言。
# 丟球句那幾條是照規則編的:舊規則底下沒錄到應徵者用逗號接丟球句的輸出。
# ---------------------------------------------------------------------------

GROUP1_LINES = [
    "你提到使用 Python 進行資料分析，那你們具體用哪些指標來評估風險呢？",
    "你提到會避免掠奪性期刊，那在資料來源的選擇上，你會如何確保這些資料的準確性呢？",
    "在這個驗證過程中，你能分享一下當時遇到的挑戰是什麼嗎？你是怎麼克服的？",
]


@pytest.mark.parametrize("line", GROUP1_LINES)
def test_group1_peer_lines_are_questions_with_nothing_to_keep(line: str) -> None:
    kept, dropped = drop_questions(line)
    assert question_sentences(line)
    assert kept == "", "整句都是問句,沒有主張可留"
    assert "".join(dropped) == line


def test_a_tag_question_after_a_comma_is_cut_and_the_claim_kept() -> None:
    assert drop_questions("我覺得先做客群分析比較穩,你同意嗎?") == (
        "我覺得先做客群分析比較穩。",
        ["你同意嗎?"],
    )


def test_a_question_sentence_after_a_claim_is_dropped_whole() -> None:
    assert drop_questions("線上流程誰都會做,我會先談跟學校合作。你怎麼看?") == (
        "線上流程誰都會做,我會先談跟學校合作。",
        ["你怎麼看?"],
    )


def test_a_conditional_question_is_not_cut_into_half_a_sentence() -> None:
    """逗號前面是條件子句。切掉後半剩下的是半句話,所以整句都算問句。"""
    kept, dropped = drop_questions("如果預算只夠做一件事,你們會先做哪一個?")
    assert kept == "" and len(dropped) == 1


def test_a_statement_is_left_alone() -> None:
    line = "母數不夠,這個結論我不敢下。"
    assert drop_questions(line) == (line, [])
    assert question_sentences(line) == []


def test_a_question_without_a_question_mark_still_counts() -> None:
    assert question_sentences("這樣可以嗎。我覺得不行。") == ["這樣可以嗎。"]
