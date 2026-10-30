"""問句形式的機械檢查。兩件事:這一句是不是只能用是/否答完,
以及群面的應徵者那一句裡有沒有問句(見檔尾那一節)。

【為什麼要程式判,規則已經寫在 prompt 裡了】
`probe_rules` 的禁則第 2 條明文寫「凡是可以用是、否、或單一名詞回答完畢的
問題一律不可用」,第 4 條禁一次問兩件事。兩條都在生成端的 system prompt 裡。

四次 panel 量測共 144 題,實際數下來 36 題(25%)整句只能用是/否答完,
各次分別 31% / 22% / 25% / 22%。規則寫了沒生效。

這跟派發平衡、重複偵測是同一類:資訊給了,模型不可靠地執行。
「規則寫兩次沒生效,就不要寫第三次」——所以這裡用程式擋。

【為什麼這件事值得擋】
面試官問「你有沒有遇到什麼挑戰?」,使用者答「有啊,就還好」,
然後評分那邊的內容深度扣他分。題目是封閉的,分數算在答題的人頭上。
`scorability.MIN_SCORABLE_CHARS` 那一關也一樣——短回答會被判成不可評,
而他之所以答得短是因為我們問得封閉。

【判準:整句都封閉才算,一個開放子句就放行】
要防的傷害是「題目讓他答得短」。只要有一個子句在要求敘述,他就已經被要求
展開了,那一題不算有害。所以

    「有沒有遇到什麼挑戰?」                  擋
    「有沒有遇到什麼挑戰?你是怎麼克服的?」    不擋

後者仍然違反禁則第 4 條(一次問兩件事),但那是另一個問題,
而且傷害方向相反——它要的東西太多不是太少。
"""

from __future__ import annotations

import re

OPEN = "open"
CLOSED = "closed"
NEUTRAL = "neutral"

#: 子句切點。逗號也算。
#:
#: 【為什麼連逗號都切,以及它實際上管到什麼】
#: 第一版寫這條的理由是「你有沒有想過,怎麼從數據裡找問題呢?」這類句子真正
#: 的問題在逗號後面。那個理由後來不成立了:下面的優先序讓強疑問詞蓋過同一
#: 子句裡的 yes/no,所以那三句不切逗號也會判成開放。拿 137 句實測過,
#: 加不加逗號切點的結果一模一樣。
#:
#: 留著是因為它管的是另一種形狀——逗號後面是**弱**疑問詞的時候:
#:
#:     你有沒有做過類似的事,哪一段最花時間?
#:
#: 不切的話「有沒有」會蓋掉「哪一段」(弱疑問詞擋不住 yes/no),整句被判封閉,
#: 而它其實在問哪一段。那個形狀沒出現在手上的 137 句裡,所以這一條是照
#: 判準的邏輯補的,不是量出來的。對應的測例在 `test_question_form.py`
#: 裡標了是我編的。
_CLAUSE = re.compile(r"[?？。;；,，、]")

#: 禮貌的祈使句。字面是 yes/no,實際是叫他講一段。
#: 「能不能分享一下你怎麼做的」不是封閉問句。
_POLITE = ("能不能", "可不可以", "可以", "能", "方便")
_TELL = ("講", "說", "分享", "描述", "聊", "談", "舉例", "介紹")

#: 強疑問詞。要的是一段敘述,所以蓋過同一子句裡的 yes/no——
#: 「有沒有想過怎麼解決」真正在問的是怎麼解決。
_STRONG = ("怎麼", "怎樣", "如何", "為什麼", "為何", "多少")

#: 主述語是 yes/no。整個子句能用「有/沒有」答完,後面接什麼都一樣。
_YESNO = ("有沒有", "是不是", "會不會", "有無")

#: 弱疑問詞。單獨出現算開放,但擋不住 yes/no 也擋不住句尾的「嗎」——
#: 「有沒有遇到什麼挑戰」與「有什麼具體的學習計畫嗎」都能用「沒有」答完,
#: 那兩句裡的「什麼」是受詞不是疑問詞。
_WEAK = ("什麼", "哪", "幾")


def clause_kind(clause: str) -> str:
    """一個子句是開放、封閉,還是兩者都不是。

    順序就是優先序,每一條都是實測調出來的,改順序會改結果:
    禮貌祈使 → 強疑問詞 → yes/no → 句尾「嗎」 → 弱疑問詞。
    """
    if any(p in clause for p in _POLITE) and any(t in clause for t in _TELL):
        return OPEN
    if any(s in clause for s in _STRONG):
        return OPEN
    if any(y in clause for y in _YESNO):
        return CLOSED
    if clause.rstrip().endswith("嗎"):
        return CLOSED
    if any(w in clause for w in _WEAK):
        return OPEN
    return NEUTRAL


def is_closed_question(question: str) -> bool:
    """整句都只能用是/否答完就回 True。

    NEUTRAL 的子句(「那你在這個過程中」這種鋪陳)不影響判斷:
    它既不要求敘述也不是 yes/no。全部都是 NEUTRAL 時回 False——
    那表示這根本不是個問句,是另一種毛病,不該從這一關出去。
    """
    kinds = [clause_kind(c) for c in _CLAUSE.split(question) if c.strip()]
    return CLOSED in kinds and OPEN not in kinds


# ---------------------------------------------------------------------------
# 群面的應徵者不發問
# ---------------------------------------------------------------------------
#
# 10/30 定案:群面裡只有主考官和主管問問題,AI 應徵者只表態——反駁、補充、
# 整合都算。prompt 裡寫了,但這個專案量過不只一次:禁則寫在 prompt 裡會漏
# (封閉式問句那次,禁則已經在,144 題裡還有 36 題)。所以這裡給程式用的判準。

#: 句子的切點。只切句末標點,逗號不切——逗號後面的問句另外看,見 `_TAG`。
_SENTENCE = re.compile(r"(?<=[。！？!?\n])")

#: 接在逗號後面、單獨一小段的丟球句。只有這種形狀會從句子裡切掉,前面的
#: 主張留下來:「我覺得先做客群分析比較穩,你同意嗎?」→「我覺得先做客群分析比較穩。」
#:
#: 範圍刻意窄。「如果預算只夠做一件事,你們會先做哪一個?」逗號前面是條件子句,
#: 切掉後半剩下的是半句話,那種整句都算問句。
_TAG = re.compile(
    r"^(?:你們?|大家)?(?:怎麼看|覺得呢|覺得如何|覺得怎麼樣|同意嗎|認同嗎|說呢|呢"
    r"|對吧|是吧|對不對|好不好|不是嗎|要不要跟)[？?]$"
)


def _is_question(sentence: str) -> bool:
    s = sentence.strip()
    return s.endswith(("？", "?")) or s.rstrip("。.!！").endswith("嗎")


def question_sentences(text: str) -> list[str]:
    """一段話裡的問句,照出現順序。問號收尾,或句尾是「嗎」的算。"""
    return [s.strip() for s in _SENTENCE.split(text or "") if s.strip() and _is_question(s)]


def drop_questions(text: str) -> tuple[str, list[str]]:
    """刪掉問句。回傳 (剩下的話, 刪掉的部分)。

    逗號後面的丟球句只切那一小段,前面的主張改成句號收尾留下來;
    其他問句整句刪。剩下的照原本的順序接回去。全部都是問句時剩下的是空字串,
    那一輪怎麼辦由呼叫端決定。
    """
    kept: list[str] = []
    dropped: list[str] = []
    for raw in _SENTENCE.split(text or ""):
        s = raw.strip()
        if not s:
            continue
        if not _is_question(s):
            kept.append(s)
            continue
        cut = max(s.rfind("，"), s.rfind(","))
        head, tail = (s[:cut].strip(), s[cut + 1 :].strip()) if cut >= 0 else ("", s)
        if head and _TAG.match(tail):
            kept.append(head + "。")
            dropped.append(tail)
        else:
            dropped.append(s)
    return "".join(kept), dropped
