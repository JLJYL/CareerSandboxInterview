"""這一輪要做什麼:NEXT / FOLLOW_UP / CLARIFY 的判準文字。

【這個檔跟 probe_rules 的分工】
`probe_rules` 與 `DIFFICULTY_TRIGGERS` 回答「追問要問什麼」,
這裡回答「要不要追問」。兩者問的是不同的問題,同時存在不會打架。

之前三種難度產出的問句幾乎逐字相同,就是因為只有前者——
模型照著「依序檢查」跑,每一輪都找得到可以追的東西。

【為什麼放行條件是從 AcceptRule 算出來的,不是寫死在這裡】
條件同時決定兩件事:prompt 怎麼寫,以及之後怎麼驗。
兩邊各寫一份一定會漂,所以文字從 dataclass 生出來,只有一個來源。

【CLARIFY 是我們加的】
Panfilova et al.(2026)的 action space 只有 FOLLOW_UP 與 NEXT_MAIN。
他們的受訪者是打字,沒有語音辨識;我們的逐字稿會出現
Angular 轉成 Andrew 這類錯誤,所以需要第三個動作。
論文與簡報要標成延伸,不能寫成論文原有的。
"""

from __future__ import annotations

from app.pipeline.interviewer_spec import AcceptRule, InterviewerSpec

DECISION_ACTIONS: tuple[str, ...] = ("NEXT", "FOLLOW_UP", "CLARIFY")


DECISION_TASK = """你的工作只有一件:決定面試官這一輪要做什麼。

不要寫問題。只回一個動作加簡短理由。

【三個動作】

NEXT       回答已經講完了,換下一個主問題。
FOLLOW_UP  回答有缺口,而且那個缺口值得追。
CLARIFY    你看不懂他在說什麼,而且像是語音辨識轉錯,不是他講得含糊。"""


CLARIFY_BY_ENGINE: dict[str, str] = {
    "api": """【這份逐字稿的來源:雲端轉錄】

不漏字、有標點。所以**破碎的句子通常是他真的在猶豫**,不是辨識壞掉。
CLARIFY 只用在單一詞彙明顯不合情境的時候——
句子很通順但某個專有名詞放在那裡說不通,那才是轉錯。

他講得零碎但看得懂意思,那是 FOLLOW_UP 不是 CLARIFY。""",
    "device": """【這份逐字稿的來源:裝置端即時辨識】

會漏字、沒有標點。看不懂的時候**先假設是辨識掉字**,不是他沒講。

但整段都看不懂時不要用 CLARIFY——那會變成每一輪都在說沒聽清楚。
挑一個關鍵的地方問就好。""",
}


DECISION_OUTPUT = """【輸出格式】

只回這個 JSON,不要加任何說明文字:

{"why": "兩三句話說明你的判斷", "action": "NEXT", "gap": ""}

    why     先寫理由再下判斷。對照上面的放行條件逐項看。
    action  NEXT / FOLLOW_UP / CLARIFY 三者之一,一字不差。
    gap     FOLLOW_UP 時填缺的是哪一項(例如「結果段」「量化」「取捨」);
            NEXT 與 CLARIFY 填空字串。"""


def render_accept_rule(rule: AcceptRule) -> str:
    """把放行條件寫成 prompt 文字。唯一的來源是 dataclass,不另外手寫一份。"""
    lines = [
        "【什麼時候算講完了】",
        "",
        f"1. STAR 四段(情境 / 任務 / 行動 / 結果)至少要交代 {rule.min_star_parts} 段。",
    ]
    n = 2
    if rule.require_result:
        lines.append(f"{n}. 一定要有結果段——那是成果的唯一證據。")
        n += 1
    if rule.require_quantification:
        lines.append(
            f"{n}. 成果要有可查證的數字。只有「提升了」「變好了」這種說法不算。"
        )
        n += 1
    if rule.require_tradeoff:
        lines.append(f"{n}. 要交代取捨:放棄了什麼、判斷依據是什麼。")
        n += 1
    lines += [
        "",
        "以上**全部**滿足 → NEXT。",
        "任何一項不滿足 → FOLLOW_UP,並在 gap 指出缺的是哪一項。",
        "",
        "沒有列在上面的東西不要拿來當追問的理由。"
        "這個難度不要求的,他沒講也算講完了。",
    ]
    return "\n".join(lines)


def compose_decision_prompt(
    spec: InterviewerSpec, *, transcription_engine: str = "api"
) -> str:
    """決策專用的 system prompt。

    刻意不帶 persona。決定「要不要追問」跟誰在問無關——
    panel 三位看的是同一份職缺需求與同一套放行條件,
    決定哪一輪輪到誰的是派發那一步。

    帶了 persona 只會把 prompt 撐長,而 prompt 一長人格就塌——
    派發那支拆成兩步就是為了這件事。
    """
    parts = [DECISION_TASK, render_accept_rule(spec.accept)]
    if spec.has_jd():
        listed = "、".join(spec.cares_about)
        parts.append(f"【這個職缺在意什麼】\n\n{listed}\n\n缺口落在這些項目上才值得追。")
    else:
        parts.append(
            "【這個職缺在意什麼】\n\n(沒有職缺需求可用)\n\n"
            "只依 STAR 判斷缺口,不要自行想像這個職位需要什麼。"
        )
    parts.append(CLARIFY_BY_ENGINE.get(transcription_engine, CLARIFY_BY_ENGINE["api"]))
    parts.append(DECISION_OUTPUT)
    return "\n\n".join(parts)
