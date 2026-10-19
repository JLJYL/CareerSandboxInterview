"""面試報告(A4)管線。

【流程】
    第一層  確定性統計(本地,毫秒級)
            TextStats、逐題切分、輸入方式判定
                ↓
    第二層  四個獨立 LLM 呼叫,並行
            faceDimensions / subScores / questionFeedbacks / starParts
                ↓
    第三層  機械修復與驗證
            固定值域歸位、verbatim 檢查、用語正規化
                ↓
    第四層  improvements 由 faceDimensions 彙整(不打 LLM)

【模型路由】
starParts 需要跨行原文引用,gpt-4o-mini 做不到,要用 gpt-4o。
其餘三個用 mini 即可。所以這支收兩個 llm callable。

【端點接線時的必要參數】
TranscriptAnalyzer 要帶 engine_filler_policy="partial"。

那對應 STT 詞彙探測的實測結論:Android 移除非詞彙填充音(嗯、呃),
保留詞彙型(那個、就是、然後)。不帶的話 filler_reliability 一律退回
"unknown",流暢度永遠走備援訊號——那是整個模組唯一乾淨的量化訊號
(實測每百字:正常 3.0 / 多講故事 6.4 / 刻意講亂 9.8),白白丟掉很可惜。

這個參數的預設值是 "unknown" 而不是 "partial",因為那是 A 側對「未探測過的
引擎」的保守預設。探測做過了,呼叫端有責任把結論帶進去。

【A5 學到的,這裡沿用】
    LLM 不做過濾也不做計數,能錨的錨在確定性統計上
    固定值域靠 repair_report,不靠 prompt 自律
    verbatim 靠 verify_star_verbatim 機械檢查,對不上就丟整條
    任何一個區塊失敗不影響其他區塊(gather_blocks 的 return_exceptions)
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Callable, Sequence

from app.contracts.interview_protocols import TextStats
from app.pipeline.jd_normalize import normalize_locale
from app.pipeline.text_fold import fold_variants
from app.pipeline.parallel import gather_blocks
from app.pipeline.scorability import LOWEST_ANCHOR, assess
from app.pipeline.speech_timing import describe_pauses, measure as measure_timing
from app.pipeline.turn_emotion import EmotionTurn, batch_input_mode, read_emotions
from app.prompts.interview_report import (
    MAX_IMPROVEMENTS,
    compose_face_prompt,
    compose_question_prompt,
    compose_star_prompt,
    compose_sub_score_prompt,
)
from app.schemas.interview import (
    FACE_DIMENSIONS,
    STAR_PARTS,
    SUB_SCORE_NAMES,
    FaceDimensionDTO,
    ProsodyItem,
    QuestionFeedbackDTO,
    ReportResponse,
    StarPartDTO,
    SubScoreDTO,
    TurnDTO,
    TurnEmotionDTO,
)
from app.schemas.interview_repair import repair_report, verify_star_verbatim

LLMCall = Callable[[str, str], str]

VERBATIM_FAIL_MARKER = "fromAnswer 引用失敗"
"""verbatim 檢查失敗時寫進 notices 的標記。

抽成常數是為了讓診斷腳本可以可靠地數失敗次數,而不是比對整句措辭——
措辭改一次,靠字串比對的統計就會靜默歸零,而且沒有人會發現,
因為它看起來只是「這次沒有失敗」。
"""

_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.S)


def parse_array(raw: str) -> list:
    """把模型輸出解析成陣列。失敗回空清單,不拋例外。

    空清單會讓該區塊被 repair_report 補成預設值並記入 notices,
    使用者看到的是一份少一塊的報告,不是 500。
    """
    text = (raw or "").strip()
    m = _FENCE.search(text)
    if m:
        text = m.group(1).strip()
    if not text.startswith("["):
        i, j = text.find("["), text.rfind("]")
        if i != -1 and j > i:
            text = text[i : j + 1]
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


# ---------------------------------------------------------------------------
# 確定性統計 → 給 prompt 的錨點
# ---------------------------------------------------------------------------


def format_stats(stats: TextStats, input_mode: str) -> tuple[str, bool]:
    """把 TextStats 轉成給 prompt 的文字,並回傳流暢度可不可測。

    不可測的兩種情況:
        引擎做了 disfluency removal(filler_reliability != "measured")
        這是打字輸入(input_mode != "voice")—— 打字本來就不會打「嗯」

    第二種特別要擋:打字的答案沒有填充詞、標點乾淨,會拿到「高」流暢度,
    認真用講的反而低分。那是在獎勵錯誤的行為。
    """
    measurable = stats.filler_reliability == "measured" and input_mode == "voice"

    lines = [f"總字數 {stats.char_count}"]
    if measurable and stats.char_count:
        per_100 = stats.filler_count / stats.char_count * 100
        detail = "、".join(f"{k} {v} 次" for k, v in stats.filler_detail.items())
        lines.append(f"詞彙型填充詞每百字 {per_100:.1f}({detail or '無'})")
        lines.append("  參考區間:每百字 3 左右為正常,6 以上偏多,9 以上明顯影響聽感")
        lines.append("  注意:只涵蓋詞彙型(那個、就是、然後),非詞彙音(嗯、呃)已被引擎移除")
    else:
        reason = "打字輸入" if input_mode != "voice" else f"計數不可信({stats.filler_reliability})"
        lines.append(f"填充詞:不可用({reason})")

    lines.append(f"數字與量化詞 {stats.quantifier_count} 處")
    if stats.segmentation == "punctuation":
        lines.append(f"平均句長 {stats.avg_sentence_len:.0f} 字(有標點,精確值)")
    elif stats.segmentation == "stt_segment":
        lines.append(
            f"平均每段 {stats.avg_sentence_len:.0f} 字"
            "(邊界來自語音辨識的自動送出點,量的是語流連續性,不是句子長度)"
        )
    else:
        lines.append("句長:不可用(無法斷句)")

    return "\n".join(lines), measurable


MIN_FLUENCY_CHARS = 100
"""算填充詞密度需要的最小字數。短於這個字數回未評分。

【這是樣本量問題,不是內容判斷】
流暢度的分母是每百字。14 個字裡觀察到 0 個填充詞,對一個每百字 3.0 的
正常講者來說完全是預期結果——那個長度下的 0 不代表講得順,代表沒得量。

實測這個洞有多大:

    「喔好啊這樣啊對啊我覺得還好。」 14 字,0 個填充詞 → 密度 0.0 → **90 分**
    認真回答,密度 3.0                                    → 82 分
    刻意講亂,密度 9.8                                    → 60 分

亂答比認真回答高 8 分。注意失誤出在**短而乾淨**那一端:刻意講亂拿 60,
公式本身是對的,壞的是在沒有樣本的時候還硬算。

【為什麼是 100】
以每百字 3.0 為正常值,100 字的期望填充詞數是 3。到這個長度,觀察到 0
才開始有訊息量。低於它,0 跟講者習慣無關。

黃金集十份整場 254–690 字,全部遠高於這個門檻,所以這條規則不會動到
任何一份真實錄音——它只擋幾乎沒講話的場次。
"""

FLUENCY_BANDS: tuple[tuple[float, int, int], ...] = (
    (3.0, 82, 90),
    (6.0, 72, 82),
    (9.0, 60, 72),
    (99.0, 45, 60),
)
"""(每百字上限, 分數下界, 分數上界)。區間內線性內插。

依據成員 A 的實測錨點:正常發揮 3.0、多講故事 6.4、刻意講亂 9.8。
用公式而非 LLM 的理由:實測填充詞率 3.0(正常)時模型給 71,
分數沒有跟著錨走。單調映射不會有這個問題,而且同一份逐字稿跑兩次結果一樣。
"""


STAR_BANDS: dict[int, tuple[int, int]] = {
    4: (80, 95),
    3: (68, 82),
    2: (56, 70),
    1: (46, 60),
    0: (40, 52),
}
"""STAR 命中段數 → (分數下界, 分數上界)。量化詞在區間內定位。

【為什麼要第二個訊號】
原本只看量化詞處數。實測黃金集十份:

    量化詞  2  2  2  2  3  3  3  4  12  12
    內容深度 64 64 64 64 68 68 68 70 83  83

十份裡有八份落在 64–70,**相差 6 分**。全距看起來是 19,但那 19 全部
來自一份剛好講了 12 個數字的逐字稿。換句話說這個維度只在有人湊巧
講很多數字的時候才分得出東西,其餘時間所有人都是同一分。

對照組:邏輯清晰度(LLM 評的)四場實跑全距 25。把內容深度改成公式
讓它穩定了,但也讓它變成六項裡最分不出好壞的一項。

【為什麼是 STAR 當主軸,量化詞當副軸】
這個分法不是我挑的,是評分標準本來的形狀:

    90–100  有具體情境、行動、結果,**且**包含量化成果或清楚的決策理由
    70–89   有具體行動與結果,但缺少量化證據、取捨理由或細節
    50–69   有回答問題,但內容偏概括,只描述「有做」而沒有說明怎麼做
    0–49    明顯答非所問、內容過短

讀一遍就看得出來:前面那一段決定落在哪一帶的是**講了幾個 STAR 要素**,
量化成果只負責把 70–89 那一帶推上 90。所以 STAR 段數定帶、量化詞定位置。

【用驗證過的 STAR,不用模型原本回的】
`verify_star_verbatim` 會把引用不存在於逐字稿的段落降成未命中。
用未驗證的版本等於讓模型編一句引用就把內容深度推上去——
那是我們已經擋過一次的失敗(`ungrounded_numbers` 同一個理由)。

【沒有校準資料】
黃金集十份的 labels 只有 skills / alias_decisions / gap_ranking,
沒有 STAR 標註也沒有品質標註。所以這些分界是照上面那份評分標準取的,
不是量出來的——跟它取代的那一版一樣。

要校準需要一批標好高/中/低品質的回答。那也正是修完之後第五步
(用不同品質的回答測區分度)要做的事。

【已知的不對稱】
STAR 是行為題的結構。整場都是非行為題(純自我介紹、動機題)的話,
段數本來就低,內容深度會跟著被壓低。目前先接受:報告的 STAR 區塊
本來就在告訴使用者「你應該講出這四段」,兩邊的立場一致。
"""

QUANT_FULL = 6
"""量化詞到幾處算把該帶推到頂。

黃金集的分布是 2 / 3 / 4 / 12,取 6 讓多數回答落在帶內的 1/3 到 2/3,
而不是一進來就頂到天花板。沒有校準資料,理由同 STAR_BANDS。
"""


def score_depth(stats: TextStats, star_present: int = 0) -> tuple[int, str]:
    """內容深度。回傳 (分數, 依據說明)。

    `star_present` 是 S/T/A/R 四段裡**驗證過**命中的段數,0–4。
    """
    hit = max(0, min(4, star_present))
    lo, hi = STAR_BANDS[hit]
    n = stats.quantifier_count
    pos = min(n, QUANT_FULL) / QUANT_FULL
    score = int(round(lo + pos * (hi - lo)))
    return score, f"STAR 命中 {hit}/4 段,數字與量化詞 {n} 處"


def score_fluency(stats: TextStats, measurable: bool) -> tuple[int | None, str]:
    """表達流暢度。回傳 (分數, 依據說明)。分數可能是 None。

    【不可測時為什麼回 None 而不是 72】
    原本回 72(正常區間下緣),理由寫的是「不可以給高分,那會讓打字輸入的人
    佔便宜;也不可以給低分,那是懲罰未知」。那個推理是對的,結論是錯的:
    兩邊都不可以,代表這一項不該有分數,不代表它該拿中間值。

    72 一旦進了平均,就跟真的量到 72 完全一樣。使用者看到的是一個
    有依據的分數,而實際上那是「我們沒量到」的委婉說法。
    """
    if not measurable or not stats.char_count:
        why = "沒有文字" if not stats.char_count else "填充詞不可測(打字輸入或引擎已移除)"
        return None, why
    if stats.char_count < MIN_FLUENCY_CHARS:
        return None, f"全場只有 {stats.char_count} 字,填充詞密度沒有解析度"
    rate = stats.filler_count / stats.char_count * 100
    lo_edge = 0.0
    for edge, lo, hi in FLUENCY_BANDS:
        if rate <= edge:
            span = edge - lo_edge or 1.0
            score = hi - (rate - lo_edge) / span * (hi - lo)
            return int(round(score)), f"詞彙型填充詞每百字 {rate:.1f}"
        lo_edge = edge
    return 45, f"詞彙型填充詞每百字 {rate:.1f}"


def overall(subs: Sequence[SubScoreDTO]) -> tuple[float | None, int]:
    """總分與它的依據項數。只平均真的有分數的項。

    【為什麼搬到後端算】
    前端原本自己平均六項,而前端看不出一個數字是量到的還是補的。
    三個症狀同一個原因:缺項補的 0 被平均進去、取整數截斷(64.67 → 64)、
    陣列為空時填一個沒有依據的固定值。

    後端知道哪幾項是真的,所以只平均那幾項。

    【為什麼不回整數】
    保留小數讓前端決定顯示幾位。這裡先取整的話,前端之後想顯示
    64.7 就拿不回來了——而那正是 64.67 被截成 64 的同一個錯誤,
    只是搬到了後端。

    【為什麼要回項數】
    總分 72 可以是六項都量到,也可以是只有一項量到。後者是那一項的
    分數換了個名字。只給總分的話,兩種在畫面上一模一樣。
    """
    got = [s.score for s in subs if s.score is not None]
    if not got:
        return None, 0
    return sum(got) / len(got), len(got)


_MEASURE = "個|位|次|人|年|月|日|天|週|小時|分鐘|倍|成|%|％|元|份|家|間|種|項|篇|場"

_NUM_ARABIC = re.compile(r"[0-9０-９]+")
_NUM_CN = re.compile(rf"([一二三四五六七八九十百千萬兩]+)({_MEASURE})")
"""better 裡出現的數量。

中文數字**必須連量詞一起比對**,不可以只比對數字本身。
實測踩到的坑:回答裡有「資管系大三的學生」,better 寫「訪談了三位消費者」,
只比對「三」的話會因為「大三」而誤判成有依據。

中文數字也必須帶量詞才算數量,否則「一直」「不一定」「十分清楚」會被誤判。
阿拉伯數字則不需要量詞——那幾乎一定是數量。
"""


def ungrounded_numbers(better: str, source: str) -> list[str]:
    """回傳 better 裡有、但原回答沒有的數量。

    這是機械檢查,對應 prompt 的「不可新增數字」規則。
    實測那條規則會失守:回答只說「約訪談的時間」,模型寫成「訪談了三位消費者」。
    判斷型規則有殘留率,所以加一道機械檢查。

    【已知限制:只查得到數量】
    「這份企劃書後來被用於實際的行銷活動」這類沒有數字的捏造查不出來,
    那仍然只能靠 prompt。數量之所以值得單獨查,是因為它最常見、最具體,
    也是使用者被追問時最答不出來的一種。

    【比對必須連量詞】
    實測踩到的坑:逐字稿有「資管系大三」,better 寫「訪談了三位消費者」,
    只比對「三」會因為「大三」而誤判成有依據。所以比對的是「三位」。

    【誤判方向】
    這個檢查會漏抓(沒有數字的捏造),但不太會誤攔——
    誤攔的代價是清掉一句好建議,漏抓的代價是使用者背了一句他答不出來的話。
    兩者都不好,但目前的設計偏向少誤攔,因為誤攔是靜默的
    (使用者只看到少一段),漏抓至少還有人眼複查的機會。
    """
    src = fold_variants(source).replace(" ", "").replace(",", "").replace(",", "")
    bad: list[str] = []
    for m in _NUM_ARABIC.finditer(better):
        if m.group() not in src:
            bad.append(m.group())

    # 【從原文抽 token,只用折過的版本做比對】
    # 折疊的結果不可以拿去顯示:繁→簡的表把「三個」折成「三个」,
    # 而這裡回傳的 token 會進 notices(「出現原回答沒有的數字(三個)」)。
    # 從原文抽、折過之後比對,兩件事分開。
    folded = fold_variants(better)
    for m in _NUM_CN.finditer(better):
        token = m.group()          # 原文的 token,例如「三位」
        start, end = m.span()
        # 折疊是逐字元替換,長度不變,所以位置對得上
        if folded[start:end] not in src:
            bad.append(token)
    return bad


def build_transcript_text(turns: Sequence[TurnDTO]) -> str:
    """組出給 LLM 讀的逐字稿。被引擎切斷的輪次會標出來。

    標記的理由:被切斷跟自己講完是兩回事。不標的話 starParts 會因為
    缺 R 段而寫「你沒有交代結果」,那是責備一個被打斷的人。
    """
    parts = []
    for i, t in enumerate(turns, 1):
        parts.append(f"[第 {i} 題] {t.question}")
        mark = "(被切斷)" if t.ended_by == "timeout" else ""
        parts.append(f"[回答]{mark} {t.answer}")
    return "\n".join(parts)


def truncated_turns(turns: Sequence[TurnDTO]) -> list[int]:
    """被引擎切斷的輪次編號,從 1 起算。"""
    return [i for i, t in enumerate(turns, 1) if t.ended_by == "timeout"]


# ---------------------------------------------------------------------------
# 四個區塊的產出
# ---------------------------------------------------------------------------


async def gen_face(
    llm: LLMCall, transcript: str, stats_text: str, engine: str = "api"
) -> list[FaceDimensionDTO]:
    raw = await asyncio.to_thread(
        llm, compose_face_prompt(engine), f"【逐字稿】\n{transcript}\n\n【逐字稿統計】\n{stats_text}"
    )
    out = []
    by_letter = {ltr: name for ltr, name in FACE_DIMENSIONS}
    for item in parse_array(raw):
        if not isinstance(item, dict):
            continue
        letter = str(item.get("letter", "")).strip()
        if letter not in by_letter:
            continue
        out.append(
            FaceDimensionDTO(
                letter=letter,
                name=by_letter[letter],
                score=0,  # 分數由 subScores 決定,這裡不重複產生
                verdict=normalize_locale(str(item.get("verdict", "")).strip()),
                points=[
                    normalize_locale(str(p).strip())
                    for p in (item.get("points") or [])
                    if str(p).strip()
                ][:3],
            )
        )
    return out


async def gen_sub_scores(
    llm: LLMCall, transcript: str, stats_text: str, engine: str = "api"
) -> list[SubScoreDTO]:
    """五項。表達流暢度不在這裡,由 score_fluency() 算。"""
    raw = await asyncio.to_thread(
        llm,
        compose_sub_score_prompt(engine),
        f"【逐字稿】\n{transcript}\n\n【逐字稿統計】\n{stats_text}",
    )
    out = []
    for item in parse_array(raw):
        if not isinstance(item, dict):
            continue
        try:
            score = int(item.get("score", 0))
        except (TypeError, ValueError):
            score = 0
        out.append(SubScoreDTO(name=str(item.get("name", "")).strip(), score=score))
    return out


async def gen_question_feedbacks(
    llm: LLMCall, turns: Sequence[TurnDTO], engine: str = "api"
) -> tuple[list[QuestionFeedbackDTO], list[str]]:
    """逐題點評。better 只能重組回答,不吃履歷。

    不吃履歷的理由:有履歷模式是捏造的通道,而且跟漏講點區塊重複。
    """
    if not turns:
        return [], []
    body = ["【逐題內容】"]
    for i, t in enumerate(turns, 1):
        body += [f"{i}. 題目:{t.question}", f"   回答:{t.answer}"]
    raw = await asyncio.to_thread(llm, compose_question_prompt(engine), "\n".join(body))
    by_id = {}
    for item in parse_array(raw):
        if not isinstance(item, dict):
            continue
        try:
            by_id[int(item.get("id"))] = item
        except (TypeError, ValueError):
            continue

    out, notices = [], []
    for i, t in enumerate(turns, 1):
        item = by_id.get(i, {})
        better = normalize_locale(str(item.get("better", "")).strip())
        bad = ungrounded_numbers(better, t.answer)
        if bad:
            notices.append(
                f"報告:第 {i} 題的 better 出現原回答沒有的數字({'、'.join(bad)}),已清空"
            )
            better = ""
        out.append(
            QuestionFeedbackDTO(
                question=t.question,
                answer=t.answer,
                comment=normalize_locale(str(item.get("comment", "")).strip()),
                better=better,
            )
        )
    return out, notices


async def gen_star_parts(
    llm: LLMCall, transcript: str, has_truncated: bool = False, engine: str = "api"
) -> list[StarPartDTO]:
    raw = await asyncio.to_thread(
        llm,
        compose_star_prompt(has_truncated=has_truncated, engine=engine),
        f"【逐字稿】\n{transcript}",
    )
    names = dict(STAR_PARTS)
    out = []
    for item in parse_array(raw):
        if not isinstance(item, dict):
            continue
        key = str(item.get("key", "")).strip().upper()
        if key not in names:
            continue
        present = bool(item.get("present", False))
        out.append(
            StarPartDTO(
                key=key,
                name=names[key],
                present=present,
                # fromAnswer 不做 normalize_locale——那是原文引用,改了就不是原文
                from_answer=str(item.get("fromAnswer", "") or "").strip() if present else "",
                hint=normalize_locale(str(item.get("hint", "")).strip()),
            )
        )
    return out


# ---------------------------------------------------------------------------
# improvements:彙整,不打 LLM
# ---------------------------------------------------------------------------


def derive_improvements(
    dims: Sequence[FaceDimensionDTO], limit: int = MAX_IMPROVEMENTS
) -> list[str]:
    """從三維面向的 points 彙整。

    不另打 LLM 的理由:省一次呼叫;讓建議與評語必然一致。
    分開產生的話,面向說「補上數字」而 improvements 說「講慢一點」,
    使用者不知道該聽哪一個。

    取法是各面向輪流取一條,不是把某一個面向的全部取完——
    否則建議會集中在同一個面向。
    """
    buckets = [list(d.points) for d in dims]
    out: list[str] = []
    seen: set[str] = set()
    while buckets and len(out) < limit:
        progressed = False
        for b in buckets:
            if not b or len(out) >= limit:
                continue
            item = b.pop(0)
            key = item.strip()
            if key and key not in seen:
                seen.add(key)
                out.append(key)
                progressed = True
        if not progressed:
            break
    return out


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def emotion_turns(turns: Sequence[TurnDTO]) -> list[EmotionTurn]:
    """把每一輪擺成情緒判讀吃的形狀,順便把停頓算出來。

    停頓是實測值,只有語音且前端送了 `segment_ends_ms` 的時候才有。
    算不出來就是空字串,那一行不會進 prompt——印「停頓:無資料」會讓模型
    以為是「沒有停頓」,而兩者差別很大。
    """
    out = []
    for t in turns:
        timing = measure_timing(
            starts_ms=t.segment_starts_ms,
            ends_ms=t.segment_ends_ms,
            segments=t.answer_segments or [t.answer],
            input_mode=t.input_mode,
        )
        out.append(EmotionTurn(t.question, t.answer, describe_pauses(timing)))
    return out


async def generate_report(
    *,
    mode: str,
    turns: Sequence[TurnDTO],
    stats: TextStats,
    input_mode: str,
    resume_text: str,
    resume_grounded: bool,
    llm: LLMCall,
    engine: str = "api",
    llm_verbatim: LLMCall | None = None,
) -> ReportResponse:
    """產出報告。五個區塊並行,任一失敗不影響其他。

    llm_verbatim 給 starParts 用。fromAnswer 需要跨行原文引用,
    gpt-4o-mini 做不到,要用 gpt-4o。省略時退回 llm。
    """
    transcript = build_transcript_text(turns)
    stats_text, fluency_measurable = format_stats(stats, input_mode)
    star_llm = llm_verbatim or llm

    # 被引擎切斷的輪次。使用者還在講就被送出,不是他自己講完的——
    # 缺的段落不該算在他頭上。
    cut = truncated_turns(turns)

    # 可評性閘門。整場都沒有可評內容時,連問都不要問——
    # 讓 LLM 替一段沒有內容的逐字稿寫評語與分數,它會寫得出來,
    # 而且看起來很合理。那是最糟的失敗方式。
    gate = assess([t.answer for t in turns])

    blocks: dict[str, object] = {
        "questions": gen_question_feedbacks(llm, turns, engine),
        "star": gen_star_parts(star_llm, transcript, bool(cut), engine),
        # 第五塊。它問的是「他當時是什麼狀態」,跟其他四塊無關,
        # 所以併進同一次 gather:多一次呼叫但不多一輪等待。
        "emotion": read_emotions(
            turns=emotion_turns(turns),
            llm=llm,
            # 逐輪的模式,不是整場那個——整場的 input_mode 是給流暢度公式用的。
            # 混用就退回 unknown,見 batch_input_mode。
            input_mode=batch_input_mode([t.input_mode for t in turns]),
        ),
    }
    if gate.scored:
        blocks["face"] = gen_face(llm, transcript, stats_text, engine)
        blocks["subs"] = gen_sub_scores(llm, transcript, stats_text, engine)

    results, notices = await gather_blocks(blocks)
    if not gate.scored:
        notices.append(f"報告:{gate.reason()},六項分數與三維面向本次未評分")

    face = results.get("face", [])
    subs = list(results.get("subs", []))
    questions, q_notices = results.get("questions", ([], []))
    notices += q_notices
    star = results.get("star", [])

    # verbatim 機械檢查:對不上就把那一段降成未命中。
    # 不做自動修復——引用不存在的原文是嚴重錯誤,而且修復等於我們自己編一句。
    #
    # 這一段排在算分之前,因為內容深度要吃驗證過的 STAR 段數。
    # 用未驗證的版本等於讓模型編一句引用就把分數推上去。
    raw_answers = "\n".join(t.answer for t in turns)
    fixed_star = []
    for part in star:
        bad = part.present and part.from_answer and verify_star_verbatim(
            ReportResponse(mode=mode, star_parts=[part]), raw_answers
        )
        if bad:
            notices.append(
                f"報告:{VERBATIM_FAIL_MARKER}({part.key})"
                f",引用的字串不存在於逐字稿,該段已降為未命中:{part.from_answer[:40]}"
            )
            fixed_star.append(
                StarPartDTO(key=part.key, name=part.name, present=False, from_answer="",
                            hint=part.hint or "這一段沒有明確講到")
            )
        else:
            fixed_star.append(part)

    star_hits = sum(1 for p in fixed_star if p.present)

    # 表達流暢度與內容深度由公式決定,不採用 LLM 的值。
    # 能用公式算的不要問 LLM——實測填充詞率 3.0(正常)時模型給 71,沒跟著錨走。
    #
    # 但公式跑之前要先過可評性閘門:兩個公式量的都是密度與處數,
    # 短而乾淨的回答密度是 0,會落在最高的 band。實測「不知道。」四個字
    # 拿到流暢度 90、內容深度 55,比認真回答的 82 還高。見 pipeline.scorability。
    if not gate.scored:
        fluency = depth = None
        f_basis = d_basis = gate.reason()
    elif gate.anchored:
        # 全場都明確答不出來。他有回應,那是關於這幾項能力的觀察證據——
        # 給最低錨點是誠實的評分,不是不評分。見 scorability 的模組註解。
        fluency = depth = LOWEST_ANCHOR
        f_basis = d_basis = gate.reason()
    else:
        fluency, f_basis = score_fluency(stats, fluency_measurable)
        depth, d_basis = score_depth(stats, star_hits)

    subs = [s for s in subs if s.name not in ("表達流暢度", "內容深度")]
    subs += [
        SubScoreDTO(name="表達流暢度", score=fluency, basis=f_basis),
        SubScoreDTO(name="內容深度", score=depth, basis=d_basis),
    ]
    for label, value, basis in (("表達流暢度", fluency, f_basis), ("內容深度", depth, d_basis)):
        shown = "未評分" if value is None else str(value)
        notices.append(f"報告:{label} {shown} 由系統計算({basis})")

    # 三維面向的分數對齊 subScores,避免同一件事出現兩個不一致的數字。
    # 來源是未評分時這裡也是未評分——抄一個 0 過來就是把 subScores 剛修好的
    # 問題在隔壁欄位重演一次。
    by_name = {s.name: s.score for s in subs}
    pair = {"內": ("內容深度",), "構": ("邏輯清晰度",), "達": ("表達流暢度",)}
    face_scored = []
    for d in face:
        src = pair.get(d.letter, ())
        score = next((by_name[n] for n in src if n in by_name), None)
        prosody = None
        if d.letter == "達" and fluency_measurable:
            prosody = [ProsodyItem(label="填充詞", value=f"共 {stats.filler_count} 次(詞彙型)")]
        face_scored.append(
            FaceDimensionDTO(
                letter=d.letter, name=d.name, score=score,
                verdict=d.verdict, points=d.points, prosody=prosody,
            )
        )

    # 情緒不進分數也不進 improvements:它跟六項分數並列但不參與計算。
    # 平穩不標註,所以這個陣列是稀疏的,空陣列是正常結果不是失敗。
    emotions = [
        TurnEmotionDTO(turn_index=e.turn_index, label=e.label, quote=e.shown_quote)
        for e in results.get("emotion", [])
    ]

    report = ReportResponse(
        mode=mode,
        face_dimensions=face_scored,
        sub_scores=subs,
        question_feedbacks=questions,
        star_parts=fixed_star,
        improvements=derive_improvements(face_scored),
        resume_grounded=resume_grounded,
        notices=notices,
        turn_emotions=emotions,
    )
    if cut:
        report.notices.append(
            f"報告:第 {'、'.join(map(str, cut))} 題的回答被語音辨識切斷,"
            "那幾題缺少的段落不代表使用者沒講"
        )

    report = repair_report(report)

    # 總分要在 repair 之後算。repair 會把缺的項補成未評分,
    # 在它之前算等於照著一份還沒補齊的清單取平均。
    report.overall_score, report.score_basis = overall(report.sub_scores)
    if report.score_basis == 0:
        report.notices.append("報告:沒有任何一項評得出來,本場無法給總分")
    elif report.score_basis < len(SUB_SCORE_NAMES):
        report.notices.append(
            f"報告:總分由 {report.score_basis} 項算出(共 {len(SUB_SCORE_NAMES)} 項)"
        )
    return report
