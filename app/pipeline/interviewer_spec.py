"""面試官規格:由職缺與難度推導,不是生成。

【推導不是生成】
候選人生成要多樣性——同一個 JD 每次跑應該產出不一樣的競爭者。
面試官相反:同一個 JD 加同一個難度,每次都該是同一位。
一對一只有一位面試官,而他是誰幾乎完全由職缺決定,不需要隨機性。

所以這個檔裡沒有 LLM,全部是純函式。

【為什麼難度要再加一層 accept 條件】
`DIFFICULTY_TRIGGERS` 已經依難度寫了「追問要問什麼」,三段文字差異很大,
但實測三種難度產出的問句幾乎逐字相同。

原因是那三段講的都是**要問什麼**,沒有一段講**什麼時候不要問**。
模型照著「依序檢查」跑下去,每一輪都找得到可以追的東西,於是三種難度
的追問率一樣。

這裡補的是另一半:

    DIFFICULTY_TRIGGERS   追問要問什麼        已存在
    AcceptRule            什麼時候算講完了    這個檔

兩者不衝突,問的是不同的問題。而且 AcceptRule 的欄位是可判定的,
所以「新手比困難少追問」變成跑十場數次數就能驗的事——
上一輪失敗的真正原因是沒有判準,只能用讀的。

【為什麼槽位 id 要留著】
前端的頭像與色票以 persona id 當鍵(`peer_assertive`、`hr`、`tech`…),
顯示名稱之後會隨職缺調整。規格掛在槽位上,換了職缺也不會讓前端對不到圖。
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Sequence

from app.contracts.interview_protocols import JDInput
from app.prompts.interview_personas import personas_for

# ---------------------------------------------------------------------------
# 什麼時候算講完了
# ---------------------------------------------------------------------------

STAR_PART_KEYS: tuple[str, ...] = ("S", "T", "A", "R")

STAR_PART_GAP: dict[str, str] = {
    "S": "缺情境",
    "T": "缺任務",
    "A": "缺行動",
    "R": "缺結果",
}
GAP_QUANTIFICATION = "結果沒量化"
GAP_TRADEOFF = "沒有取捨"

GAP_VALUES: tuple[str, ...] = tuple(STAR_PART_GAP[p] for p in STAR_PART_KEYS) + (
    GAP_QUANTIFICATION,
    GAP_TRADEOFF,
)
"""缺口只會是這六個值其中一個,或空字串。

【為什麼跟放行條件放在同一個檔】
每一條放行條件剛好對應一個缺口——這不是巧合,是同一組事實的兩種讀法:
「夠不夠」與「差在哪」。放在兩個檔裡就會變成加一條條件卻忘了給它缺口,
然後那條條件永遠追問、永遠講不出要問什麼。

原本放在 `app/prompts/turn_decision.py`,因為那時 gap 是問模型的,
選項文字與解析必須共用一份清單。現在 gap 是算出來的,它的來源是
`AcceptRule` 不是 prompt,所以搬回來。prompt 那邊改成 import。
"""

def _declared_parts(star_parts: Sequence[str]) -> set[str]:
    """只數宣告過的段落。

    認不得的值丟掉,不要讓它墊高段數——墊高的方向是放行,
    而「多報一段就放行」正是這一整輪要修掉的失敗模式。
    """
    return {p for p in star_parts if p in STAR_PART_KEYS}


GAP_PRIORITY: tuple[str, ...] = (
    "缺行動",
    "缺結果",
    GAP_QUANTIFICATION,
    GAP_TRADEOFF,
    "缺任務",
    "缺情境",
)
"""不放行的時候先追哪一個缺口。只回一個,所以得排序。

這是判斷不是量測。順序的理由——

    缺行動      沒有行動就沒有東西可評,其他都免談
    缺結果      沒有結果沒辦法判斷做得好不好
    結果沒量化  有結果但不可查證。實際面試最常見的缺口就是這一型(A3)
    沒有取捨    這個難度明確要求、而他確實沒做到的一項
    缺任務      只透過段數下限發生作用,等於「四段裡再補一段」的湊數
    缺情境      同上,而且幾乎每個人開口第一句就在講情境

【為什麼改成一份優先序,不是「條件順序 + 段落順序」】
2b 原本的寫法是先看 `require_result`、再按段落順序挑、最後才看量化與取捨。
2d 看 dump 的時候發現它在困難難度的 A3 上算出「缺任務」——那一筆交代了
情境、行動、結果,真正缺的是數字。「你當時的任務是什麼」對一個已經講完
情境行動結果的人來說價值很低,而「快了多少」是整段話能不能被查證的關鍵。

錯的原因是把 `accepts()` 的條件順序當成了追問的價值順序。那是兩件事:
`accepts()` 的順序只影響它怎麼短路,不影響結果;追問的順序決定問哪一句。
所以改成一份單獨的優先序,照價值排。

這個順序只決定**先**問哪一個,不決定會不會問到——`max_depth` 允許的範圍內
補完一個缺口之後,下一輪會算出下一個。困難的 max_depth 是 3。

階梯裡七筆答案第一句全是情境,所以把情境排在後面在實測資料上站得住。
"""


@dataclass(frozen=True)
class AcceptRule:
    """回答滿足這些條件就放行,不再追問。

    每一欄都是可判定的,不是語氣形容詞。「更嚴格」沒辦法照著做,
    「缺結果段就追」可以。
    """

    min_star_parts: int
    """STAR 四段裡至少要交代幾段。"""

    require_result: bool
    """是否一定要有結果段。缺結果最值得追,那是成果的唯一證據。"""

    require_quantification: bool
    """是否要求可查證的數字。只有「提升了」「變好了」不算。"""

    require_tradeoff: bool
    """是否要求交代取捨——放棄了什麼、判斷依據是什麼。"""

    max_depth: int
    """同一個缺口最多追幾次。到了就換題,不管他補得完整沒有。

    這是計數不是判讀:追到第三次還講不出來,繼續追只是折磨,
    而那本來就該誠實反映在報告裡。
    """

    def accepts(
        self,
        *,
        star_parts: Sequence[str],
        quantified: bool,
        tradeoff: bool,
    ) -> bool:
        """這些事實夠不夠放行。

        【為什麼判斷在這裡而不是在 prompt 裡】
        1d 實測:放行條件寫進 prompt 兩次都沒生效。三種難度的追問率都是 0.500,
        而「有沒有可查證的數字」一個 bit 就能預測 180/180 次決定——
        模型在看那一個訊號,不是在對照條件。

        五個欄位本來就是可判定的,所以這一步根本不該問模型。
        「能用程式算的不要問 LLM」。

        【為什麼收原始值不收 AnswerFacts】
        `star_read` 要 import `STAR_PART_KEYS`,所以這個檔不能反過來 import 它,
        不然就循環了。想用物件的那一端有 `AnswerFacts.accepted_by(rule)`。

        同一支函式同時吃標註與判讀結果,是 1d 之後唯一確定有效的做法:
        量測算期望值、線上算真實決定,兩邊同一條路。
        """
        parts = _declared_parts(star_parts)
        if len(parts) < self.min_star_parts:
            return False
        if self.require_result and "R" not in parts:
            return False
        if self.require_quantification and not quantified:
            return False
        if self.require_tradeoff and not tradeoff:
            return False
        return True

    def first_gap(
        self,
        *,
        star_parts: Sequence[str],
        quantified: bool,
        tradeoff: bool,
    ) -> str:
        """最該追的那一個缺口。放行的話回空字串。

        【為什麼只回一個】
        1d 實測自由填寫的 gap 時量到:模型列幾項取決於**放行條件寫了幾條**,
        不取決於答案。同一筆 A1,新手點 1–2 段,困難點 3–4 段。
        列一串看起來很完整,其實是在數 prompt 的行數。

        所以只回一個,而且是算出來的——哪一條條件沒過是確定的事,不用問。

        【順序】
        照 `GAP_PRIORITY` 挑第一個「確實是這次不放行的原因」的缺口。
        那份順序是判斷,理由寫在它旁邊。
        """
        if self.accepts(star_parts=star_parts, quantified=quantified, tradeoff=tradeoff):
            return ""
        live = self._live_gaps(
            _declared_parts(star_parts), quantified=quantified, tradeoff=tradeoff
        )
        for gap in GAP_PRIORITY:
            if gap in live:
                return gap
        # accepts() 回 False 但沒有任何缺口成立,那是矛盾。
        # 不丟例外——決定路徑不該因為這個崩掉——但測試會逐格檢查這件事不發生。
        return ""

    def _live_gaps(
        self, parts: set[str], *, quantified: bool, tradeoff: bool
    ) -> set[str]:
        """這一次不放行的原因有哪些。

        「這一段缺了」不等於「這一段是不放行的原因」:段落只透過段數下限
        發生作用,所以段數夠了的時候缺哪一段都不是原因。
        少了這一層,新手難度會指出「沒有取捨」——而新手根本不要求取捨。
        """
        live: set[str] = set()
        short = len(parts) < self.min_star_parts
        for p in STAR_PART_KEYS:
            if p in parts:
                continue
            if short or (p == "R" and self.require_result):
                live.add(STAR_PART_GAP[p])
        if self.require_quantification and not quantified:
            live.add(GAP_QUANTIFICATION)
        if self.require_tradeoff and not tradeoff:
            live.add(GAP_TRADEOFF)
        return live


ACCEPT_RULES: dict[str, AcceptRule] = {
    "新手": AcceptRule(
        min_star_parts=2,
        require_result=False,
        require_quantification=False,
        require_tradeoff=False,
        max_depth=1,
    ),
    "中等": AcceptRule(
        min_star_parts=3,
        require_result=True,
        require_quantification=True,
        require_tradeoff=False,
        max_depth=2,
    ),
    "困難": AcceptRule(
        min_star_parts=4,
        require_result=True,
        require_quantification=True,
        require_tradeoff=True,
        max_depth=3,
    ),
}
"""三種難度的放行條件。

【預期】
條件逐級變嚴,所以追問率應該是 新手 < 中等 < 困難。
這是這一版唯一要驗的事,而且用數的:跑十場數每場的追問次數。

Panfilova et al.(2026)量的正是這個數字——GPT-5 Chat 一場 19.4 次、
Grok 4 一場 45.7 次,而且後者被三位專家評為 over-questioning。
所以追問率不只要分得開,高的那一端也不能無上限,`max_depth` 管這件事。

【如果三種難度的追問率還是分不開】
那就是這次也失敗了。但這次量得出來,不像上次只能說「讀起來差不多」。
"""

DEFAULT_DIFFICULTY = "中等"


def accept_rule_for(difficulty: str) -> AcceptRule:
    """難度對應的放行條件。認不得的值退回中等。"""
    return ACCEPT_RULES.get(difficulty, ACCEPT_RULES[DEFAULT_DIFFICULTY])


# ---------------------------------------------------------------------------
# 每一個欄位都要有人用
# ---------------------------------------------------------------------------

_ACCEPTS_FIELDS = frozenset(
    {"min_star_parts", "require_result", "require_quantification", "require_tradeoff"}
)
_DEPTH_FIELDS = frozenset({"max_depth"})

_declared = _ACCEPTS_FIELDS | _DEPTH_FIELDS
_actual = frozenset(f.name for f in fields(AcceptRule))
if _declared != _actual:
    raise RuntimeError(
        "AcceptRule 的欄位變了,但沒人處理新的那一個。\n"
        f"  只有這份清單知道的:{sorted(_declared - _actual)}\n"
        f"  還沒被處理的:{sorted(_actual - _declared)}\n"
        "新增一條條件卻沒接進 accepts() / first_gap(),"
        "結果是那條條件永遠不生效而且全部測試照樣綠。"
    )

# 「每一個缺口都算得出來」這件事要掃 16 種段落組合 × 3 難度 × 量化 × 取捨,
# 放在 import 時跑太重,而且掃不過的時候 import 爆掉比測試紅更難查。
# 那條在 tests/test_interviewer_spec.py 裡逐格檢查。


# ---------------------------------------------------------------------------
# 面試官規格
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InterviewerSpec:
    """一位面試官。掛在槽位上,不掛在顯示名稱上。

    【cares_about 為什麼不依槽位切開】
    「這個回答有沒有缺口」跟誰在問無關,四個槽位看的是同一份職缺需求。
    決定哪一輪輪到誰的是 persona 的 `routes_when`,那個已經存在,
    不需要在這裡再切一次——切了就要發明一套「技能屬於 HR 還是技術主管」
    的對應,而那個對應沒有唯一正確答案。
    """

    slot_id: str
    """穩定鍵。對齊 PersonaSpec.id,前端的頭像與色票以此查表。"""

    display_name: str
    """目前沿用 persona 的顯示名稱。之後依職缺調整時只改這一欄。"""

    role: str
    """interviewer / moderator。沿用 PersonaSpec.role。"""

    cares_about: tuple[str, ...]
    """這個職缺在意什麼。直接取結構化 JD 的需求項目,不另外生成。

    JD 的順序即優先序(見 JDInput),所以這裡保持原順序不重排。
    """

    accept: AcceptRule

    def has_jd(self) -> bool:
        """JD 抽取失敗或使用者沒填時為空。空的時候只能靠 STAR 判缺口。"""
        return bool(self.cares_about)


def derive_specs(
    mode: str,
    *,
    jd: JDInput | None = None,
    difficulty: str = DEFAULT_DIFFICULTY,
    group_interviewers: int = 1,
    group_size: int = 4,
) -> tuple[InterviewerSpec, ...]:
    """這一場的面試官規格。純函式——同樣的輸入永遠回同樣的結果。

    槽位來自既有的 persona 設定,所以 panel 仍然是三位、群面仍然是
    一位主持或三位主管。這裡只替每個槽位補上「在意什麼」與「何時放行」。

    競爭者(role == "peer")不在這裡——他們不追問,動作空間完全不同。
    """
    cares = tuple(jd.required_skills) if jd else ()
    rule = accept_rule_for(difficulty)
    return tuple(
        InterviewerSpec(
            slot_id=p.id,
            display_name=p.display_name,
            role=p.role,
            cares_about=cares,
            accept=rule,
        )
        for p in personas_for(mode, group_interviewers, group_size)
        if p.role != "peer"
    )
