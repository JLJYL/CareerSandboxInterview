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

from dataclasses import dataclass

from app.contracts.interview_protocols import JDInput
from app.prompts.interview_personas import personas_for

# ---------------------------------------------------------------------------
# 什麼時候算講完了
# ---------------------------------------------------------------------------

STAR_PART_KEYS: tuple[str, ...] = ("S", "T", "A", "R")


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
