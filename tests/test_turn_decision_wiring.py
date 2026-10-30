"""決策接進 next_turn 之後的行為。

【為什麼要獨立一支】
`test_turn_decision.py` 測的是決定本身,`test_interview_live.py` 測的是
生成那一段(它不傳 decision,維持舊行為)。這裡測的是**兩者接起來**
之後才出現的事:

    is_follow_up 從猜變成決定
    決定會進到生成端的 user 訊息
    輪次上限比決定大

【一個差點漏掉的】
decision 接上之後,`test_api` 的假 LLM 因為沒有決策分支,每一輪都走降級
(source=fallback、action=NEXT),`is_follow_up` 從 True 變成 False——
而 602 個測試全綠,沒有一個發現。假 LLM 少一條分支就會量到降級路徑,
所以這裡明確測 source。
"""

from __future__ import annotations

import json

import pytest

from app.pipeline.interview_live import next_turn
from app.pipeline.interviewer_spec import derive_specs
from app.pipeline.turn_decision import CLARIFY, FOLLOW_UP, NEXT, TurnDecision

ANSWER = "我把每週三小時的人工整理改成自動產出。"
GEN = {"nextQuestion": "那個時間是怎麼量的?", "topic": "成果", "speaker": ""}


def gen_llm(system: str, user: str) -> str:
    """只負責生成。決策是另外傳進去的,不經過這裡。"""
    gen_llm.seen.append(user)
    return json.dumps(GEN, ensure_ascii=False)


gen_llm.seen = []  # type: ignore[attr-defined]


@pytest.fixture(autouse=True)
def _clear():
    gen_llm.seen = []  # type: ignore[attr-defined]


def spec():
    return derive_specs("single", jd=None, difficulty="中等")[0]


async def turn(decision, **kw):
    return await next_turn(
        mode=kw.pop("mode", "single"),
        answer=kw.pop("answer", ANSWER),
        follow_up_idx=kw.pop("follow_up_idx", 1),
        asked_questions=kw.pop("asked_questions", ["資料分析"]),
        fallback=kw.pop("fallback", ["備援題一"]),
        llm=gen_llm,
        decision=decision,
        **kw,
    )


# ---------------------------------------------------------------------------
# is_follow_up 從猜變成決定
# ---------------------------------------------------------------------------


async def test_no_decision_keeps_the_old_guess():
    """不傳 decision 時維持舊行為,既有測試才測得到生成那一段。"""
    r = await turn(None)
    assert r.is_follow_up is True


@pytest.mark.parametrize(
    "action,expected",
    [(FOLLOW_UP, True), (CLARIFY, True), (NEXT, False)],
)
async def test_is_follow_up_comes_from_the_decision(action, expected):
    r = await turn(TurnDecision(action))
    assert r.is_follow_up is expected


async def test_next_runs_the_duplicate_check():
    """決定是 NEXT 時新主問題要過重複偵測——一對一以前永遠跳過這一關。"""
    r = await turn(
        TurnDecision(NEXT),
        asked_questions=["那個時間是怎麼量的?"],
        fallback=["換個方向:你最近學了什麼?"],
    )
    assert r.next_question == "換個方向:你最近學了什麼?"
    assert any("重複" in n for n in r.notices)


async def test_follow_up_is_not_killed_by_the_same_topic_check():
    """追問該留在同一個話題上,所以不能拿「問同一件事」當判準。

    這一條原本的 fixture 是拿**一模一樣**的問句來證明追問跳過查重。
    那個行為後來改掉了:追問現在會擋同一句話(`REPEAT_THRESHOLD`),
    所以 fixture 換成真的落在兩個門檻之間的一組,取自 `data/panel_v5.json`,
    重疊 0.688:主問題的門檻 0.6 會擋,追問的門檻 0.9 不擋。

    【這一組是改寫,不是兩個不同的問題,而那是刻意的】
    手上四次量測的資料全部來自同一個模型對同四段回答的輸出,所以 0.6–0.9
    那個區間裡找不到語意上真的不同的配對——`REPEAT_THRESHOLD` 的 docstring
    記了同一件事。0.9 放行改寫是取捨:拿 0.6 去擋會把合法的同話題追問
    一起殺掉,而那個代價比偶爾放過一句改寫大。

    要收緊得先有同一場裡不同輪的資料。
    """
    prior = "那你覺得在分析過程中,如何有效地確定要問的問題呢?"
    probe = "那你覺得在分析過程中,有哪些方法可以幫助你更清楚地定義問題呢?"

    from app.pipeline.interview_live import is_near_duplicate, is_verbatim_repeat
    from app.pipeline.question_form import is_closed_question

    assert is_near_duplicate(probe, [prior]), "fixture 要落在主問題門檻會擋的區間"
    assert not is_verbatim_repeat(probe, [prior]), "fixture 不該是同一句話"
    assert not is_closed_question(probe), (
        "fixture 不能是封閉式的,否則擋下來的是另一關,這一條就測不到東西"
    )

    def gen(system: str, user: str) -> str:
        gen.seen.append(user)
        return json.dumps({**GEN, "nextQuestion": probe}, ensure_ascii=False)

    gen.seen = []
    r = await next_turn(
        mode="single", answer=ANSWER, follow_up_idx=1,
        asked_questions=[prior], fallback=["備援題一"],
        llm=gen, decision=TurnDecision(FOLLOW_UP),
    )
    assert r.next_question == probe
    assert not any("重複" in n for n in r.notices)
    assert len(gen.seen) == 1, "沒有重複就不該多打一次重生成"


async def test_follow_up_that_repeats_a_question_is_regenerated():
    """同一句話問兩次要擋。

    `data/panel_v5.json` 量到 HR 的「那件事對你來說最難的是哪一段?」
    三輪一字不差。那三次分屬三場面試,使用者看不到;但同一場裡 HR 開口
    兩次就會踩到,而追問這條路原本完全不查重。

    擋到之後不能換備援題——備援題是另一個話題,那就不是追問了。
    追問沒有池可以換,所以重生成一次。
    """
    repeated = "那件事對你來說最難的是哪一段?"
    fixed = "那一段裡你花最多時間的是哪一塊?"
    out = iter([repeated, fixed])

    def gen(system: str, user: str) -> str:
        gen.seen.append(user)
        return json.dumps({**GEN, "nextQuestion": next(out)}, ensure_ascii=False)

    gen.seen = []
    r = await next_turn(
        mode="single", answer=ANSWER, follow_up_idx=1,
        asked_questions=[repeated], fallback=["備援題一"],
        llm=gen, decision=TurnDecision(FOLLOW_UP),
    )
    assert r.next_question == fixed
    assert any("重生成" in n for n in r.notices)
    assert len(gen.seen) == 2, "應該剛好重生成一次"
    assert repeated in gen.seen[1], (
        "重生成沒有把重複的那一句告訴模型,同樣的輸入只會給同樣的輸出"
    )
    assert "備援題一" != r.next_question, "追問不可以換成備援題,那是換話題"


async def test_a_second_repeat_is_sent_as_is_not_retried_again():
    """重生成還是重複就照原樣送出,不要再試。

    面試進行中每一次呼叫都是使用者在等。一句重複的問題比讓他多等一輪好,
    而且 notice 查得到。
    """
    repeated = "那件事對你來說最難的是哪一段?"

    def gen(system: str, user: str) -> str:
        gen.seen.append(user)
        return json.dumps({**GEN, "nextQuestion": repeated}, ensure_ascii=False)

    gen.seen = []
    r = await next_turn(
        mode="single", answer=ANSWER, follow_up_idx=1,
        asked_questions=[repeated], fallback=["備援題一"],
        llm=gen, decision=TurnDecision(FOLLOW_UP),
    )
    assert r.next_question == repeated
    assert any("沒有改善" in n for n in r.notices)
    assert len(gen.seen) == 2, "只能重生成一次,不可以一直試"


# ---------------------------------------------------------------------------
# 決定要傳到生成端
# ---------------------------------------------------------------------------


async def test_decision_reaches_the_generator():
    """不講的話決定說 NEXT 而模型生出追問,欄位就跟實際內容不符。"""
    await turn(TurnDecision(FOLLOW_UP, gap="缺結果"))
    sent = gen_llm.seen[-1]  # type: ignore[attr-defined]
    assert "【這一輪要做什麼】" in sent
    assert "不要換題" in sent


async def test_the_gap_does_not_reach_the_generator():
    """只傳動作,不傳 gap。

    1d 實測:A3 那筆答案講了結果只是沒有數字,20/20 次 gap 都寫「結果段」。
    照著生成的追問會去問他剛剛講過的東西。生成端看得到完整的回答,
    不需要一個會指錯的標籤。
    """
    await turn(TurnDecision(FOLLOW_UP, gap="缺結果"))
    sent = gen_llm.seen[-1]  # type: ignore[attr-defined]
    assert "缺結果" not in sent
    assert "缺的是" not in sent


async def test_clarify_tells_the_generator_not_to_apologise():
    await turn(TurnDecision(CLARIFY))
    sent = gen_llm.seen[-1]  # type: ignore[attr-defined]
    assert "不要為系統的問題道歉" in sent


async def test_no_decision_adds_nothing_to_the_prompt():
    await turn(None)
    assert "【這一輪要做什麼】" not in gen_llm.seen[-1]  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# 輪次上限比決定大
# ---------------------------------------------------------------------------


async def test_turn_cap_overrides_the_decision():
    """到上限的那一句就是最後一個主問題,語意上必然是 NEXT。

    決定說 FOLLOW_UP 也一樣——輪次是確定性事實,比判斷大。
    """
    r = await turn(TurnDecision(FOLLOW_UP), follow_up_idx=4)
    assert r.is_follow_up is False
    assert r.should_advance is True
    assert "【這一輪要做什麼】" not in gen_llm.seen[-1]  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# 降級看得見
# ---------------------------------------------------------------------------


async def test_source_is_recorded_in_notices():
    """code / llm / fallback 要看得出來,不然會把降級當成真正的決定。"""
    r = await turn(TurnDecision(NEXT, source="fallback"))
    assert any("fallback" in n for n in r.notices)
    r = await turn(TurnDecision(FOLLOW_UP, source="code"))
    assert any("code" in n for n in r.notices)


# ---------------------------------------------------------------------------
# NEXT 要換話題,但不要丟掉回答
# ---------------------------------------------------------------------------


async def test_next_tells_the_generator_to_reuse_the_answer():
    """NEXT 的舊寫法把兩件事混在一起,模型讀成「不要用那段回答」。

    panel v3 量到「在這個過程中,你有沒有遇到什麼挑戰?」出現五次,
    5/5 都在 NEXT,FOLLOW_UP 的三十筆一次都沒有;那一輪三位的角色重疊
    0.551,其他三段是 0.31–0.38。回答是人格唯一的材料,不准用回答,
    三份人格就沒有東西可以分化。

    三段式面試每次換題都走這條路,所以這裡切乾淨等於整場變成四段
    互不相干的小面試。
    """
    await turn(TurnDecision(NEXT))
    sent = gen_llm.seen[-1]  # type: ignore[attr-defined]
    assert "【這一輪要做什麼】" in sent
    assert "他剛剛講的內容照用" in sent, "沒有明講回答要用,就會被讀成不要用"
    assert "接得上他剛說的話" in sent


async def test_next_still_means_change_the_topic():
    """照用回答不等於可以繼續挖同一個坑。

    這條跟上面那條是一對:只留「照用回答」而拿掉換題的要求,
    NEXT 就退化成 FOLLOW_UP,而 `is_follow_up` 會跟實際內容不符——
    那正是這個欄位最初被誤判時出的事。
    """
    await turn(TurnDecision(NEXT))
    sent = gen_llm.seen[-1]  # type: ignore[attr-defined]
    assert "換一個新的能力面向" in sent
    assert "不要再追問同一個話題" in sent


async def test_next_carries_no_prohibition_list():
    """不要在這裡加「不要問通用問題」那種禁令。

    HR 人格那邊量過同型的禁令,兩次 FOLLOW_UP 合計:有禁令 2/24、
    沒禁令 3/44,沒有差別。禁令只是在一支已經很長的 prompt 裡多佔位置。
    這一句只給材料與動作。
    """
    from app.prompts.turn_decision import ACTION_INSTRUCTION

    nxt = ACTION_INSTRUCTION["NEXT"]
    assert "通用" not in nxt and "任何一段" not in nxt, (
        "又在加禁令了——量過沒有效果,先看 interview_personas.py 上方的數字"
    )


async def test_next_keeps_is_follow_up_false_even_though_it_quotes_the_answer():
    """欄位記的是有沒有換話題,不是有沒有提到剛剛那段。

    新版的問題會引用上一段回答,畫面上看起來像追問。但前端的 asked_topics
    與 `is_near_duplicate` 都靠這個欄位判斷「這是不是一個新的主問題」,
    而新的主問題本來就該做重複比對。所以維持 False 是刻意的。
    """
    r = await turn(TurnDecision(NEXT))
    assert r.is_follow_up is False


async def test_next_still_goes_through_the_duplicate_check():
    """上一條的後果:NEXT 仍然要查重複,引用回答不能讓它繞過。

    `interview_live` 那段查重是 `if not is_follow_up` 才跑的。
    維持 False 的理由就是要留著這條路,所以這裡把它釘住——
    有人把 NEXT 改成 True 的話,重複的主問題會直接送出去。
    """
    r = await turn(TurnDecision(NEXT), asked_questions=[GEN["nextQuestion"]])
    assert any("重複" in n for n in r.notices), (
        "NEXT 沒有進查重路徑,新主問題可以跟問過的一模一樣"
    )


async def test_clarify_is_not_treated_as_a_repeat():
    """CLARIFY 重複是合理的,不要擋。

    它的 `is_follow_up` 也是 True,所以會掉進追問那一條路。但 CLARIFY 問的是
    「那個詞再講一次」——裝置端辨識把同一個字連續轉錯兩次的話,同一句
    CLARIFY 本來就該再問一次。

    而且重生成的指示是「問他剛剛那段裡還沒講清楚的另一個部分」,
    那對 CLARIFY 是錯的方向:它不是要問別的部分,是要同一個部分再聽一次。
    """
    repeated = "你剛剛說的那個工具名稱,可以再講一次嗎?"

    def gen(system: str, user: str) -> str:
        gen.seen.append(user)
        return json.dumps({**GEN, "nextQuestion": repeated}, ensure_ascii=False)

    gen.seen = []
    r = await next_turn(
        mode="single", answer=ANSWER, follow_up_idx=1,
        asked_questions=[repeated], fallback=["備援題一"],
        llm=gen, decision=TurnDecision(CLARIFY),
    )
    assert r.next_question == repeated
    assert not any("重生成" in n for n in r.notices)
    assert len(gen.seen) == 1, "CLARIFY 不該進重生成那條路"


def test_the_clarify_name_cannot_drift_from_the_decision_module():
    """`interview_live` 複寫了一份 CLARIFY,兩邊不能分岔。

    直接匯入 `app.pipeline.turn_decision` 會循環(它匯入 interview_live),
    所以那個值是手寫的。分岔的話 CLARIFY 會被當成追問送去重生成,
    而那個錯是靜默的——測試不會紅,面試照跑,只是方向錯了。
    """
    from app.pipeline.interview_live import _CLARIFY
    from app.pipeline.turn_decision import CLARIFY as CANONICAL

    assert _CLARIFY == CANONICAL


# ---------------------------------------------------------------------------
# 封閉式問句
# ---------------------------------------------------------------------------


async def test_a_closed_question_is_regenerated():
    """只能用是/否答完的問題要重生成。

    四次量測 144 題裡 36 題(25%)是這樣。禁則第 2 條明文禁過了,沒生效。

    為什麼值得擋:面試官問「你有沒有遇到什麼挑戰」,使用者答「有啊還好」,
    然後評分那邊的內容深度扣他分。題目是封閉的,分數算在答題的人頭上。
    """
    closed = "在這個過程中,你有沒有遇到什麼挑戰?"
    fixed = "那一段裡你花最多時間的是哪一塊?"
    out = iter([closed, fixed])

    def gen(system: str, user: str) -> str:
        gen.seen.append(user)
        return json.dumps({**GEN, "nextQuestion": next(out)}, ensure_ascii=False)

    gen.seen = []
    r = await next_turn(
        mode="single", answer=ANSWER, follow_up_idx=1,
        asked_questions=["資料分析"], fallback=["備援題一"],
        llm=gen, decision=TurnDecision(FOLLOW_UP),
    )
    assert r.next_question == fixed
    assert any("是/否" in n for n in r.notices)
    assert len(gen.seen) == 2
    assert "有」或「沒有」答完" in gen.seen[1], (
        "重生成沒說是哪裡不對,同樣的輸入只會給同樣的輸出"
    )


async def test_a_closed_new_main_question_is_also_regenerated():
    """新主問題也要過這一關,不是只有追問。

    量測資料裡 NEXT 那兩次的封閉率(25% / 22%)跟 FOLLOW_UP 兩次
    (31% / 22%)差不多,兩條路都要擋。
    """
    closed = "那在這個過程中,你有沒有跟其他部門合作的經驗?"
    fixed = "跨部門那段你實際上是怎麼談的?"
    out = iter([closed, fixed])

    def gen(system: str, user: str) -> str:
        gen.seen.append(user)
        return json.dumps({**GEN, "nextQuestion": next(out)}, ensure_ascii=False)

    gen.seen = []
    r = await next_turn(
        mode="single", answer=ANSWER, follow_up_idx=1,
        asked_questions=["資料分析"], fallback=["備援題一"],
        llm=gen, decision=TurnDecision(NEXT),
    )
    assert r.next_question == fixed
    assert any("是/否" in n for n in r.notices)


async def test_both_faults_share_one_regeneration():
    """同一句同時重複又封閉,只能花一次呼叫。

    面試進行中每一次呼叫都是使用者在等。兩個檢查各自重生成的話,
    最壞情況一輪要多等兩次。
    """
    bad = "在這個過程中,你有沒有遇到什麼挑戰?"
    fixed = "那一段裡你花最多時間的是哪一塊?"
    out = iter([bad, fixed])

    def gen(system: str, user: str) -> str:
        gen.seen.append(user)
        return json.dumps({**GEN, "nextQuestion": next(out)}, ensure_ascii=False)

    gen.seen = []
    r = await next_turn(
        mode="single", answer=ANSWER, follow_up_idx=1,
        asked_questions=[bad], fallback=["備援題一"],
        llm=gen, decision=TurnDecision(FOLLOW_UP),
    )
    assert r.next_question == fixed
    assert len(gen.seen) == 2, "兩種毛病共用一次重生成"
    note = next(n for n in r.notices if "問句" in n)
    assert "同一句" in note and "是/否" in note, f"兩種毛病都要記進 notice:{note}"
    sent = gen.seen[1]
    assert "重複的是" in sent and "有」或「沒有」答完" in sent, (
        "一次重生成要把兩個問題都講給模型聽"
    )


async def test_a_clarify_may_be_a_yes_no_question():
    """CLARIFY 不進封閉式那一關。

    「你是說 Angular 嗎?」就是一個 yes/no 問句,而那正是 CLARIFY 的形狀——
    它要的是確認一個聽錯的詞,不是一段敘述。
    """
    confirm = "你剛剛說的是 Angular 嗎?"

    from app.pipeline.question_form import is_closed_question

    assert is_closed_question(confirm), "fixture 要真的是封閉式,否則測不到豁免"

    def gen(system: str, user: str) -> str:
        gen.seen.append(user)
        return json.dumps({**GEN, "nextQuestion": confirm}, ensure_ascii=False)

    gen.seen = []
    r = await next_turn(
        mode="single", answer=ANSWER, follow_up_idx=1,
        asked_questions=["資料分析"], fallback=["備援題一"],
        llm=gen, decision=TurnDecision(CLARIFY),
    )
    assert r.next_question == confirm
    assert not any("是/否" in n for n in r.notices)
    assert len(gen.seen) == 1, "CLARIFY 不該進重生成那條路"


async def test_a_still_closed_regeneration_is_sent_as_is():
    """重生成還是封閉就照原樣送出,不要再試。"""
    closed = "在這個過程中,你有沒有遇到什麼挑戰?"
    still = "那你有沒有遇到什麼困難?"

    def gen(system: str, user: str) -> str:
        gen.seen.append(user)
        return json.dumps(
            {**GEN, "nextQuestion": still if gen.seen[1:] else closed},
            ensure_ascii=False,
        )

    gen.seen = []
    r = await next_turn(
        mode="single", answer=ANSWER, follow_up_idx=1,
        asked_questions=["資料分析"], fallback=["備援題一"],
        llm=gen, decision=TurnDecision(FOLLOW_UP),
    )
    assert r.next_question == closed, "沒有改善就留原本那一句,不要換成同樣壞的"
    assert any("沒有改善" in n for n in r.notices)
    assert len(gen.seen) == 2


# ---------------------------------------------------------------------------
# 維度派發
# ---------------------------------------------------------------------------


def _panel(decision, counts=None, **kw):
    """panel 一輪。記下派發有沒有被呼叫。"""
    seen = {"dispatched": False}

    def llm(system: str, user: str) -> str:
        if "決定這一輪由誰開口" in system:
            seen["dispatched"] = True
            return "HR 主管"
        return json.dumps({**GEN, "speaker": "x"}, ensure_ascii=False)

    spoken = []
    for name, n in (counts or {}).items():
        spoken += [name] * n
    return llm, seen, spoken


async def test_a_missing_dimension_routes_to_its_owner_without_asking_the_model():
    """缺的維度有負責人就直接給他,不問模型。

    語意派發是在沒有判準時用的。有判準還去問模型只是多一次呼叫加一次
    不確定——而那個不確定正是 panel 量測裡「同一段回答派給不同人」的來源。
    """
    llm, seen, spoken = _panel(None)
    r = await next_turn(
        mode="panel", answer=ANSWER, follow_up_idx=1, asked_questions=["資料分析"],
        spoken_by=spoken, question="q", fallback=["備援題一"], llm=llm,
        decision=TurnDecision(FOLLOW_UP, gap="沒有取捨", dimension="取捨"),
    )
    assert r.speaker == "用人主管", "取捨那一維的負責人是用人主管"
    assert not seen["dispatched"], "有判準就不該再問模型誰開口"
    assert any("取捨" in n and "用人主管" in n for n in r.notices)


async def test_a_gap_with_no_owner_falls_back_to_semantic_dispatch():
    """缺情境與缺任務刻意沒有負責人。

    那兩個是敘事完整度,不是某一位的守備範圍。硬塞一個負責人就是在發明
    那套「技能屬於誰」的對應,而那個對應沒有唯一正確答案。
    """
    from app.pipeline.interviewer_spec import GAP_DIMENSION

    assert "缺情境" not in GAP_DIMENSION
    llm, seen, spoken = _panel(None)
    r = await next_turn(
        mode="panel", answer=ANSWER, follow_up_idx=1, asked_questions=["資料分析"],
        spoken_by=spoken, question="q", fallback=["備援題一"], llm=llm,
        decision=TurnDecision(FOLLOW_UP, gap="缺情境", dimension=""),
    )
    assert seen["dispatched"], "沒有判準就該照舊問模型"
    assert r.speaker == "HR 主管"


async def test_the_dimension_owner_yields_when_someone_has_not_spoken():
    """同一個維度連續是缺的時候,不可以讓同一位連問到底。

    使用者答得越一致,同一個維度就越可能連續被判成缺的那一個。
    平衡規則在派發 prompt 裡本來就有,但那是寫給模型的,而這條路不問模型。
    """
    llm, seen, spoken = _panel(None, counts={"用人主管": 2})
    r = await next_turn(
        mode="panel", answer=ANSWER, follow_up_idx=1, asked_questions=["資料分析"],
        spoken_by=spoken, question="q", fallback=["備援題一"], llm=llm,
        decision=TurnDecision(FOLLOW_UP, gap="沒有取捨", dimension="取捨"),
    )
    assert seen["dispatched"], "該讓位的時候要退回語意派發"
    assert r.speaker == "HR 主管"
    assert any("還沒開口" in n for n in r.notices)


async def test_the_dimension_owner_keeps_the_turn_before_the_quota_bites():
    """邊界:講過一次還不算霸佔。

    門檻是「已發言兩次而且有人還沒開口」。這一條跟上一條夾住那個邊界,
    有人改門檻會紅在這裡。
    """
    llm, seen, spoken = _panel(None, counts={"用人主管": 1})
    r = await next_turn(
        mode="panel", answer=ANSWER, follow_up_idx=1, asked_questions=["資料分析"],
        spoken_by=spoken, question="q", fallback=["備援題一"], llm=llm,
        decision=TurnDecision(FOLLOW_UP, gap="沒有取捨", dimension="取捨"),
    )
    assert r.speaker == "用人主管"
    assert not seen["dispatched"]


async def test_force_speaker_still_beats_the_dimension():
    """量測工具用 force_speaker 繞過派發,維度不能把它蓋掉。

    不然 panel 量測又會變成「同角色等於同一段回答」那個問題——
    第一版就是栽在派發決定了誰開口。
    """
    llm, seen, spoken = _panel(None)
    r = await next_turn(
        mode="panel", answer=ANSWER, follow_up_idx=1, asked_questions=["資料分析"],
        spoken_by=spoken, question="q", fallback=["備援題一"], llm=llm,
        decision=TurnDecision(FOLLOW_UP, gap="沒有取捨", dimension="取捨"),
        force_speaker="技術主管",
    )
    assert r.speaker == "技術主管"
    assert not seen["dispatched"]


async def test_a_dimension_does_nothing_in_single_mode():
    """一對一沒有別人可以派,維度不該產生任何差別。

    `dimension_speaker` 回空字串是因為一對一的 display_name 是空的——
    前端只有一位面試官,不顯示說話者。
    """
    from app.pipeline.interview_live import dimension_speaker

    assert dimension_speaker("single", "取捨") == ""
    assert dimension_speaker("panel", "取捨") == "用人主管"
    assert dimension_speaker("panel", "不存在的維度") == ""


async def test_the_struggle_dimension_is_dormant_until_the_fact_is_read():
    """`STRUGGLE_FACT` 接上之前,那一維永遠不會觸發。

    這是兩邊各自進行的保證:成員A 還沒把那件事加進 `FACTS` 的期間,
    派發行為與現在完全相同,不會有壞掉的中間狀態。

    會壞的寫法是直接看 `facts.read(STRUGGLE_FACT).present`——那對沒讀過的鍵
    回 False,跟「讀過了他確實沒講」長得一樣,於是 HR 會拿到所有 NEXT 輪。
    """
    from app.pipeline.interviewer_spec import STRUGGLE_FACT
    from app.pipeline.star_read import ASSIGN, AnswerFacts, FactRead
    from app.pipeline.turn_decision import struggle_dimension

    unread = AnswerFacts(reads={}, read_by={"S": ASSIGN})
    assert struggle_dimension(unread) == "", "沒讀到不等於缺"

    read_absent = AnswerFacts(
        reads={STRUGGLE_FACT: FactRead(False, "", False)},
        read_by={STRUGGLE_FACT: ASSIGN},
    )
    assert struggle_dimension(read_absent) == "自述困難"

    read_present = AnswerFacts(
        reads={STRUGGLE_FACT: FactRead(True, "那時候我卡在不知道要問什麼", True)},
        read_by={STRUGGLE_FACT: ASSIGN},
    )
    assert struggle_dimension(read_present) == ""


async def test_group_mode_does_not_use_dimension_dispatch():
    """群面不走維度派發,那是 panel 的機制。

    兩個理由。機制上:群面場上有三位主管加三位 AI 應徵者,而 `derive_specs`
    不含 peer,所以維度派發永遠選不到競爭者——會把前幾輪固定給主管,
    而群面的價值就在應徵者之間的互動。

    範疇上:群面的「誰接話」問的是討論動態(有人被淹沒、討論變成兩人對話),
    不是「哪個能力維度還沒問到」。那兩件事用兩支不同的派發 prompt。

    這一條是我自己踩到才加的——維度派發第一版沒有擋模式,
    群面照樣會走進去。
    """
    seen = {"dispatched": False}

    def llm(system: str, user: str) -> str:
        if "決定這一輪由誰開口" in system:
            seen["dispatched"] = True
            return "AI-邏輯"
        # 應徵者的句子不能是問句(10/30 定案),不然這一輪會改由主持人講,
        # 而這一條要看的是派發選到了誰。
        return json.dumps(
            {**GEN, "speaker": "x", "nextQuestion": "這個時間沒說怎麼量,我會先定基準。"},
            ensure_ascii=False,
        )

    # 要帶 context:群面的陣容靠 group_interviewers 決定。不帶的話場上只有
    # 主考官,連用人主管都不存在,維度查表回空字串——那樣這一條測不到東西。
    # 第一版就是這樣寫的,拿掉模式判斷照樣綠。
    from app.schemas.interview import InterviewContext

    ctx = InterviewContext(
        custom_role="資料分析實習生", difficulty="中等",
        group_interviewers=3, group_size=4,
    )
    from app.pipeline.interview_live import dimension_speaker

    assert dimension_speaker("group", "取捨", 3, 4) == "用人主管", (
        "fixture 要讓那一位真的在場,否則擋不擋都一樣"
    )

    r = await next_turn(
        mode="group", answer=ANSWER, follow_up_idx=1, asked_questions=["資料分析"],
        spoken_by=["用人主管", "技術主管", "HR 主管", "AI-強勢", "AI-邏輯", "AI-親切"],
        question="q", fallback=["備援題一"], llm=llm, context=ctx,
        decision=TurnDecision(FOLLOW_UP, gap="沒有取捨", dimension="取捨"),
    )
    assert seen["dispatched"], "群面該照舊問模型,不可以被維度攔下來"
    assert r.speaker == "AI-邏輯", "競爭者要選得到"
    assert not any("派給負責的" in n for n in r.notices)


async def test_the_owner_lookup_itself_still_works_for_group():
    """`dimension_speaker` 本身不分模式,擋在呼叫端。

    分開測是因為那個查表函式在群面也回得出主管的名字——
    如果有人把模式判斷從呼叫端搬進函式裡,這一條會提醒他那不是同一件事:
    查表是查表,要不要用是政策。
    """
    from app.pipeline.interview_live import dimension_speaker

    assert dimension_speaker("group", "取捨", 3, 4) == "用人主管"
    assert dimension_speaker("panel", "取捨") == "用人主管"
