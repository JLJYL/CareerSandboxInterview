"""決定這一輪要做什麼之後,告訴生成端的那一句話。

【這個檔剩下什麼】
原本這裡還有一整套判準文字:把 `AcceptRule` 寫成 prompt、列出 gap 的封閉集合、
要求模型回 `{"why", "action", "gap"}`。2d 量完之後那些全部刪掉了。

刪的依據:

    1d(條件寫進 prompt)   三種難度的追問率都是 0.500,而且 19/180 次降級
    2d(程式用事實算)      0.143 / 0.429 / 0.857,一致率 1.00,0/630 次降級

「規則寫兩次沒生效,就不要寫第三次」——寫了兩次,量了兩次,所以拿掉。
判斷現在在 `AcceptRule.accepts()`,缺口在 `AcceptRule.first_gap()`,
事實由 `app/pipeline/star_read.py` 供應。

要翻舊版對照的話在 git 裡:刪掉它的那個 commit 之前。

【CLARIFY 是我們加的】
Panfilova et al.(2026)的 action space 只有 FOLLOW_UP 與 NEXT_MAIN。
他們的受訪者是打字,沒有語音辨識;我們的逐字稿會出現
Angular 轉成 Andrew 這類錯誤,所以需要第三個動作。
論文與簡報要標成延伸,不能寫成論文原有的。
"""

from __future__ import annotations

DECISION_ACTIONS: tuple[str, ...] = ("NEXT", "FOLLOW_UP", "CLARIFY")


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
"""兩種引擎下「什麼才算辨識轉錯」的判準。

**目前沒有呼叫端。** 判斷 CLARIFY 需要問模型(程式算不出「這個詞像是轉錯」),
而那一次窄問法還沒做,所以這兩段文字現在是備著的。

留著而不是刪掉是因為它記的是兩種引擎的差異——雲端轉錄不漏字所以破碎
通常是猶豫、裝置端會掉字所以看不懂要先假設是辨識問題。那個區別是實測來的,
重寫一次的成本遠高於留著這三十行。

接上去的做法:照 `app/prompts/star_read.py` 的格式問一次窄問題
(「有沒有哪個詞明顯不合情境」加抄原文),抄不出原文就不採用——
yes/no 加抄原文對窄判斷是準的(量化 60/60),對寬判斷不準(任務 0.29)。
"""


ACTION_INSTRUCTION: dict[str, str] = {
    "NEXT": "換一個新的主問題。不要接著他剛剛那段追問。",
    "FOLLOW_UP": "接著他剛剛那段追問,不要換題。",
    "CLARIFY": (
        "你沒聽清楚他說的某個地方,請他把那裡再講一次。"
        "不要換題,也不要追問別的。不要為系統的問題道歉。"
    ),
}
"""決定做完之後,告訴生成端這一輪要做什麼。

【為什麼只加一行,不改 system prompt】
生成端那支 prompt 是六輪實測調出來的,動它的風險遠大於收益。
決定是「這一輪做什麼」,屬於這一輪的輸入,所以放在 user 訊息裡。

【為什麼一定要講】
不講的話,決定說 NEXT 而模型生出追問,`is_follow_up` 就會跟實際內容不符,
然後重複偵測會對一句追問做比對——那正是這個欄位最初被誤判時出的事。
"""

_missing_instruction = set(DECISION_ACTIONS) - set(ACTION_INSTRUCTION)
if _missing_instruction:
    raise RuntimeError(
        f"這些動作沒有告訴生成端要做什麼:{sorted(_missing_instruction)}。"
        "少一個的話那個動作會生出跟決定不符的問句,而 is_follow_up 跟著錯。"
    )
