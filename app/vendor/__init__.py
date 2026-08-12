"""從 CareerSandboxModule 複製過來的元件。

【這個目錄的規則】
1. 這裡的檔案是複本,禁止修改。
2. 面試側需要的任何調整,寫在 app/pipeline/ 的擴充層,不要動複本。
3. 每個檔案的來源與 commit hash 記在 README.md。
4. tests/test_vendor_integrity.py 會檢查檔案有沒有被動過。

【為什麼要這樣】
本 repo 與 CareerSandboxModule 獨立部署,共用元件只能用複製。
複本一旦被修改,兩個 repo 就分岔:面試側的改善 C1 拿不到,
C1 的修正面試側也不知道,而且第四週想合併時無從 diff。

把複本鎖住、擴充另外寫,可以讓分岔只發生在一個明確的地方。
"""
