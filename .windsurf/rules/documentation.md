---
trigger: always_on
---

# Documentation Rules (Simple Version)

> 重點：  
>- overview README 描述 **workflow、function flow、project 架構**  
> - subproject README 文件主要描述：**workflow、function flow、data flow、I/O、project 架構**  
> - 理論與研究設計寫在 `PROJECT_DRAFT.md`，實作寫在各種 README / QUICK_START

---

## 1. Doc Types

1. **Root `OVERVIEW_README.md`**  
2. **Per-project `README.md`**（例如：`projects/assr/README.md`）  
3. **Per-project `QUICK_START.md`**（或 `IMPLEMENTATION_SUMMARY.md`）  
4. **Per-project `PROJECT_DRAFT.md` / `PLAN.md`**  
5. **資料夾內的 `DATA_README.md`**（描述該資料夾的 data）

嚴格規則：
如果有類似內容的文件存在，**優先更新，不要創新名詞**（例如不要再創 `notes.md`）。

---

## 2. `OVERVIEW_README.md`（整體架構 & 高層實作）

目的：描述「整個 repo 在做什麼」與「主要區塊長怎樣」，而不是細節指令。

內容重點：

- 一句話說明這個 repo 在實作什麼
- 列出主要資料夾與角色，例如：
  - `nmm_core/` – 共用 Wendling / PSD / fitting 工具
- 做一個簡單的 **high-level implementation / architecture**：
  - 哪些檔案負責 model、哪些負責 PSD、哪些負責 fitting
- 簡單畫出各 project 的高層 **workflow**（文字 flow 就好）  
  例如：`raw data → preprocess → PSD → model fitting`
- 結尾：指向各 project 的 `README.md` 作為更多細節入口  

Dependency / 安裝在這裡不用詳細描述，只要一行類似：

> Dependencies can be installed as needed (e.g. via pip or conda).

---

## 3. Per-project `README.md`（實作架構 & flow）

目的：讓人一眼看懂「這個 project 的實作長怎樣、檔案放哪裡、資料怎麼流動」。

內容重點：

1. **Project 概述**
   - 1–2 段話：這個 project 在分析什麼、大概做什麼

2. **Module / function map**
   - 列出主要檔案與其功能，例如：
     - `scripts/preprocess_assr.py` – load raw EEG, filter, epoch

3. **Data flow（I/O）**
   - 清楚寫：
     - Input 資料夾與格式（`data/raw/`, `.edf`…）
     - 中間產物（`data/epochs/`, `data/psd/`…）
     - Output 結果（`results/params/`, `results/plots/`…）

4. **Workflow summary**
   - 用簡單箭頭描述 script 之間的流程：
     - `preprocess -> compute_psd -> fit_model`
   - 具體指令不要放這裡，放到 `QUICK_START.md`

5. **Links**
   - 指向：
     - `QUICK_START.md`（如何實際執行）
     - `PROJECT_DRAFT.md`（理論與設計）

---

## 4. `QUICK_START.md`（實際 workflow / 指令）

目的：告訴使用者（和未來的我）**怎麼跑這個 project**。

內容重點：

1. **Main pipelines**
   - 列出主要流程步驟（preprocess / PSD / fit / plot）

2. **Commands**
   - 每個步驟給一個最常用的 command，例如：

     ```bash
     python projects/assr/scripts/preprocess_assr.py --subject S01
     ```

3. **重要參數**
   - 簡單說明常用 flags，例如 `--subject`, `--overwrite`, `--roi-channels`

4. **I/O recap**
   - 簡短再提醒這些 commands 的 input / output 路徑

**只要 pipeline / script 介面改到，讓原本的 command 不能用，必須更新 `QUICK_START.md`。**

---

## 5. `PROJECT_DRAFT.md`（理論與研究設計）

目的：放 **theory level / research design**，不是實作。

內容可以包含：

- 研究問題與假設
- 實驗 paradigm（例如 ASSR 的頻率、marker code、timing）
- 分析 plan（preprocess → epoch → PSD → fitting）
- 如何比較 patient vs control、哪些參數代表什麼

這裡 **不要放** function flow 或 script 說明，那些在 `README.md` / `QUICK_START.md`。

---

## 6. `DATA_README.md`（資料夾內 data 說明）

目的：描述某個 data 資料夾的內容與格式。

內容：

- 資料來源與型態（raw EEG / preprocessed / PSD / params）
- 檔名規則
- 檔案格式（`.edf`, `.npz`…）
- 基本欄位解釋（例如 channel, sampling rate）
- 若有專門 loader function，可以提到它的名稱

---

## 7. 從「單一 project」變成「多 project」時怎麼處理 README

情境：一開始只有一個 project，後來新增第二個 project。

1. **建立 `projects/` 結構**
   - 把原本的 code 移到 `projects/main/` 或你選的名字
   - 新的 project 放在 `projects/new_project/`

2. **搬移 README**
   - 原本 `README.md` 如果是在描述那個舊 project：
     - 把其中「舊 project 的詳細內容」移到 `projects/main/README.md`
     - 建立新的OVERVIEW_README
     - 說明：這個 repo 現在有多個 project
     - 列出 `projects/main/`、`projects/new_project/` 等
     - 簡要說明它們彼此的關係與共用核心

3. **更新連結**
   - 修正 README 中原本指向舊路徑的文字為新路徑

---

## 8. 文檔維護 Checklist（給 Windsurf）

每次改 code，請檢查：

1. 我有改變 **script flow / pipeline 結構** 嗎？  
   → 有：更新對應 project 的 `README.md`（flow）與 `QUICK_START.md`（指令）

2. 我有改變 **data flow / 輸入輸出位置** 嗎？  
   → 有：更新 project 的 `README.md` 的 Data flow 區塊，必要時更新 `DATA_README.md`

3. 我有改變 **研究設計 / 分析計畫** 嗎？  
   → 有：更新該 project 的 `PROJECT_DRAFT.md`

4. 沒有以上改動 → 一般情況下 **不需要** 更新文件。
