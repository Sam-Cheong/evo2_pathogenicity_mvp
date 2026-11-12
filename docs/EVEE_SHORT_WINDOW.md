# EVEE 短視窗探針試驗

這個流程用 NVIDIA 託管 Evo 2 7B 的 `blocks.27` 擷取公開 ClinVar SNV 的特徵，再用本機訓練 covariance probe 和均值基線。資料沿用倉庫的 GRCh38 FASTA、ClinVar CSV 及按基因分組的切分。

## 方法

- 在原有的 train、val、test 中固定抽取 24、8、8 筆。每個分區內兩類各半，每個相連基因群組至多抽一筆。候選樣本須通過兩個方向的 GRCh38 REF 和完整視窗檢查。
- 預設每個方向各取 256 bp，含 192 bp 上游及從變異位置開始的 64 bp。反向使用相對於該方向的側翼座標，再取反向互補。每筆變異最多四次前向請求，分別取得正反向的 REF、ALT 表徵。
- 每個方向在候選位置上依 REF、ALT 的 cosine divergence 選出 32 個位置，快取帶符號的 `ALT−REF` 差值與位置索引。均值基線使用全部 64 個候選位置的平均差值。
- 協方差探針將兩個方向的選定位置沿位置軸合併，用兩個可學習的 64 維線性投影計算交叉 Gram 矩陣，除以位置數，再用線性頭分類。預設使用 AdamW 與一個 epoch。均值基線拼接兩個方向的均值成 8,192 維，在訓練集擬合標準化和 L2 logistic regression（`C=1`）。兩者都以同一批保留基因的變異評估。

此試驗採用短視窗、較少保留位置、現有 ClinVar 版本與單次基因保留切分。使用本次資料重新訓練探針。40 筆平衡樣本的測試結果只用於檢查流程與觀察模型行為。原方法見 [EVEE Figure 1 實驗協議](https://github.com/goodfire-ai/evee-manuscript#figure-1-benchmark-protocol)。

## 在倉庫根目錄執行

開啟 [EVEE_short_window_experiment.ipynb](../EVEE_short_window_experiment.ipynb)，選擇專案 `.venv` 的 Python kernel，執行全部儲存格。Notebook 預設驗證並重用既有特徵，再將新訓練結果寫入 `outputs/evee_short_window_notebook_20261111`。若要在 notebook 中重新擷取，將第一格的 `FETCH_FROM_NVIDIA` 改為 `True`。

```powershell
.\.venv\Scripts\python.exe -m notebook EVEE_short_window_experiment.ipynb
```

使用既有虛擬環境及專案 `.env` 中的 NVIDIA 設定。`prepare` 建立固定的樣本清單。`extract` 對預設樣本最多發出 160 次前向請求（重試另計），中斷後可用同一輸出目錄續跑。`--workers` 控制並行請求數，預設為 1。`train` 讀取快取並在本機訓練。

```powershell
$run = 'outputs/evee_short_window_20261111'
.\.venv\Scripts\python.exe -m evo2_mvp.evee_short_window prepare --output $run
.\.venv\Scripts\python.exe -m evo2_mvp.evee_short_window extract --output $run --workers 4
.\.venv\Scripts\python.exe -m evo2_mvp.evee_short_window train --output $run
```

`cohort.json` 記錄樣本與資料雜湊。`features/` 保存選定位置、差值、均值及擷取設定。`probe/` 保存權重、預測和分區指標。`run_summary.json` 包含本次的保留基因測試指標。

## 2026-11-11 試跑

使用 2026-11-01 ClinVar 資料。40 筆變異的 80 個方向都完成擷取，共記錄 160 個不同的 NVIDIA 請求 ID。壓縮後的特徵檔合計 19.56 MiB。兩個模型在本機 CPU 完成訓練，從保存的權重和特徵重新計算可重現全部 40 筆輸出。

| 探針 | 測試 AUROC | 測試 AUPRC |
|---|---:|---:|
| 協方差 | 0.5000 | 0.7321 |
| 均值 | 0.6875 | 0.7929 |

測試集含 4 筆 P/LP 和 4 筆 B/LB。這個結果驗證了擷取、快取、訓練和測試流程，樣本數不足以判斷兩種探針在較大資料集的優劣。

## 固定 7B 的學習曲線

延伸隊列沿用 24 筆訓練和 8 筆驗證特徵，增加 16 筆訓練及 24 筆測試變異。24 個測試基因群組來自原有歷史留出分區，排除先前 7B 試跑的測試基因。各分區內兩類各半，每個基因群組一筆。新舊特徵使用相同的 7B `blocks.27` 擷取設定，服務的權重版本由 NVIDIA 管理。

兩種探針都讀取 256 bp 視窗的同一組雙向 top-32 差值。每個種子將訓練樣本按類別抽成嵌套的 12／24／40 筆，固定驗證與測試資料。均值使用固定的 L2 logistic，協方差固定 16 維投影、AdamW 及 60 次更新，學習率 0.001。報告三個種子的平均 AUROC，以及在同一批測試基因上配對重抽 2,000 次的 AUROC 差值區間。驗證與測試分數不參與超參數選擇。

```powershell
$run = 'outputs/evee_learning_curve_20261111'
.\.venv\Scripts\python.exe -m evo2_mvp.evee_learning_curve prepare --output $run
.\.venv\Scripts\python.exe -m evo2_mvp.evee_short_window extract --output $run --workers 4
.\.venv\Scripts\python.exe -m evo2_mvp.evee_learning_curve evaluate --output $run
```

`learning_curve.csv` 記錄各訓練規模的比較，`learning_curve.json` 另存種子、逐筆預測和 40 筆相對 12 筆的增益變化，`learning_curve.png` 顯示 AUROC 與配對區間。

## 2,000 筆 SNV 訓練

全量試驗沿用原有的 1,200／400／400 基因互斥切分，使用同一個 7B `blocks.27`、256 bp 視窗與雙向 top-32 特徵。協方差採 16 維投影、batch 8、最多 8 個 epoch，以驗證集 BCE 選擇權重。均值基線讀取相同位置。測試集沿用倉庫既有歷史留出集。

```powershell
.\.venv\Scripts\python.exe scripts/run_evee_full.py prepare
.\.venv\Scripts\python.exe -u scripts/run_evee_full.py run
```

輸出預設為 `outputs/evee_full_2000_20261111`。`job_status.json` 記錄目前階段，`features/extraction_status.json` 記錄當次擷取進度。擷取完成後會自動訓練與評估。若工作因網路或程序中斷而停止，可在確認原程序已退出後，重跑上面的 `run` 命令以沿用已完成的特徵。

## 短 indel 評估

使用現有 SNV 分類頭的權重，評估 GRCh38 上的簡單插入與缺失。每個方向各建立 256 bp 的 REF 和 ALT 視窗，從編輯之後配對相同參考位置，跳過插入或缺失的鹼基，再保留差異最大的 32 個位置。

預設從本地公開 ClinVar VCF 選取 32 筆至少兩星、長 1–8 bp 的 indel，依插入／缺失及致病／良性平衡取樣，排除原 SNV 隊列的基因。執行 `run` 會準備隊列、透過 NVIDIA 擷取表徵，並在本機用凍結的協方差與均值池化分類頭評估。

```powershell
.\.venv\Scripts\python.exe -m evo2_mvp.evee_indel_evaluation run --output outputs/evee_indel_holdout_20261112 --workers 4
```

擷取中斷時，可用 `extract` 和同一輸出目錄接續，再用 `evaluate` 讀取快取並重新計算。`run_summary.json` 記錄整體與按 indel 類型計算的 AUROC 和 AP。`predictions.csv` 保存逐筆結果。
