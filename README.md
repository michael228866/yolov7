# Head Counter

Real-time **head-only** people counter — webcam in, IN/OUT line crossing
events out, every crossing logged into a local SQLite database for daily
reports.

| Component | Tech |
|---|---|
| Detection | YOLOv8 trained on CrowdHuman (head class only) — [Abcfsa/YOLOv8_head_detector](https://github.com/Abcfsa/YOLOv8_head_detector) |
| Tracking | Ultralytics built-in ByteTrack (`model.track(persist=True)`) |
| Line crossing | `supervision.LineZone` |
| Storage | SQLite (single file `counter.db`) |
| GPU | CUDA 11.8 (tested on GTX 1080 Ti) |

---

---

## 0. 給同事：git clone 就能跑

**前置**：Windows 10/11 + **NVIDIA GPU 驅動**。Python 不用先裝（setup.bat 會處理）。

```powershell
git clone https://github.com/michael228866/yolov7.git
cd yolov7
.\setup.bat        # 雙擊也可以，第一次跑大約 5-10 分鐘
.\start.bat        # 之後每次雙擊這個就好
```

`setup.bat` 自動做的事：
1. 沒 Python 3.10 → 用 winget 裝
2. 建虛擬環境 `.venv\`
3. 裝 PyTorch (CUDA 11.8) + ultralytics + supervision
4. 從 GitHub 下載 `yolov8_head_medium.pt` 跟 `yolov8_head_nano.pt`

| 雙擊 | 做什麼 |
|---|---|
| `setup.bat` | **僅第一次**：裝環境、下載權重 |
| `start.bat` | 啟動頭部計數器（webcam） |
| `query.bat` | 看今天 / 指定日期的人流報表 |

如果 winget 不能用（極舊的 Win 10）：去 [App Installer](https://apps.microsoft.com/detail/9NBLGGH4NNS1) 裝，或手動裝 [Python 3.10.11](https://www.python.org/downloads/release/python-31011/)（勾選 *Add python.exe to PATH*）。

如果同事完全不想裝任何東西，改用 `make_release.py` 打包出來的完整 `release/` 資料夾（見第 6 節）。

---

## 1. 開發機環境

需要 **conda** 跟 **NVIDIA GPU + 驅動**。

```powershell
# 建環境
conda create -n yolov7 python=3.10 pip -y
conda activate yolov7

# PyTorch with CUDA 11.8
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 `
    --index-url https://download.pytorch.org/whl/cu118

# 其他套件
pip install ultralytics supervision conda-pack
```

驗證 GPU：
```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 2. 下載權重

兩個檔，nano 快、medium 準（預設）：

```powershell
curl.exe -L -o yolov8_head_nano.pt   https://github.com/Abcfsa/YOLOv8_head_detector/raw/main/nano.pt
curl.exe -L -o yolov8_head_medium.pt https://github.com/Abcfsa/YOLOv8_head_detector/raw/main/medium.pt
```

---

## 3. 主程式：[count_inout_head.py](count_inout_head.py)

```powershell
# 預設 webcam 0 + 互動式畫線（第一張畫面點 2 個點）
python count_inout_head.py

# 垂直中線
python count_inout_head.py --line v

# 水平中線
python count_inout_head.py --line h

# 自訂線座標 x1,y1,x2,y2
python count_inout_head.py --line 100,400,1180,500

# 換 nano 模型（快）
python count_inout_head.py --weights yolov8_head_nano.pt

# 跑影片檔
python count_inout_head.py --source path\to\video.mp4 --line h

# 同時錄影輸出
python count_inout_head.py --line v --save out.mp4

# IN/OUT 方向反了
python count_inout_head.py --line v --flip
```

### CLI 旗標

| 旗標 | 預設 | 說明 |
|---|---|---|
| `--weights` | `yolov8_head_medium.pt` | 模型權重檔 |
| `--source` | `0` | webcam index / 影片路徑 / RTSP URL |
| `--img-size` | `640` | 推論解析度 |
| `--conf-thres` | `0.60` | 偵測信心門檻 (0-1) |
| `--iou-thres` | `0.45` | NMS IoU 門檻 |
| `--device` | `0` | GPU index 或 `cpu` |
| `--line` | `(空)` | `x1,y1,x2,y2` / `h` / `v` / 空 = 互動式畫線 |
| `--flip` | – | 對調 IN/OUT 方向 |
| `--anchor` | `center` | 跨線判定點：`center` / `bottom` / `top` / `corners` |
| `--save` | `(空)` | 輸出 mp4 路徑 |

### 鍵盤

- `q` / `Esc` → 結束
- `r` → 螢幕計數歸零（DB 紀錄不受影響）
- 滑鼠（互動式畫線）→ 點 2 個點定義線

### 畫面顯示

左上角只有一個大字 **TOTAL : N**（= IN 累計，不算 OUT）。
LineZone 線會畫出來但不顯示文字標籤。
每顆頭會有綠框 + `#track_id confidence` + 拖尾軌跡（30 frame）。

---

## 4. 資料庫：[db.py](db.py) + `counter.db`

每次跑 `count_inout_head.py` 會：
1. 在 `sessions` 表插入一筆 session
2. 每次跨線在 `events` 表插入一筆事件

### Schema

```sql
sessions(id, started_at, ended_at, source, weights, notes)
events  (id, ts, direction, track_id, session_id)
```

時間戳是**本機時間** ISO-8601，秒級精度。

### 直接查 DB

```powershell
# Python 一行印最近 20 筆事件
python --% -c "import sqlite3; [print(r) for r in sqlite3.connect('counter.db').execute('SELECT * FROM events ORDER BY id DESC LIMIT 20')]"
```

也可以用 [DB Browser for SQLite](https://sqlitebrowser.org/) 開 `counter.db` 用 GUI 查。

---

## 5. 報表：[query_daily.py](query_daily.py)

```powershell
# 今天總計 + 每小時長條圖
python query_daily.py

# 指定日期
python query_daily.py 2026-05-26

# 近 N 天趨勢
python query_daily.py --last 7

# 列出 session 紀錄
python query_daily.py --sessions
```

範例：
```
=== Daily report: 2026-05-26 ===
  IN     :     8
  OUT    :     5
  TOTAL  :     8   (IN only)
  NET    :    +3

   Hour     IN    OUT  Bar (IN)
  -----  -----  -----  ------------------------------
  14:00      3      1  ##############################
  15:00      5      4  ##################
```

---

## 6. 打包給別人用：[make_release.py](make_release.py)

```powershell
python make_release.py
```

會產出 `release/` 資料夾（約 6-7 GB），結構：
```
release/
├─ env/                      整個 conda 環境（解壓後可直接用）
├─ count_inout_head.py
├─ db.py
├─ query_daily.py
├─ yolov8_head_medium.pt
├─ yolov8_head_nano.pt
├─ start.bat                 雙擊啟動計數器
├─ query.bat                 雙擊看今天報表
└─ README.txt                接收者使用說明（中文）
```

### 給對方步驟

1. 把整個 `release/` 資料夾壓成 zip
2. 透過 USB / 雲端傳給對方
3. 對方解壓 → 雙擊 `start.bat`

對方需要的：
- Windows 10/11 64-bit
- NVIDIA GPU + 任何近期驅動（支援 CUDA 11.8 即可）
- 10 GB 空閒硬碟

對方**不需要**裝 Python / conda / 任何套件。

---

## 7. 檔案總覽

```
yolov7/
├─ count_inout_head.py     主程式
├─ db.py                   SQLite schema + 寫入 helper
├─ query_daily.py          CLI 報表工具
├─ make_release.py         打包腳本（給沒裝 Python 的同事用）
├─ requirements.txt        pip 套件清單（setup.bat 會用到）
├─ setup.bat               同事第一次跑的安裝腳本
├─ start.bat               啟動計數器
├─ query.bat               看報表
├─ counter.db              SQLite 資料庫（自動建立、不進 git）
├─ yolov8_head_medium.pt   權重（setup.bat 下載、不進 git）
├─ yolov8_head_nano.pt     權重（setup.bat 下載、不進 git）
├─ README.md
└─ .gitignore
```

---

## 8. 常見問題

**Q: 計數錯了**
A: 調 `--conf-thres`（高→嚴格，低→寬鬆），或改 `--anchor bottom` 用腳底判定。

**Q: 一直跳 #ID 大數字（例如 #87）**
A: 那是 ByteTrack 內部計數，不是「第 87 個人」。左上 **TOTAL** 才是真實計數。

**Q: 想砍掉 DB 重來**
A: 刪掉 `counter.db` 檔，下次跑會自動重建。

**Q: 換相機角度後，方向反了**
A: 加 `--flip`，或畫線時兩個點順序顛倒。

**Q: 一直跳 `Line zone counting skipped` 警告**
A: 已修，只在有 tracker_id 時才呼叫 LineZone。如果你的版本還有，請更新 `count_inout_head.py`。
