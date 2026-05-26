# Head Counter

即時人頭計數器：webcam 抓人頭、跨線判定 IN/OUT、自動寫進本地資料庫，可隨時查每天人流。

---

## 快速使用

**要求**：Windows 10/11 + NVIDIA GPU

```powershell
git clone https://github.com/michael228866/yolov7.git
cd yolov7
.\setup.bat        # 第一次跑，自動裝所有東西（5-10 分鐘）
.\start.bat        # 之後每次雙擊這個
```

`setup.bat` 會自動：裝 Python 3.10、建 venv、裝 PyTorch + ultralytics + supervision、下載權重檔。沒有 Python 也不用先裝。

---

## 操作

### start.bat（計數）

雙擊 `start.bat` 開啟，第一張畫面**點兩個點**畫出計數線，按任意鍵開始。
之後每次有人跨線，左上 **TOTAL** 數字會 +1。

鍵盤：
- `q` 或 `Esc` → 結束
- `r` → 螢幕計數歸零（資料庫不受影響）

常用參數：
```powershell
.\start.bat --line v                          # 直接用畫面中間垂直線
.\start.bat --line h                          # 水平線
.\start.bat --line 100,400,1180,500           # 自訂線座標
.\start.bat --weights yolov8_head_nano.pt     # 換更快但準度低的模型
.\start.bat --source path\to\video.mp4        # 跑影片檔
.\start.bat --flip                            # IN/OUT 方向反過來
.\start.bat --conf-thres 0.5                  # 信心門檻調寬鬆（預設 0.6）
.\start.bat --save out.mp4                    # 同時錄影
```

### query.bat（看報表）

```powershell
.\query.bat                # 今天的 IN/OUT 統計 + 每小時長條圖
.\query.bat 2026-05-26     # 指定日期
.\query.bat --last 7       # 近 7 天趨勢
.\query.bat --sessions     # 列出每次開機紀錄
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

## 資料

所有跨線事件都寫進 `counter.db`（SQLite，跟 .bat 同資料夾）。

- **備份** → 複製 `counter.db` 走
- **重新計算** → 刪掉 `counter.db`，下次跑會自動重建
- **直接查 SQL** → 用 [DB Browser for SQLite](https://sqlitebrowser.org/) 開檔案
