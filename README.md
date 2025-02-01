# camera_calibration
毎回カメラキャリブレーションの準備をするのに時間かかるから自分用に

## 準備
### python環境構築

### クローン
```bash
git clone https://github.com/keigo1110/camera_calibration.git
cd camera_calibration
```

### 必要なライブラリをインストール
```bash
pip install -r requirements.txt
```

### チェスボードの準備
`checkerboard.png`という名前のチェスボード画像を使用する。この画像を印刷して、キャリブレーション時に撮影用の対象物として使用する。

## キャリブレーション用の画像を撮影
- 単眼カメラの場合
```bash
python capt_pin.py
```
- 魚眼レンズの場合
```bash
python capt_fish.py
```

## キャリブレーション実行
- 単眼カメラの場合
```bash
python calib_pin.py
```
- 魚眼レンズの場合
```bash
python calib_fish.py
```

# Camera Calibration

A repository for quickly setting up camera calibration to save preparation time.

## Setup
### Setting Up the Python Environment

### Clone the Repository
```bash
git clone https://github.com/keigo1110/camera_calibration.git
cd camera_calibration
```

### Install Required Libraries

```bash
pip install -r requirements.txt
```

### Preparing the Checkerboard
A checkerboard image named `checkerboard.png` should be used. Print this image to use it as the target during calibration.

## Capturing Calibration Images

- **For a monocular camera:**
  ```bash
  python capt_pin.py
  ```
- **For a fisheye lens:**
  ```bash
  python capt_fish.py
  ```

## Running Calibration

- **For a monocular camera:**
  ```bash
  python calib_pin.py
  ```
- **For a fisheye lens:**
  ```bash
  python calib_fish.py
  ```