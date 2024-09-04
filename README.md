# camera_calibration
毎回準備するのに時間かかるから自分用に

## 準備
### python環境構築

### 必要なものインストール
```
pip install -r requirements.txt
```

### チェスボードの準備
`checkerboard.png`という名前のチェスボード画像を使用する。この画像を印刷して、キャリブレーション時に撮影用の対象物として使用する。

## キャリブレーション用の画像を撮影
動画で自動でキャプチャされる\
"q"キーで終了
```
python take.py
```
## キャリブレーション実行
`calib.npz`ができる
```
python carib.py
```
## 値の確認
```
python check.py
```
