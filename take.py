import cv2
import numpy as np
import os

# チェスボードの内角の数（縦・横のマス数 - 1）
CHECKERBOARD = (6, 9)

# 出力フォルダの作成
output_dir = 'images'
os.makedirs(output_dir, exist_ok=True)

# カメラの設定（カメラIDを0に設定、外付けカメラなら1や他のIDを使う場合も）
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("カメラを開くことができませんでした")
    exit()

# 画像の保存カウント
img_counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("フレームを取得できませんでした")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    # チェスボードが見つかった場合
    if ret:
        cv2.drawChessboardCorners(frame, CHECKERBOARD, corners, ret)
        # チェスボードが検出された場合に画像を保存
        img_name = os.path.join(output_dir, f"image_{img_counter:03d}.jpg")
        cv2.imwrite(img_name, frame)
        print(f"{img_name} を保存しました")
        img_counter += 1

    cv2.imshow('Camera', frame)

    # キー入力待ち（qを押すと終了）
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# リソースの解放
cap.release()
cv2.destroyAllWindows()
