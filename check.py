import numpy as np

# 保存されたキャリブレーションパラメータの読み込み
data = np.load('calib.npz')
mtx = data['mtx']
dist = data['dist']
rvecs = data['rvecs']
tvecs = data['tvecs']

# カメラ行列、歪み係数、回転ベクトル、平行移動ベクトルの確認
print("カメラ行列 (mtx):\n", mtx)
print("\n歪み係数 (dist):\n", dist)
print("\n回転ベクトル (rvecs):\n", rvecs)
print("\n平行移動ベクトル (tvecs):\n", tvecs)

# キャリブレーションの品質を確認する
print("\nキャリブレーションが成功している場合、カメラ行列や歪み係数が適切な値であることを確認できます。")
