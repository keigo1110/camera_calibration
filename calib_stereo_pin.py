import cv2
import numpy as np
import glob
import os

# ====================================================
# 設定パラメータ（ステレオカメラキャリブレーション用）
# ====================================================
CONFIG = {
    "image_dir": "calib_images_stereo",           # キャリブレーション画像のディレクトリ
    "left_image_format": "left_*.png",       # 左カメラ画像のファイル名パターン
    "right_image_format": "right_*.png",     # 右カメラ画像のファイル名パターン
    "checkerboard_dims": (9, 6),             # チェスボードの内部角点数 (横, 縦)
    "square_size": 28.0,                     # チェスボード１マスの実寸（例：28.0 mm）
    "subpix_window_size": (3, 3),            # サブピクセル補正時の探索窓サイズ
    "subpix_zero_zone": (-1, -1),            # サブピクセル補正のゼロゾーン（-1なら自動）
    "subpix_criteria": (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                        30, 1e-6),         # サブピクセル補正の収束基準
}

# ====================================================
# 3Dオブジェクトポイントの作成（ステレオカメラ用）
# ====================================================
# チェスボード上の各内部コーナーの実世界座標（z=0 の平面上）
checkerboard_dims = CONFIG["checkerboard_dims"]
square_size = CONFIG["square_size"]

num_points = checkerboard_dims[0] * checkerboard_dims[1]
objp = np.zeros((num_points, 3), np.float32)
objp[:, :2] = np.mgrid[0:checkerboard_dims[0], 0:checkerboard_dims[1]].T.reshape(-1, 2)
objp *= square_size

# オブジェクトポイント（3D座標）と画像ポイント（2D座標）のリスト
objpoints      = []  # 各画像ペアで共通の実世界上の座標
imgpoints_left = []  # 左画像の検出された角点
imgpoints_right= []  # 右画像の検出された角点

# ====================================================
# キャリブレーション画像の読み込み
# ====================================================
left_image_paths  = glob.glob(os.path.join(CONFIG["image_dir"], CONFIG["left_image_format"]))
right_image_paths = glob.glob(os.path.join(CONFIG["image_dir"], CONFIG["right_image_format"]))

left_image_paths.sort()
right_image_paths.sort()

if len(left_image_paths) == 0 or len(right_image_paths) == 0:
    print("指定されたディレクトリにキャリブレーション画像が見つかりません。")
    exit()

if len(left_image_paths) != len(right_image_paths):
    print("左と右の画像枚数が一致していません。")
    exit()

print(f"検出された画像ペア: {len(left_image_paths)} 組")

img_shape = None
# 画像ペアごとにチェスボード検出を実施
for left_path, right_path in zip(left_image_paths, right_image_paths):
    img_left  = cv2.imread(left_path)
    img_right = cv2.imread(right_path)

    if img_left is None or img_right is None:
        print("画像の読み込みに失敗:", left_path, right_path)
        continue

    gray_left  = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

    if img_shape is None:
        img_shape = gray_left.shape[::-1]  # (width, height)

    # 左右それぞれでチェスボードのコーナー検出
    ret_left, corners_left   = cv2.findChessboardCorners(gray_left, checkerboard_dims,
                                                         cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
    ret_right, corners_right = cv2.findChessboardCorners(gray_right, checkerboard_dims,
                                                         cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)

    if ret_left and ret_right:
        # サブピクセル精度で角点を補正
        cv2.cornerSubPix(gray_left, corners_left, CONFIG["subpix_window_size"],
                         CONFIG["subpix_zero_zone"], CONFIG["subpix_criteria"])
        cv2.cornerSubPix(gray_right, corners_right, CONFIG["subpix_window_size"],
                         CONFIG["subpix_zero_zone"], CONFIG["subpix_criteria"])

        objpoints.append(objp)
        imgpoints_left.append(corners_left)
        imgpoints_right.append(corners_right)
        print(f"チェスボード検出成功: {left_path} と {right_path}")
    else:
        print(f"チェスボード検出失敗: {left_path} または {right_path}")

if len(objpoints) < 5:
    print("有効なキャリブレーション画像が十分ではありません。（最低5組以上必要）")
    exit()

# ====================================================
# 各カメラの個別キャリブレーションの実行
# ====================================================
# 左カメラ
ret_left, mtx_left, dist_left, rvecs_left, tvecs_left = cv2.calibrateCamera(
    objpoints, imgpoints_left, img_shape, None, None
)
# 右カメラ
ret_right, mtx_right, dist_right, rvecs_right, tvecs_right = cv2.calibrateCamera(
    objpoints, imgpoints_right, img_shape, None, None
)

print("\n=== 左カメラキャリブレーション結果 ===")
print(f"再投影誤差: {ret_left}")
print("カメラ行列:")
print(mtx_left)
print("歪み係数:")
print(dist_left)

print("\n=== 右カメラキャリブレーション結果 ===")
print(f"再投影誤差: {ret_right}")
print("カメラ行列:")
print(mtx_right)
print("歪み係数:")
print(dist_right)

# ====================================================
# ステレオキャリブレーションの実行
# ====================================================
# 内部パラメータは固定して、左右カメラ間の回転 R と平行移動 T を推定
flags = cv2.CALIB_FIX_INTRINSIC
criteria_stereo = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
ret_stereo, mtx_left, dist_left, mtx_right, dist_right, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpoints_left, imgpoints_right,
    mtx_left, dist_left, mtx_right, dist_right,
    img_shape, criteria=criteria_stereo, flags=flags
)

print("\n=== ステレオキャリブレーション結果 ===")
print(f"再投影誤差: {ret_stereo}")
print("回転行列 R:")
print(R)
print("平行移動ベクトル T:")
print(T)
print("エッセンシャル行列 E:")
print(E)
print("ファンダメンタル行列 F:")
print(F)

# ====================================================
# ステレオ補正（Rectification）の計算
# ====================================================
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
    mtx_left, dist_left, mtx_right, dist_right, img_shape, R, T, alpha=0
)

left_map1, left_map2 = cv2.initUndistortRectifyMap(
    mtx_left, dist_left, R1, P1, img_shape, cv2.CV_16SC2)
right_map1, right_map2 = cv2.initUndistortRectifyMap(
    mtx_right, dist_right, R2, P2, img_shape, cv2.CV_16SC2)

# ====================================================
# オプション：補正結果の確認（最初の画像ペアを使用）
# ====================================================
img_left  = cv2.imread(left_image_paths[0])
img_right = cv2.imread(right_image_paths[0])
rectified_left  = cv2.remap(img_left, left_map1, left_map2, cv2.INTER_LINEAR)
rectified_right = cv2.remap(img_right, right_map1, right_map2, cv2.INTER_LINEAR)

cv2.imshow("Original Left", img_left)
cv2.imshow("Rectified Left", rectified_left)
cv2.imshow("Original Right", img_right)
cv2.imshow("Rectified Right", rectified_right)
print("\nウィンドウ上で補正結果を確認できます。任意のキーを押すと終了します。")
cv2.waitKey(0)
cv2.destroyAllWindows()
