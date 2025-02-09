import cv2
import numpy as np
import glob
import os

# ====================================================
# 設定パラメータ（魚眼用ステレオキャリブレーション）
# ====================================================
CONFIG = {
    "image_dir": "calib_images_stereo",       # キャリブレーション画像のディレクトリ
    "left_image_format": "left_*.png",          # 左カメラ画像のファイル名パターン
    "right_image_format": "right_*.png",        # 右カメラ画像のファイル名パターン
    "checkerboard_dims": (9, 6),                # チェスボードの内部角点数 (横, 縦)
    "square_size": 28.0,                        # チェスボード１マスの実寸（例：28.0 mm）
    "subpix_window_size": (3, 3),               # サブピクセル補正時の探索窓サイズ
    "subpix_zero_zone": (-1, -1),               # サブピクセル補正のゼロゾーン（-1なら自動）
    "subpix_criteria": (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                        30, 1e-6),            # サブピクセル補正の収束基準
}

# ====================================================
# 3Dオブジェクトポイントの作成（魚眼用・ステレオカメラ）
# ====================================================
checkerboard_dims = CONFIG["checkerboard_dims"]
square_size = CONFIG["square_size"]

num_points = checkerboard_dims[0] * checkerboard_dims[1]
objp = np.zeros((num_points, 3), np.float32)
objp[:, :2] = np.mgrid[0:checkerboard_dims[0], 0:checkerboard_dims[1]].T.reshape(-1, 2)
objp *= square_size
# fish-eye calibrate では各画像ごとの object points は (N,1,3) の形状である必要がある
objp = objp.reshape(-1, 1, 3)

objpoints      = []  # 各画像ペアで共通の実世界座標
imgpoints_left = []  # 左画像の検出された角点 (N,1,2)
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

    ret_left, corners_left = cv2.findChessboardCorners(
        gray_left, checkerboard_dims,
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
    ret_right, corners_right = cv2.findChessboardCorners(
        gray_right, checkerboard_dims,
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)

    if ret_left and ret_right:
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
# 各カメラの個別キャリブレーション（魚眼モデル）の実行
# ====================================================
K_left = np.eye(3, dtype=np.float64)
D_left = np.zeros((4, 1), dtype=np.float64)
K_right = np.eye(3, dtype=np.float64)
D_right = np.zeros((4, 1), dtype=np.float64)

fisheye_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_CHECK_COND | cv2.fisheye.CALIB_FIX_SKEW

rvecs_left, tvecs_left = [], []
ret_left, K_left, D_left, rvecs_left, tvecs_left = cv2.fisheye.calibrate(
    objpoints, imgpoints_left, img_shape, K_left, D_left, rvecs_left, tvecs_left,
    fisheye_flags,
    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
)

rvecs_right, tvecs_right = [], []
ret_right, K_right, D_right, rvecs_right, tvecs_right = cv2.fisheye.calibrate(
    objpoints, imgpoints_right, img_shape, K_right, D_right, rvecs_right, tvecs_right,
    fisheye_flags,
    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
)

print("\n=== 左カメラ（魚眼）キャリブレーション結果 ===")
print(f"再投影誤差: {ret_left}")
print("カメラ行列 (左):\n", K_left)
print("歪み係数 (左):\n", D_left)

print("\n=== 右カメラ（魚眼）キャリブレーション結果 ===")
print(f"再投影誤差: {ret_right}")
print("カメラ行列 (右):\n", K_right)
print("歪み係数 (右):\n", D_right)

# ====================================================
# ステレオキャリブレーション（魚眼モデル）の実行
# ====================================================
R = np.zeros((3, 3), dtype=np.float64)
T = np.zeros((3, 1), dtype=np.float64)

ret_stereo, K_left, D_left, K_right, D_right, R, T = cv2.fisheye.stereoCalibrate(
    objpoints, imgpoints_left, imgpoints_right,
    K_left, D_left, K_right, D_right,
    img_shape, R, T,
    fisheye_flags,
    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
)

print("\n=== 魚眼ステレオキャリブレーション結果 ===")
print(f"再投影誤差: {ret_stereo}")
print("回転行列 R:\n", R)
print("平行移動ベクトル T:\n", T)

# ====================================================
# ステレオ補正（Rectification）の計算（魚眼モデル）
# ====================================================
R1 = np.zeros((3, 3), dtype=np.float64)
R2 = np.zeros((3, 3), dtype=np.float64)
P1 = np.zeros((3, 4), dtype=np.float64)
P2 = np.zeros((3, 4), dtype=np.float64)
Q  = np.zeros((4, 4), dtype=np.float64)

flags = 0  # または cv2.fisheye.CALIB_ZERO_DISPARITY
balance = 0.0
new_size = img_shape  # 新しい画像サイズ。元画像サイズをそのまま利用

cv2.fisheye.stereoRectify(K_left, D_left, K_right, D_right, img_shape, R, T,
                          R1, R2, P1, P2, Q, flags, balance, new_size)

left_map1, left_map2 = cv2.fisheye.initUndistortRectifyMap(
    K_left, D_left, R1, P1, img_shape, cv2.CV_16SC2)
right_map1, right_map2 = cv2.fisheye.initUndistortRectifyMap(
    K_right, D_right, R2, P2, img_shape, cv2.CV_16SC2)

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