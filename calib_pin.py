import cv2
import numpy as np
import glob
import os

# ====================================================
# 設定パラメータ（単眼カメラキャリブレーション）
# ====================================================
CONFIG = {
    "image_dir": "calib_images",         # 撮影済み画像のディレクトリ
    "image_format": "*.jpg",             # 対象画像のファイル形式
    "checkerboard_dims": (9, 6),         # チェスボードの内部角点数 (横, 縦)
    "square_size": 28.0,                 # チェスボード１マスの実寸（例：28.0 mm）
    "subpix_window_size": (3, 3),        # サブピクセル補正時の探索窓サイズ
    "subpix_zero_zone": (-1, -1),        # サブピクセル補正のゼロゾーン（-1なら自動）
    "subpix_criteria": (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                        30, 1e-6),       # サブピクセル補正の収束基準
}

# ====================================================
# 3Dオブジェクトポイントの作成（単眼カメラ用）
# ====================================================
# チェスボード上の各内部コーナーの実世界座標（z=0 の平面上）
# （各マスのサイズは square_size でスケーリング）
num_points = CONFIG["checkerboard_dims"][0] * CONFIG["checkerboard_dims"][1]
objp = np.zeros((num_points, 3), np.float32)
objp[:, :2] = np.mgrid[0:CONFIG["checkerboard_dims"][0],
                       0:CONFIG["checkerboard_dims"][1]].T.reshape(-1, 2)
objp *= CONFIG["square_size"]

# オブジェクトポイント（3D座標）と画像ポイント（2D座標）のリスト
objpoints = []  # 各画像ごとの実世界上の点（shape: (N, 3)）
imgpoints = []  # 各画像ごとの検出されたコーナー（shape: (N, 1, 2) でも可）

# ====================================================
# キャリブレーション画像の読み込み
# ====================================================
image_paths = glob.glob(os.path.join(CONFIG["image_dir"], CONFIG["image_format"]))
if len(image_paths) == 0:
    print("指定されたディレクトリにキャリブレーション画像が見つかりません。")
    exit()

print(f"検出されたキャリブレーション画像: {len(image_paths)} 枚")

img_shape = None
for image_path in image_paths:
    img = cv2.imread(image_path)
    if img is None:
        print("画像の読み込みに失敗:", image_path)
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img_shape is None:
        img_shape = gray.shape[::-1]  # (width, height)

    # チェスボードコーナーの検出
    ret, corners = cv2.findChessboardCorners(gray, CONFIG["checkerboard_dims"],
                                             cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
    if ret:
        # サブピクセル補正
        cv2.cornerSubPix(gray, corners, CONFIG["subpix_window_size"],
                         CONFIG["subpix_zero_zone"], CONFIG["subpix_criteria"])
        objpoints.append(objp)
        imgpoints.append(corners)
        print(f"チェスボード検出成功: {image_path}")
    else:
        print(f"チェスボード検出失敗: {image_path}")

if len(objpoints) < 5:
    print("有効なキャリブレーション画像が十分ではありません。（最低5枚以上必要）")
    exit()

# ====================================================
# 単眼カメラキャリブレーションの実行
# ====================================================
# cv2.calibrateCamera() によりカメラ行列、歪み係数、回転・並進ベクトルを推定
ret, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, img_shape, None, None
)

print("\n=== キャリブレーション結果 ===")
print(f"再投影誤差: {ret}")
print("\nカメラ行列:")
print(cameraMatrix)
print("\n歪み係数:")
print(distCoeffs)

# ====================================================
# 再投影誤差の計算
# ====================================================
mean_error = 0
for i in range(len(objpoints)):
    # 推定パラメータを用いて、3D点を画像平面に再投影
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], cameraMatrix, distCoeffs)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    mean_error += error
mean_error /= len(objpoints)
print(f"\n平均再投影誤差: {mean_error}")

# ====================================================
# オプション：歪み補正結果の確認（最初の画像を使用）
# ====================================================
test_img = cv2.imread(image_paths[0])
if test_img is not None:
    undistorted_img = cv2.undistort(test_img, cameraMatrix, distCoeffs, None, cameraMatrix)
    cv2.imshow("Original Image", test_img)
    cv2.imshow("Undistorted Image", undistorted_img)
    print("\nウィンドウ上で歪み補正結果を確認できます。任意のキーを押すと終了します。")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("テスト画像の読み込みに失敗しました。")