import cv2
import numpy as np
import glob
import os

# ====================================================
# 設定パラメータ（魚眼カメラキャリブレーション）
# ====================================================
CONFIG = {
    "image_dir": "calib_images",         # 撮影済み画像のディレクトリ
    "image_format": "*.png",             # 対象画像のファイル形式
    "checkerboard_dims": (9, 6),         # チェスボードの内部角点数 (横, 縦)
    "square_size": 28.0,                 # チェスボード１マスの実寸（例：28.0 mm）
    "subpix_window_size": (3, 3),        # サブピクセル補正時の探索窓サイズ
    "subpix_zero_zone": (-1, -1),        # サブピクセル補正のゼロゾーン（-1なら自動）
    "subpix_criteria": (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                        30, 1e-6),       # サブピクセル補正の収束基準
    "fisheye_flags": cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC +
                     cv2.fisheye.CALIB_CHECK_COND +
                     cv2.fisheye.CALIB_FIX_SKEW,  # 魚眼キャリブレーション用フラグ
    "calib_criteria": (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                       100, 1e-6),       # キャリブレーション処理の収束基準
}

# ====================================================
# 3Dオブジェクトポイントの作成
# ====================================================
# チェスボード上の各コーナーの実世界座標（z=0の平面上）
# 各画像に対して、shape (1, num_points, 3) の配列を用意する必要があります。
num_points = CONFIG["checkerboard_dims"][0] * CONFIG["checkerboard_dims"][1]
objp = np.zeros((1, num_points, 3), np.float32)
objp[0, :, :2] = np.mgrid[0:CONFIG["checkerboard_dims"][0],
                           0:CONFIG["checkerboard_dims"][1]].T.reshape(-1, 2)
objp *= CONFIG["square_size"]

# オブジェクトポイント（3D座標）と画像ポイント（2D座標）のリスト
objpoints = []  # 各画像ごとの objectPoints（shape: (1, num_points, 3)）
imgpoints = []  # 各画像ごとの検出されたコーナー（通常は shape: (num_points, 1, 2)）

# ====================================================
# キャリブレーション画像の読み込み
# ====================================================
image_paths = glob.glob(os.path.join(CONFIG["image_dir"], CONFIG["image_format"]))
if len(image_paths) == 0:
    print("指定されたディレクトリにキャリブレーション画像が見つかりません。")
    exit()

print(f"検出されたキャリブレーション画像: {len(image_paths)} 枚")

# ====================================================
# 各画像からチェスボードのコーナーを検出（サブピクセル補正付き）
# ====================================================
img_shape = None
for image_path in image_paths:
    img = cv2.imread(image_path)
    if img is None:
        print("画像の読み込みに失敗:", image_path)
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 画像サイズの取得（すべての画像が同じサイズであればOK）
    if img_shape is None:
        img_shape = gray.shape[::-1]  # (width, height)

    # チェスボードコーナーの検出
    ret, corners = cv2.findChessboardCorners(
        gray,
        CONFIG["checkerboard_dims"],
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    if ret:
        # サブピクセル補正を実施
        cv2.cornerSubPix(gray, corners,
                         CONFIG["subpix_window_size"],
                         CONFIG["subpix_zero_zone"],
                         CONFIG["subpix_criteria"])
        # objectPoints は各画像に対して objp のコピーをそのまま追加
        objpoints.append(objp.copy())
        imgpoints.append(corners)
        print(f"チェスボード検出成功: {image_path}")
    else:
        print(f"チェスボード検出失敗: {image_path}")

if len(objpoints) < 5:
    print("有効なキャリブレーション画像が十分ではありません。（最低5枚以上必要）")
    exit()

# ====================================================
# 魚眼カメラキャリブレーションの実行
# ====================================================
# 初期パラメータ（カメラ行列 K と歪み係数 D）の初期化
K = np.zeros((3, 3))
D = np.zeros((4, 1))
rvecs = []  # 各画像ごとの回転ベクトル
tvecs = []  # 各画像ごとの並進ベクトル

# キャリブレーション実行（魚眼用関数）
rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
    objpoints,
    imgpoints,
    img_shape,
    K,
    D,
    rvecs,
    tvecs,
    CONFIG["fisheye_flags"],
    CONFIG["calib_criteria"]
)

print("\n=== キャリブレーション結果 ===")
print(f"RMS再投影誤差: {rms}")
print("\nカメラ行列 (K):")
print(K)
print("\n歪み係数 (D):")
print(D)

# ====================================================
# 再投影誤差の計算（各画像ごとの誤差の平均）
# ====================================================
mean_error = 0
for i in range(len(objpoints)):
    # 推定パラメータを用いて、3D点群を再投影
    imgpoints2, _ = cv2.fisheye.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, D)
    # 両方の配列の形状を (N, 2) に統一
    ip1 = imgpoints[i].reshape(-1, 2)
    ip2 = imgpoints2.reshape(-1, 2)
    error = cv2.norm(ip1, ip2, cv2.NORM_L2) / len(ip2)
    mean_error += error
mean_error /= len(objpoints)
print(f"\n平均再投影誤差: {mean_error}")

# ====================================================
# オプション：歪み補正結果の確認（最初の画像を使用）
# ====================================================
test_img = cv2.imread(image_paths[0])
if test_img is not None:
    undistorted_img = cv2.fisheye.undistortImage(test_img, K, D, None, K)
    cv2.imshow("Original Image", test_img)
    cv2.imshow("Undistorted Image", undistorted_img)
    print("\nウィンドウ上で歪み補正結果を確認できます。任意のキーを押すと終了します。")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("テスト画像の読み込みに失敗しました。")
