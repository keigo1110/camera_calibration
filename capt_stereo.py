import cv2
import numpy as np
import os

# ====================================================
# 設定パラメータ（ステレオカメラ用）
# ====================================================
CONFIG = {
    "left_camera_index": 0,              # 左カメラのデバイスID（必要に応じて変更）
    "right_camera_index": 1,             # 右カメラのデバイスID（必要に応じて変更）
    "checkerboard_dims": (9, 6),         # チェスボードの内部角点数 (横, 縦)
    "subpix_window_size": (3, 3),          # サブピクセル補正ウィンドウサイズ
    "subpix_zero_zone": (-1, -1),          # サブピクセル補正のゼロゾーン（-1は自動）
    "subpix_criteria": (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                        30, 0.001),    # サブピクセル補正の収束基準
    "focus_threshold": 100.0,            # 画像のシャープネス評価（ラプラシアン分散）の閾値
    "target_image_count": 30,            # 保存目標枚数
    "save_dir": "calib_images_stereo",   # 画像保存先ディレクトリ
}

# 保存ディレクトリの作成（存在しない場合）
if not os.path.exists(CONFIG["save_dir"]):
    os.makedirs(CONFIG["save_dir"])

# ====================================================
# 画像のブレ（フォーカス）の評価関数
# ====================================================
def is_image_blurry(gray_img, threshold):
    """
    ラプラシアンの分散を用いて画像のシャープネスを評価する関数。
    threshold 以下なら「ブレている」と判定する。

    Parameters:
        gray_img (ndarray): グレースケール画像
        threshold (float): シャープネスの閾値

    Returns:
        tuple: (is_blurry (bool), focus_measure (float))
    """
    focus_measure = cv2.Laplacian(gray_img, cv2.CV_64F).var()
    return (focus_measure < threshold, focus_measure)

# ====================================================
# メイン処理（ステレオキャリブレーション撮影）
# ====================================================
def main():
    # 左右カメラのキャプチャ開始
    left_cap = cv2.VideoCapture(CONFIG["left_camera_index"])
    right_cap = cv2.VideoCapture(CONFIG["right_camera_index"])

    if not left_cap.isOpened() or not right_cap.isOpened():
        print("カメラのオープンに失敗しました。各カメラのデバイスIDを確認してください。")
        return

    saved_count = 0
    print("【ステレオカメラ用】プロフェッショナル向けキャリブレーション撮影プログラム開始")
    print("左右両方のチェスボードが適切に写り、十分なシャープネスがある場合に 's' キーで画像を保存してください。")
    print("終了は 'q' キー。目標保存枚数は {} 組です。".format(CONFIG["target_image_count"]))

    while True:
        ret_left, frame_left = left_cap.read()
        ret_right, frame_right = right_cap.read()

        if not ret_left or not ret_right:
            print("いずれかのカメラからフレームが取得できませんでした。")
            break

        # 保存用にオーバーレイ前の生画像をコピー
        raw_left = frame_left.copy()
        raw_right = frame_right.copy()

        # グレースケール変換
        gray_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)

        # フォーカス評価
        blurry_left, focus_left = is_image_blurry(gray_left, CONFIG["focus_threshold"])
        blurry_right, focus_right = is_image_blurry(gray_right, CONFIG["focus_threshold"])

        # チェスボード検出
        ret_left_cb, corners_left = cv2.findChessboardCorners(
            gray_left,
            CONFIG["checkerboard_dims"],
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        ret_right_cb, corners_right = cv2.findChessboardCorners(
            gray_right,
            CONFIG["checkerboard_dims"],
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        if ret_left_cb:
            cv2.cornerSubPix(gray_left, corners_left, CONFIG["subpix_window_size"],
                             CONFIG["subpix_zero_zone"], CONFIG["subpix_criteria"])
            cv2.drawChessboardCorners(frame_left, CONFIG["checkerboard_dims"], corners_left, ret_left_cb)
            left_status = "Chessboard Detected"
            left_color = (0, 255, 0)
        else:
            left_status = "No Chessboard"
            left_color = (0, 0, 255)

        if ret_right_cb:
            cv2.cornerSubPix(gray_right, corners_right, CONFIG["subpix_window_size"],
                             CONFIG["subpix_zero_zone"], CONFIG["subpix_criteria"])
            cv2.drawChessboardCorners(frame_right, CONFIG["checkerboard_dims"], corners_right, ret_right_cb)
            right_status = "Chessboard Detected"
            right_color = (0, 255, 0)
        else:
            right_status = "No Chessboard"
            right_color = (0, 0, 255)

        # オーバーレイ表示（左画像）
        cv2.putText(frame_left, left_status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, left_color, 2)
        focus_text_left = "Focus: {:.2f} {}".format(focus_left, "(Blurry)" if blurry_left else "(Sharp)")
        cv2.putText(frame_left, focus_text_left, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame_left, "Saved: {}/{}".format(saved_count, CONFIG["target_image_count"]),
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # オーバーレイ表示（右画像）
        cv2.putText(frame_right, right_status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, right_color, 2)
        focus_text_right = "Focus: {:.2f} {}".format(focus_right, "(Blurry)" if blurry_right else "(Sharp)")
        cv2.putText(frame_right, focus_text_right, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame_right, "Press 's' to save, 'q' to quit", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        # 左右画像を横に連結して表示
        combined = np.hstack((frame_left, frame_right))
        cv2.imshow("Stereo Calibration Capture", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("終了命令を受信しました。")
            break
        elif key == ord('s'):
            # 両方の画像でチェスボードが検出され、かつシャープな画像であれば保存
            if ret_left_cb and ret_right_cb and (not blurry_left) and (not blurry_right):
                left_filename = os.path.join(CONFIG["save_dir"], "left_{:02d}.png".format(saved_count))
                right_filename = os.path.join(CONFIG["save_dir"], "right_{:02d}.png".format(saved_count))
                cv2.imwrite(left_filename, raw_left)
                cv2.imwrite(right_filename, raw_right)
                print("画像保存: {} と {} (Focus L: {:.2f}, Focus R: {:.2f})".format(
                    left_filename, right_filename, focus_left, focus_right))
                saved_count += 1
            else:
                print("保存条件不一致:")
                print("  左側: {} / Focus: {:.2f}".format("Chessboard Detected" if ret_left_cb else "No Chessboard", focus_left))
                print("  右側: {} / Focus: {:.2f}".format("Chessboard Detected" if ret_right_cb else "No Chessboard", focus_right))

        # 目標枚数に達したら自動終了
        if saved_count >= CONFIG["target_image_count"]:
            print("目標画像枚数 {} 組を保存しました。終了します。".format(CONFIG["target_image_count"]))
            break

    left_cap.release()
    right_cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()