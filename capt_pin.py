import cv2
import numpy as np
import os

# ====================================================
# 設定パラメータ（単眼カメラ用）
# ====================================================
CONFIG = {
    "camera_index": 0,                   # カメラのデバイスID（必要に応じて変更）
    "checkerboard_dims": (9, 6),         # チェスボードの内部角点数 (横, 縦)
    "subpix_window_size": (3, 3),          # サブピクセル補正ウィンドウサイズ
    "subpix_zero_zone": (-1, -1),          # サブピクセル補正のゼロゾーン（-1は自動）
    "subpix_criteria": (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                        30, 0.001),    # サブピクセル補正の収束基準
    "focus_threshold": 100.0,            # 画像のシャープネス評価（ラプラシアン分散）の閾値
    "target_image_count": 30,            # 保存目標枚数
    "save_dir": "calib_images",          # 画像保存先ディレクトリ
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
# メイン処理
# ====================================================
def main():
    # カメラキャプチャの初期化
    cap = cv2.VideoCapture(CONFIG["camera_index"])
    if not cap.isOpened():
        print("カメラのオープンに失敗しました。デバイスIDを確認してください。")
        return

    saved_count = 0
    print("【単眼カメラ用】プロフェッショナル向けキャリブレーション撮影プログラム開始")
    print("チェスボードが適切に写り、十分なシャープネスがある場合に 's' キーで画像を保存してください。")
    print("終了は 'q' キー。目標保存枚数は {} 枚です。".format(CONFIG["target_image_count"]))

    while True:
        ret, frame = cap.read()
        if not ret:
            print("フレームの取得に失敗しました。")
            break

        # 生画像を保存用にコピー（オーバーレイ前の状態）
        raw_frame = frame.copy()

        # グレースケール画像に変換
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurry, focus_measure = is_image_blurry(gray, CONFIG["focus_threshold"])

        # チェスボードの検出
        found, corners = cv2.findChessboardCorners(
            gray,
            CONFIG["checkerboard_dims"],
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        # チェスボードが検出された場合、サブピクセル補正を実施
        if found:
            cv2.cornerSubPix(gray, corners, CONFIG["subpix_window_size"],
                             CONFIG["subpix_zero_zone"], CONFIG["subpix_criteria"])
            cv2.drawChessboardCorners(frame, CONFIG["checkerboard_dims"], corners, found)
            status_text = "Chessboard Detected"
            status_color = (0, 255, 0)
        else:
            status_text = "No Chessboard"
            status_color = (0, 0, 255)

        # オーバーレイ表示（プレビュー用）
        cv2.putText(frame, status_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
        focus_text = "Focus: {:.2f} {}".format(focus_measure, "(Blurry)" if blurry else "(Sharp)")
        cv2.putText(frame, focus_text, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, "Saved: {}/{}".format(saved_count, CONFIG["target_image_count"]),
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, "Press 's' to save, 'q' to quit", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        # 画面表示
        cv2.imshow("Calibration Capture (Single Camera)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("終了命令を受信しました。")
            break
        elif key == ord('s'):
            # チェスボード検出済みかつシャープな画像であれば保存
            if found and not blurry:
                filename = os.path.join(CONFIG["save_dir"], "calib_{:02d}.jpg".format(saved_count))
                # 生画像（raw_frame）を保存することで、オーバーレイが入らない画像を保存
                cv2.imwrite(filename, raw_frame)
                print("画像保存: {} (Focus: {:.2f})".format(filename, focus_measure))
                saved_count += 1
            else:
                print("保存条件不一致: {} / Focus: {:.2f}".format(
                    "Chessboard Detected" if found else "No Chessboard", focus_measure))

        # 目標枚数に達したら自動終了
        if saved_count >= CONFIG["target_image_count"]:
            print("目標画像枚数 {} 枚を保存しました。終了します。".format(CONFIG["target_image_count"]))
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()