import numpy as np
import matplotlib.pyplot as plt

def prepare_matrix(uv, xyz):
    # mat_leftは11列で、各点につき2行必要
    mat_left = np.zeros((2 * len(xyz), 11))
    # mat_rightは各点につき2要素
    mat_right = np.zeros(2 * len(xyz))

    for i in range(len(xyz)):
        u = uv[i][0]
        v = uv[i][1]
        x, y, z = xyz[i]

        # 1行目を割り当てる
        mat_left[2*i] = [x, y, z, 1, 0, 0, 0, 0, -u*x, -u*y, -u*z]
        # 2行目を割り当てる
        mat_left[2*i+1] = [0, 0, 0, 0, x, y, z, 1, -v*x, -v*y, -v*z]

        # mat_rightの要素を割り当てる
        mat_right[2*i] = u
        mat_right[2*i+1] = v

    p, residuals, rank, s = np.linalg.lstsq(mat_left, mat_right, rcond=None)
    return p

def p_vector(p):
    p1 = np.array([p[0], p[1], p[2]])
    p2 = np.array([p[4], p[5], p[6]])
    p3 = np.array([p[8], p[9], p[10]])
    p14 = p[3]
    p24 = p[7]
    return p1, p2, p3, p14, p24

def pose_recon_2c(cam_num, P, POSE):
    result = [] 
    for i in range(len(POSE[1])):
        mat_l = np.empty((0, 3))
        mat_r = np.empty((0, 1))
        for j in range(cam_num):
            p1, p2, p3, p14, p24 = p_vector(P[j])
            u = POSE[j][i][0] * p3 - p1
            v = POSE[j][i][1] * p3 - p2
            mat_l = np.vstack((mat_l, u))
            mat_l = np.vstack((mat_l, v))
            mat_r = np.vstack((mat_r, np.array([p14 - POSE[j][i][0]])))
            mat_r = np.vstack((mat_r, np.array([p24 - POSE[j][i][1]])))

        mat_l_pinv = np.linalg.pinv(mat_l)
        glo = np.dot(mat_l_pinv, mat_r)
        result.append(glo.ravel().tolist())
    return np.array(result)


def MPJPE(predicted_points: np.ndarray, true_points: np.ndarray) -> float:
    # Ensure that 'predicted_points' and 'true_points' are numpy arrays with the shape (n_points, 3)
    assert predicted_points.shape == true_points.shape, "Shapes must match"

    # Calculate the Euclidean distance for each corresponding pair of points
    distances = np.linalg.norm(predicted_points - true_points, axis=1)

    # Compute the average of these distances
    mpjpe = np.mean(distances)
    return mpjpe

def sigmoid(x: np.ndarray, k: int = 50) -> np.ndarray:
    weights = 1 / (1 + np.exp(-k * (x - np.mean(x))))
    normalized_weights = weights / np.sum(weights)
    return normalized_weights

# 2つのdfの共通フレームを特定する（同じFPSに限る）
def sync_and_plot_keypoint(df1, df2, keypoint):
    """
    2つのデータフレームの指定キーポイントについて
    ・信号の可視化
    ・相互相関によるラグ推定
    ・同期
    ・同期後の信号可視化
    を行う

    Parameters
    ----------
    df1, df2 : pandas.DataFrame
        比較する2つのデータフレーム
    keypoint : str
        キーポイント名（例: 'LEFT_ANKLE_y'）

    Returns
    -------
    df1_sync, df2_sync : pandas.DataFrame
        同期後のデータフレーム
    time_lag : int
        推定されたラグ（フレーム数）
    """
    # --- 1. 両データのキーポイント信号を抽出 ---
    signal1 = df1[keypoint].values
    signal2 = df2[keypoint].values

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # 1段目：元データ
    axes[0].plot(signal1, label='cam1')
    axes[0].plot(signal2, label='cam2')
    axes[0].set_ylabel(keypoint)
    axes[0].set_title(f'Signal for {keypoint}')
    axes[0].legend()
    axes[0].grid()

    # --- 2. 平均値を差し引く（中心化） ---
    signal1_centered = signal1 - np.mean(signal1)
    signal2_centered = signal2 - np.mean(signal2)

    # --- 3. 相互相関関数を計算 ---
    correlation = np.correlate(signal1_centered, signal2_centered, mode='full')
    lags = np.arange(-len(signal1_centered)+1, len(signal2_centered))

    # --- 4. ピーク位置の差分からラグを特定 ---
    lag_index = np.argmax(correlation)
    time_lag = lags[lag_index]
    print(f"最大相関のラグ: {time_lag} フレーム")

    # 2段目：相互相関関数
    axes[1].plot(lags, correlation)
    axes[1].set_xlabel('Lag (frame)')
    axes[1].set_ylabel('Cross-correlation')
    axes[1].set_title(f'Cross-correlation: {keypoint}')
    axes[1].grid()

    plt.tight_layout()
    plt.show()

    # --- 6. データの同期 ---
    if time_lag > 0:
        df1_sync = df1.iloc[time_lag:].reset_index(drop=True)
        df2_sync = df2.iloc[time_lag:len(df1_sync)].reset_index(drop=True)
    elif time_lag < 0:
        df2_sync = df2.iloc[-time_lag:].reset_index(drop=True)
        df1_sync = df1.iloc[:len(df2_sync)].reset_index(drop=True)
    else:
        df1_sync = df1.reset_index(drop=True)
        df2_sync = df2.reset_index(drop=True)

    print(f"同期後のデータ数: {len(df1_sync)}, {len(df2_sync)}")

    # --- 7. 同期後のデータをグラフで可視化（Subplotで追加） ---
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df1_sync[keypoint], label='df1_sync')
    ax.plot(df2_sync[keypoint], label='df2_sync')
    ax.set_xlabel('Frame')
    ax.set_ylabel(keypoint)
    ax.set_title(f'Synchronized {keypoint} signals')
    ax.legend()
    ax.grid()
    plt.show()

    return df1_sync, df2_sync, time_lag