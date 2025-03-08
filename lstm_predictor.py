import pandas as pd
import numpy as np
import os

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam


def train_lstm_model(
    csv_file="History.csv",
    model_file="lstm_model.h5",
    window_size=5,
    epochs=50,
    batch_size=16,
):
    """
    读取 CSV 数据并训练 LSTM 模型，将训练好的模型保存到 model_file.

    参数：
    - csv_file: CSV 文件路径，默认为 "History.csv"。
      要求包含列: "Red1"~"Red6", "blue" (或 "Blue")。
    - model_file: 训练好的模型保存路径，默认 "lstm_model.h5"。
    - window_size: 序列窗口长度，默认 5，即用过去5期数据预测第6期。
    - epochs: 训练轮数，默认 50。
    - batch_size: mini-batch 大小，默认 16。

    返回：
    - model: 训练好的 Keras 模型对象
    """
    # 1) 读取 CSV
    df_raw = pd.read_csv("History.csv")
    df = df_raw.iloc[::-1].reset_index(drop=True)

    # 若列名中是 "Blue" 而非 "blue", 可改如下：
    rename_map = {}
    if "Blue" in df.columns and "blue" not in df.columns:
        rename_map["Blue"] = "blue"
    for i in range(1, 7):
        if f"Red{i}" in df.columns and f"red{i}" not in df.columns:
            rename_map[f"Red{i}"] = f"red{i}"
    if rename_map:
        df = df.rename(columns=rename_map)

    required_cols = [f"red{i}" for i in range(1, 7)] + ["blue"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"CSV中缺少必要列: {c}")

    # 2) 构建 (num_samples, 7) 的数组，每行 [red1,red2,red3,red4,red5,red6,blue]
    all_draws = []
    for _, row in df.iterrows():
        draw = [row[f"red{i}"] for i in range(1, 7)] + [row["blue"]]
        all_draws.append(draw)
    all_draws = np.array(all_draws, dtype=np.float32)  # (N, 7)

    # 3) 构造序列数据: 使用 window_size 条数据预测第 (window_size+1) 条
    X_seq = []
    Y_seq = []
    for i in range(len(all_draws) - window_size):
        X_seq.append(all_draws[i : i + window_size])  # shape (window_size, 7)
        Y_seq.append(all_draws[i + window_size])  # shape (7,)

    X_seq = np.array(X_seq)  # (N_seq, window_size, 7)
    Y_seq = np.array(Y_seq)  # (N_seq, 7)

    # 4) 简单归一化(将红球除以33.0，蓝球除以16.0)
    max_red = 33.0
    max_blue = 16.0
    X_seq_norm = X_seq.copy()
    Y_seq_norm = Y_seq.copy()

    # 前 6 列是红球
    X_seq_norm[..., :6] /= max_red
    # 第 7 列是蓝球
    X_seq_norm[..., 6] /= max_blue
    Y_seq_norm[..., :6] /= max_red
    Y_seq_norm[..., 6] /= max_blue

    # 5) 划分训练/测试集 (简单做法：前80%训练，后20%测试)
    split_idx = int(len(X_seq_norm) * 0.8)
    X_train, X_test = X_seq_norm[:split_idx], X_seq_norm[split_idx:]
    Y_train, Y_test = Y_seq_norm[:split_idx], Y_seq_norm[split_idx:]

    # 6) 构建并训练模型
    model = Sequential()
    model.add(LSTM(64, input_shape=(window_size, 7)))
    model.add(Dense(7, activation="sigmoid"))
    model.compile(
        loss="mean_squared_error",  # 原先写 "mse"，改成更规范的 "mean_squared_error"
        optimizer=Adam(0.001),
    )

    model.fit(
        X_train,
        Y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, Y_test),
        verbose=1,
    )

    # 7) 保存模型
    model.save(model_file)
    print(f"模型已保存到: {model_file}")
    return model


def predict_next_draw(
    csv_file="History.csv", model_file="lstm_model.h5", window_size=5
):
    """
    使用已经训练好的 LSTM 模型(model_file)，读取 CSV 的最后 window_size 期作为输入，
    预测下一期的 (red1~red6, blue)。

    参数：
    - csv_file: CSV 文件路径
    - model_file: 训练好的模型路径
    - window_size: 与训练时相同的序列长度

    返回：
    - pred_reds: 长度6的 numpy数组(预测红球)
    - pred_blue: 整数(预测蓝球)
    """
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"模型文件 {model_file} 不存在，请先训练。")

    # 1) 加载模型
    model = load_model(model_file)

    # 2) 读取数据并做同样处理
    df_raw = pd.read_csv("History.csv")
    df = df_raw.iloc[::-1].reset_index(drop=True)

    rename_map = {}
    if "Blue" in df.columns and "blue" not in df.columns:
        rename_map["Blue"] = "blue"
    for i in range(1, 7):
        if f"Red{i}" in df.columns and f"red{i}" not in df.columns:
            rename_map[f"Red{i}"] = f"red{i}"
    if rename_map:
        df = df.rename(columns=rename_map)

    required_cols = [f"red{i}" for i in range(1, 7)] + ["blue"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"CSV中缺少必要列: {c}")

    # 3) 构建全部 (N,7) 数据
    all_draws = []
    for _, row in df.iterrows():
        draw = [row[f"red{i}"] for i in range(1, 7)] + [row["blue"]]
        all_draws.append(draw)
    all_draws = np.array(all_draws, dtype=np.float32)

    # 若数据总长 < window_size，无法预测
    if len(all_draws) < window_size:
        raise ValueError(f"数据不足，无法取到最后 {window_size} 期样本")

    # 4) 取最后 window_size 期作为输入
    last_seq = all_draws[-window_size:]  # shape (window_size, 7)

    # 做同样的归一化
    max_red = 33.0
    max_blue = 16.0
    seq_norm = last_seq.copy()
    seq_norm[..., :6] /= max_red
    seq_norm[..., 6] /= max_blue

    # 5) 扩展维度 => (1,window_size,7)
    seq_norm = np.expand_dims(seq_norm, axis=0)

    # 6) 模型预测 => 返回 shape (1,7)
    pred_norm = model.predict(seq_norm)[0]  # (7,)

    # 7) 反归一化 => [0..1] => [1..33/16]
    pred_reds = np.round(pred_norm[:6] * max_red).astype(int)
    pred_blue = int(round(pred_norm[6] * max_blue))

    # 可做截断，确保在合法范围内(1..33, 1..16)
    pred_reds = np.clip(pred_reds, 1, 33)
    pred_blue = max(1, min(16, pred_blue))

    return pred_reds, pred_blue


if __name__ == "__main__":
    # 示例：训练 + 预测
    # 第一次训练
    model = train_lstm_model(
        csv_file="History.csv",
        model_file="lstm_model.h5",
        window_size=100,
        epochs=20,
        batch_size=16,
    )

    # 预测下一期
    reds, blue = predict_next_draw(
        csv_file="History.csv", model_file="lstm_model.h5", window_size=100
    )
    print("LSTM 预测红球:", reds)
    print("LSTM 预测蓝球:", blue)
