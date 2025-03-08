import os
import joblib  # 或者用 pickle
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def build_features_for_num(df, target_num, is_red=True):
    """
    针对单个号码(红球/蓝球)构造特征 X, 标签 y.
    - df: 包含历史数据的 DataFrame (需包含Red1..Red6,Blue,等)
    - target_num: 要预测的号码, 红球1~33, 蓝球1~16
    - is_red: True表示红球, False表示蓝球

    返回 (X, y):
      - X: 形如 (N, d) 的特征矩阵
      - y: 形如 (N,) 的标签数组(0/1)

    样例：简单地用“最近两期是否出现该号码、最近10期出现次数、月份”等做示范，
          你可根据需要自行扩展/修改。
    """
    # 简单演示：每行样本表示当前期(第 i 行)的数据，
    # 特征需要用到前两期( i-1, i-2 ) 以及近10期统计
    # y=当前期是否出现 target_num

    data_list = []
    label_list = []
    for i in range(2, len(df)):
        # 当前行
        curr = df.iloc[i]
        prev = df.iloc[i - 1]
        prev2 = df.iloc[i - 2]

        row_feats = []

        # 特征1：前一期是否含 target_num
        if is_red:
            # 取 Red1..Red6 列(你需确认CSV列名对应)
            reds_prev = [prev[f"Red{j}"] for j in range(1, 7)]
            row_feats.append(1 if target_num in reds_prev else 0)
        else:
            row_feats.append(1 if target_num == prev["Blue"] else 0)

        # 特征2：前两期是否含 target_num
        if is_red:
            reds_prev2 = [prev2[f"Red{j}"] for j in range(1, 7)]
            row_feats.append(1 if target_num in reds_prev2 else 0)
        else:
            row_feats.append(1 if target_num == prev2["Blue"] else 0)

        # 特征3：最近10期出现次数
        start_idx = max(0, i - 10)
        slice_df = df.iloc[start_idx:i]
        count_appear = 0
        if is_red:
            for _, rrow in slice_df.iterrows():
                reds_ = [rrow[f"Red{k}"] for k in range(1, 7)]
                if target_num in reds_:
                    count_appear += 1
        else:
            for _, rrow in slice_df.iterrows():
                if target_num == rrow["Blue"]:
                    count_appear += 1
        row_feats.append(count_appear)

        # 特征4：月份(如果有 Lottery Date 或 date 列)
        # 假设已经转成 datetime, 否则需要先 pd.to_datetime
        if "date" in curr and not pd.isnull(curr["date"]):
            month = curr["date"].month
        elif "Lottery Date" in curr and not pd.isnull(curr["Lottery Date"]):
            # 如果CSV里原列名是 Lottery Date
            month = pd.to_datetime(curr["Lottery Date"]).month
        else:
            month = 0
        row_feats.append(month)

        # 组装特征
        data_list.append(row_feats)

        # 标签: 当前期是否含 target_num
        if is_red:
            reds_curr = [curr[f"Red{j}"] for j in range(1, 7)]
            label_list.append(1 if target_num in reds_curr else 0)
        else:
            label_list.append(1 if target_num == curr["Blue"] else 0)

    X = np.array(data_list, dtype=np.float32)
    y = np.array(label_list, dtype=np.int32)
    return X, y


def train_xgb_models(csv_file="History.csv", model_dir="xgb_models"):
    """
    读取CSV，对红球1..33和蓝球1..16分别训练一个XGBoost二分类模型，
    并将模型保存到 model_dir。

    假设 CSV 包含列: Red1..Red6, Blue, (date或Lottery Date可选)。
    如果列名不同，请自行修改build_features_for_num。
    """
    df_raw = pd.read_csv("History.csv")
    df = df_raw.iloc[::-1].reset_index(drop=True)

    # 如果CSV里日期列叫 'Lottery Date', 先转成 'date' 并解析
    if "Lottery Date" in df.columns:
        df["date"] = pd.to_datetime(df["Lottery Date"], errors="coerce")
    else:
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        # 如果没有任何日期列，则 month=0 也没关系

    # 创建保存模型的目录
    os.makedirs(model_dir, exist_ok=True)

    # 先训练红球(1..33)
    for rnum in range(1, 34):
        X, y = build_features_for_num(df, rnum, is_red=True)
        if len(X) < 10:
            continue  # 样本太少

        # 时间序列切分(不随机shuffle)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        if len(X_train) < 5:
            continue

        model = xgb.XGBClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1, use_label_encoder=False
        )
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            # eval_metric="logloss",  # 移除
            # early_stopping_rounds=10, # 如果也不行，就一起移除
            verbose=False,
        )

        # 保存模型
        joblib.dump(model, os.path.join(model_dir, f"red_{rnum}.pkl"))
        print(f"[red {rnum}] 训练完成，样本数={len(X)}, 测试集={len(X_test)}")

    # 再训练蓝球(1..16)
    for bnum in range(1, 17):
        X, y = build_features_for_num(df, bnum, is_red=False)
        if len(X) < 10:
            continue

        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        if len(X_train) < 5:
            continue

        model = xgb.XGBClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1, use_label_encoder=False
        )
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            # eval_metric="logloss",  # 移除
            # early_stopping_rounds=10, # 如果也不行，就一起移除
            verbose=False,
        )

        # 保存模型
        joblib.dump(model, os.path.join(model_dir, f"blue_{bnum}.pkl"))
        print(f"[blue {bnum}] 训练完成，样本数={len(X)}, 测试集={len(X_test)}")


def build_next_features_for_num(df, target_num, is_red=True):
    """
    为某个号码(红/蓝)构建"下一期"的特征 (仅1条记录).
    思路和 build_features_for_num 类似，但只拿最后1行(若足够前几行存在)做特征。
    如果数据不足(比如少于3行)，可返回 None。
    """
    if len(df) < 3:
        return None

    i_cur = len(df) - 1
    i_prev = i_cur - 1
    i_prev2 = i_cur - 2
    curr = df.iloc[i_cur]
    prev = df.iloc[i_prev]
    prev2 = df.iloc[i_prev2]

    row_feats = []

    # 特征1: 前一期是否含 target_num
    if is_red:
        reds_prev = [prev[f"Red{j}"] for j in range(1, 7)]
        row_feats.append(1 if target_num in reds_prev else 0)
    else:
        row_feats.append(1 if target_num == prev["Blue"] else 0)

    # 特征2: 前两期是否含
    if is_red:
        reds_prev2 = [prev2[f"Red{j}"] for j in range(1, 7)]
        row_feats.append(1 if target_num in reds_prev2 else 0)
    else:
        row_feats.append(1 if target_num == prev2["Blue"] else 0)

    # 特征3: 最近10期出现次数
    start_idx = max(0, i_cur - 10)
    slice_df = df.iloc[start_idx:i_cur]
    count_appear = 0
    if is_red:
        for _, rrow in slice_df.iterrows():
            reds_ = [rrow[f"Red{k}"] for k in range(1, 7)]
            if target_num in reds_:
                count_appear += 1
    else:
        for _, rrow in slice_df.iterrows():
            if target_num == rrow["Blue"]:
                count_appear += 1
    row_feats.append(count_appear)

    # 特征4: 月份
    if "date" in curr and not pd.isnull(curr["date"]):
        month = curr["date"].month
    elif "Lottery Date" in curr and not pd.isnull(curr["Lottery Date"]):
        month = pd.to_datetime(curr["Lottery Date"]).month
    else:
        month = 0
    row_feats.append(month)

    X_next = np.array([row_feats], dtype=np.float32)  # shape (1,d)
    return X_next


def predict_next_draw(csv_file="History.csv", model_dir="xgb_models"):
    """
    从 model_dir 加载已训练的模型，对 CSV 的'最后一行'认为是当前期，
    构造下一期特征，并预测每个号码出现的概率。
    最终返回(reds, blue)，其中:
      reds: 概率最高的6个红球(从大到小选6,再按号码升序)
      blue: 概率最高的1个蓝球
    """
    df_raw = pd.read_csv("History.csv")
    df = df_raw.iloc[::-1].reset_index(drop=True)

    # 同训练时对日期列做一致处理
    if "Lottery Date" in df.columns:
        df["date"] = pd.to_datetime(df["Lottery Date"], errors="coerce")
    else:
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # 预测红球
    red_probs = {}
    for rnum in range(1, 34):
        model_path = os.path.join(model_dir, f"red_{rnum}.pkl")
        if not os.path.isfile(model_path):
            red_probs[rnum] = 0.0
            continue
        model = joblib.load(model_path)

        X_next = build_next_features_for_num(df, rnum, is_red=True)
        if X_next is None:
            red_probs[rnum] = 0.0
            continue

        prob = model.predict_proba(X_next)[0][1]
        red_probs[rnum] = prob

    # 选出概率最高的6个红球
    top6 = sorted(red_probs.keys(), key=lambda x: red_probs[x], reverse=True)[:6]
    top6_sorted = sorted(top6)

    # 预测蓝球
    blue_probs = {}
    for bnum in range(1, 17):
        model_path = os.path.join(model_dir, f"blue_{bnum}.pkl")
        if not os.path.isfile(model_path):
            blue_probs[bnum] = 0.0
            continue
        model = joblib.load(model_path)

        X_next = build_next_features_for_num(df, bnum, is_red=False)
        if X_next is None:
            blue_probs[bnum] = 0.0
            continue

        prob = model.predict_proba(X_next)[0][1]
        blue_probs[bnum] = prob

    # 选出概率最高的蓝球
    top_blue = max(blue_probs.keys(), key=lambda x: blue_probs[x])

    return top6_sorted, top_blue


if __name__ == "__main__":
    # 第一次训练并保存模型
    train_xgb_models(csv_file="History.csv", model_dir="xgb_models")

    # 预测下一期
    reds, b = predict_next_draw(csv_file="History.csv", model_dir="xgb_models")
    print("下一期红球:", reds)
    print("下一期蓝球:", b)
