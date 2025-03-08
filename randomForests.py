import os
import pandas as pd
import numpy as np
import joblib  # pip install joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def build_train_data_single_number(df, target_num, is_red=True):
    """
    针对单个号码(红球 or 蓝球)，构造用于训练/测试的 (X, y).
    注意，这里不包含“下一期”的特征，专注于历史期做训练/评估。

    参数:
    - df: 包含历史数据的 DataFrame
          必须包含: "Red1"~"Red6","Blue","date"(datetime64 或可转化)等列
    - target_num: 要预测的号码 (红球 1..33 / 蓝球 1..16)
    - is_red: True 表示红球，False 表示蓝球

    返回: (X, y)
      - X: N×d 的特征矩阵
      - y: N×1 的标签，表示这一期是否出现 target_num
    """
    data_list = []
    labels = []

    # 从第3行开始，因为需要前两行做参考特征
    for i in range(2, len(df)):
        # 当前期 (行i)、前一期 (行 i-1)、前两期 (行 i-2)
        cur_row = df.iloc[i]
        prev_row = df.iloc[i - 1]
        prev2_row = df.iloc[i - 2]

        feats = []

        # 特征1: 前一期是否含 target_num
        if is_red:
            reds_prev = [prev_row[f"Red{j}"] for j in range(1, 7)]
            feats.append(1 if target_num in reds_prev else 0)
        else:
            feats.append(1 if target_num == prev_row["Blue"] else 0)

        # 特征2: 前两期是否含 target_num
        if is_red:
            reds_prev2 = [prev2_row[f"Red{j}"] for j in range(1, 7)]
            feats.append(1 if target_num in reds_prev2 else 0)
        else:
            feats.append(1 if target_num == prev2_row["Blue"] else 0)

        # 特征3: 最近10期出现次数
        start_idx = max(0, i - 10)
        slice_df = df.iloc[start_idx:i]  # 不含行i
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
        feats.append(count_appear)

        # 特征4: 月份 (如果无效日期，则设为0)
        cdate = cur_row["date"]
        month = cdate.month if pd.notnull(cdate) else 0
        feats.append(month)

        data_list.append(feats)

        # 标签: 当前期是否出现 target_num
        if is_red:
            reds_cur = [cur_row[f"Red{j}"] for j in range(1, 7)]
            labels.append(1 if target_num in reds_cur else 0)
        else:
            labels.append(1 if target_num == cur_row["Blue"] else 0)

    X = np.array(data_list)
    y = np.array(labels)
    return X, y


def build_next_feature_single_number(df, target_num, is_red=True):
    """
    针对单个号码(红球/蓝球)，构造“下一期”的特征 (仅 1×d),
    用于模型预测下一期出现概率。

    思路：把 df 的最后一行当做“当前期”，再往前2行当做“前两期”。
    注：若 df 行数少于 2，则无法构造 next feature, 返回 None。
    """
    if len(df) < 3:
        return None  # 数据不足

    # 最后一行(假设是当前期)
    i_cur = len(df) - 1
    i_prev = i_cur - 1
    i_prev2 = i_cur - 2

    cur_row = df.iloc[i_cur]
    prev_row = df.iloc[i_prev]
    prev2_row = df.iloc[i_prev2]

    feats = []

    # 特征1: 前一期是否含
    if is_red:
        reds_prev = [prev_row[f"Red{j}"] for j in range(1, 7)]
        feats.append(1 if target_num in reds_prev else 0)
    else:
        feats.append(1 if target_num == prev_row["Blue"] else 0)

    # 特征2: 前两期是否含
    if is_red:
        reds_prev2 = [prev2_row[f"Red{j}"] for j in range(1, 7)]
        feats.append(1 if target_num in reds_prev2 else 0)
    else:
        feats.append(1 if target_num == prev2_row["Blue"] else 0)

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
    feats.append(count_appear)

    # 特征4: 月份
    cdate = cur_row["date"]
    month = cdate.month if pd.notnull(cdate) else 0
    feats.append(month)

    return np.array(feats).reshape(1, -1)


def train_and_save_all_models(csv_file="History.csv", model_dir="models"):
    """
    读取csv_file，依次为 33个红球+16个蓝球训练独立的随机森林模型，并将模型保存到 model_dir 下。

    第一次训练时执行此函数；后续只需加载模型进行预测。
    """
    # 1) 读取CSV
    df_raw = pd.read_csv("History.csv")
    df = df_raw.iloc[::-1].reset_index(drop=True)

    # 如果CSV里存的日期列叫 "Lottery Date"，重命名为 "date"
    if "Lottery Date" in df.columns:
        df = df.rename(columns={"Lottery Date": "date"})

    # 转成 datetime
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # 若 model_dir 不存在则创建
    os.makedirs(model_dir, exist_ok=True)

    # 2) 分别训练红球1..33, 蓝球1..16
    #    并保存到 model_dir/red_{n}.pkl 和 model_dir/blue_{n}.pkl
    for rnum in range(1, 34):
        X, y = build_train_data_single_number(df, rnum, is_red=True)
        if len(X) < 10:
            # 数据不足，跳过或保存个 None
            continue

        # 分割训练/测试(简单做法，前80%训练，后20%测试)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        if len(X_train) < 5:
            continue

        clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        clf.fit(X_train, y_train)
        test_acc = clf.score(X_test, y_test)

        model_path = os.path.join(model_dir, f"red_{rnum}.pkl")
        joblib.dump(clf, model_path)
        print(f"[红球{rnum}] 模型保存至: {model_path}, 测试准确率={test_acc:.3f}")

    for bnum in range(1, 17):
        X, y = build_train_data_single_number(df, bnum, is_red=False)
        if len(X) < 10:
            continue
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        if len(X_train) < 5:
            continue

        clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        clf.fit(X_train, y_train)
        test_acc = clf.score(X_test, y_test)

        model_path = os.path.join(model_dir, f"blue_{bnum}.pkl")
        joblib.dump(clf, model_path)
        print(f"[蓝球{bnum}] 模型保存至: {model_path}, 测试准确率={test_acc:.3f}")


def load_models_and_predict_next_draw(csv_file="History.csv", model_dir="models"):
    """
    从 model_dir 加载之前训练好的模型，
    并结合 csv_file 的“最后一行” (当做当前期)，
    预测下一期各号码出现的概率；选出概率最高的 6 个红球 & 1 个蓝球。

    返回: (pred_reds, pred_blue)
    - pred_reds: 长度6的红球(从小到大)
    - pred_blue: 1个蓝球
    """
    df_raw = pd.read_csv("History.csv")
    df = df_raw.iloc[::-1].reset_index(drop=True)
    if "Lottery Date" in df.columns:
        df = df.rename(columns={"Lottery Date": "date"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # 先确保 model_dir 存在
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"模型目录 {model_dir} 不存在，请先训练并保存模型！")

    # 对红球 1..33 依次加载模型 & 计算下一期出现概率
    red_probs = {}
    for rnum in range(1, 34):
        model_path = os.path.join(model_dir, f"red_{rnum}.pkl")
        if not os.path.isfile(model_path):
            # 说明此号码未成功训练或未保存，设定一个默认概率
            red_probs[rnum] = 0.0
            continue

        clf = joblib.load(model_path)
        # 构造下一期特征
        next_feat = build_next_feature_single_number(df, rnum, is_red=True)
        if next_feat is None:
            red_probs[rnum] = 0.0
            continue
        prob = clf.predict_proba(next_feat)[0][1]  # "出现"这一类的概率
        red_probs[rnum] = prob

    # 选出概率最高的6个红球
    top6_reds = sorted(red_probs.keys(), key=lambda x: red_probs[x], reverse=True)[:6]
    top6_reds_sorted = sorted(top6_reds)

    # 对蓝球 1..16 同理
    blue_probs = {}
    for bnum in range(1, 17):
        model_path = os.path.join(model_dir, f"blue_{bnum}.pkl")
        if not os.path.isfile(model_path):
            blue_probs[bnum] = 0.0
            continue

        clf = joblib.load(model_path)
        next_feat = build_next_feature_single_number(df, bnum, is_red=False)
        if next_feat is None:
            blue_probs[bnum] = 0.0
            continue
        prob = clf.predict_proba(next_feat)[0][1]
        blue_probs[bnum] = prob

    # 选出概率最高的1个蓝球
    top1_blue = max(blue_probs.keys(), key=lambda x: blue_probs[x])

    return top6_reds_sorted, top1_blue


if __name__ == "__main__":
    # 第一次训练 & 保存模型
    train_and_save_all_models(csv_file="History.csv", model_dir="models")

    # 以后只需加载模型 & 预测下一期
    reds, blue = load_models_and_predict_next_draw(
        csv_file="History.csv", model_dir="models"
    )
    print("预测红球:", reds)
    print("预测蓝球:", blue)
