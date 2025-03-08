import csv
import pandas as pd
import numpy as np


def recommend_numbers(csv_file="History.csv", n_recent=200):
    """
    从csv_file读取双色球历史数据（需包含表头 ["Period","Red1","Red2","Red3","Red4","Red5","Red6","Blue"]），
    选取最近 n_recent 期做统计，并计算每个红蓝球的综合得分 = 出现频率 + 0.1 * log(遗漏 + 1)，
    最后返回综合得分最高的 6 个红球（升序）以及 1 个蓝球。

    参数：
    - csv_file: CSV 文件路径，默认 "History.csv"
    - n_recent: 统计的最近期数，默认 200

    返回：
    - top_reds_sorted: 选出的 6 个红球（从小到大排序的列表）
    - top_blue: 选出的蓝球（整数）
    """
    # 1) 读取 CSV
    df_raw = pd.read_csv("History.csv")
    df = df_raw.iloc[::-1].reset_index(drop=True)

    # 2) 可选择重命名列为小写(如需)，此处仅示范
    rename_map = {
        "Period": "period",
        "Red1": "red1",
        "Red2": "red2",
        "Red3": "red3",
        "Red4": "red4",
        "Red5": "red5",
        "Red6": "red6",
        "Blue": "blue",
    }
    for old_name in rename_map:
        if old_name in df.columns:
            df = df.rename(columns={old_name: rename_map[old_name]})

    # 如果最终列名中没有所需字段，会报错，可在此检查
    required_cols = ["period", "red1", "red2", "red3", "red4", "red5", "red6", "blue"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV缺少必要列: {missing}")

    # 3) 取最近 n_recent 期
    # 假设 df 是按时间/期号递增，若要倒序可以先 sort_values
    # 如果 period 越大表示越新的期，可以先:
    # df = df.sort_values(by="period", ascending=True)
    # 然后取 tail(n_recent)
    recent_draws = df.tail(n_recent)

    RED_RANGE = range(1, 34)
    BLUE_RANGE = range(1, 17)

    # 4) 统计出现次数
    red_counts = {r: 0 for r in RED_RANGE}
    blue_counts = {b: 0 for b in BLUE_RANGE}

    for idx, row in recent_draws.iterrows():
        for i in range(1, 7):
            red_counts[row[f"red{i}"]] += 1
        blue_counts[row["blue"]] += 1

    # 5) 统计遗漏 => “最近一次出现”距离
    #    我们用 DataFrame 的 index，计算 "all_indexes[-1] - appeared[-1]"
    #    当然也可以直接用行号来做近似。
    last_appear_red = {r: -1 for r in RED_RANGE}
    last_appear_blue = {b: -1 for b in BLUE_RANGE}

    all_indexes = recent_draws.index.tolist()
    if not all_indexes:
        raise ValueError(f"CSV 里可用的数据行不足，无法统计最近 {n_recent} 期！")

    max_idx = all_indexes[-1]  # 最近一期所在的行号

    # 对红球
    for r in RED_RANGE:
        appeared_idx = recent_draws[
            (recent_draws["red1"] == r)
            | (recent_draws["red2"] == r)
            | (recent_draws["red3"] == r)
            | (recent_draws["red4"] == r)
            | (recent_draws["red5"] == r)
            | (recent_draws["red6"] == r)
        ].index
        if len(appeared_idx) > 0:
            last_appear_idx = appeared_idx[-1]  # 最后一次出现
            last_appear_red[r] = max_idx - last_appear_idx

    # 对蓝球
    for b in BLUE_RANGE:
        appeared_idx = recent_draws[recent_draws["blue"] == b].index
        if len(appeared_idx) > 0:
            last_appear_idx = appeared_idx[-1]
            last_appear_blue[b] = max_idx - last_appear_idx

    # 6) 计算综合分数：出现频率 + 0.1 * log(遗漏+1)
    red_scores = {}
    for r in RED_RANGE:
        freq_part = red_counts[r] / float(n_recent)
        omit_part = 0
        if last_appear_red[r] >= 0:
            omit_part = np.log(1 + last_appear_red[r])
        red_scores[r] = freq_part + 0.1 * omit_part

    blue_scores = {}
    for b in BLUE_RANGE:
        freq_part = blue_counts[b] / float(n_recent)
        omit_part = 0
        if last_appear_blue[b] >= 0:
            omit_part = np.log(1 + last_appear_blue[b])
        blue_scores[b] = freq_part + 0.1 * omit_part

    # 7) 按分数从高到低选 6 个红球、1 个蓝球
    top_reds_desc = sorted(
        red_scores.keys(), key=lambda x: red_scores[x], reverse=True
    )[:6]
    top_blue = max(blue_scores.keys(), key=lambda x: blue_scores[x])

    # 最终红球从小到大排序
    top_reds_sorted = sorted(top_reds_desc)

    return top_reds_sorted, top_blue


# 测试入口
if __name__ == "__main__":
    reds, blue = recommend_numbers(csv_file="History.csv", n_recent=100)
    print("推荐红球(从小到大):", reds)
    print("推荐蓝球:", blue)
