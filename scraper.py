import os
import csv
import requests
from bs4 import BeautifulSoup


def scrape_history_data(limit=100):
    """
    从远端抓取最近 limit 期的数据。
    返回结果是一个列表，每个元素形如：
    [期号, 红1, 红2, 红3, 红4, 红5, 红6, 蓝, 开奖日期]
    如果抓取失败或未找到数据则返回空列表。
    """
    ajax_url = (
        f"https://datachart.500.com/ssq/history/newinc/history.php?limit={limit}&sort=0"
    )
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0 Safari/537.36",
        "Referer": "https://datachart.500.com/ssq/history/history.shtml",
    }
    try:
        response = requests.get(ajax_url, headers=headers, timeout=10)
        response.encoding = "gb2312"
    except Exception as e:
        print(f"请求出错: {e}")
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    tbody = soup.find("tbody", id="tdata")
    if not tbody:
        print("未找到数据区域！")
        return []

    rows = tbody.find_all("tr")
    data = []
    for row in rows:
        cols = row.find_all("td")
        cols_text = [col.get_text(strip=True) for col in cols]
        # 检查数据行是否足够（期号、红球6个、蓝球及开奖日期共 9 列需要的位置分别在：索引 0、1-6、7、15）
        if len(cols_text) >= 16:
            period = cols_text[0]
            red_balls = cols_text[1:7]
            blue_ball = cols_text[7]
            lottery_date = cols_text[15]
            row_data = [period] + red_balls + [blue_ball, lottery_date]
            data.append(row_data)

    return data


def get_local_latest_date(filename):
    """
    从本地CSV文件中获取最新一期的开奖日期。
    这里假设最新的数据在文件开头（因为你写入的顺序可能是从最新到最旧）。
    也可以根据实际情况读取最后一行。
    如果文件不存在或没有任何数据，返回 None。
    """
    if not os.path.exists(filename):
        return None

    with open(filename, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)
        if len(rows) <= 1:
            # 说明只有表头或者是空文件
            return None
        # rows[1] 是写入的第一行数据（如果你是从新到旧写入的）
        # 其中开奖日期在索引位置 -1
        # 如果你表头是: "Period", "Red1", ..., "Blue", "Lottery Date"
        # 则数据行就是 [期号, 红1, ..., 蓝, 日期]
        # newest_date = rows[1][-1]  # 看你写入的顺序
        # 但是这取决于你在append_data_to_file里是怎样写入的，若越新越上，则第一行（rows[1]）确实是最新的
        # 如果你最新的是在文件末尾，就要换成 rows[-1]
        newest_date = rows[1][-1]
        return newest_date


def append_data_to_file(filename, data):
    """
    追加写入到 CSV, 并返回成功写入的新数据条数。
    如果本地已存在的期号，就不再重复写入。
    """
    existing_periods = set()
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                if row:
                    existing_periods.add(row[0])

    new_count = 0
    with open(filename, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        # 如果文件还没有写过任何数据，则写入表头
        if not existing_periods:
            header = [
                "Period",
                "Red1",
                "Red2",
                "Red3",
                "Red4",
                "Red5",
                "Red6",
                "Blue",
                "Lottery Date",
            ]
            writer.writerow(header)

        for row in data:
            period = row[0]
            if period not in existing_periods:
                writer.writerow(row)
                existing_periods.add(period)
                new_count += 1

    return new_count


def scrape_and_save():
    """
    先尝试抓取最新数据（limit=100）。
    无论有没有新数据，都要得到一个“最新一期的日期”和“本次新增条数”返回。

    返回: (newest_lottery_date, added_count)
    """
    filename = "History.csv"

    # 先抓取远端数据
    data = scrape_history_data(limit=100)

    if not data:
        # 如果远端没有抓到任何数据，则尝试从本地文件获取“最新一期的日期”
        print("未爬取到数据，从本地获取最新日期...")
        newest_date_local = get_local_latest_date(filename)
        if newest_date_local:
            # 没有新数据，新增条数=0，但仍能返回本地已有的最新日期
            return newest_date_local, 0
        else:
            # 本地也没有数据
            return None, 0
    else:
        # 抓取到的数据，通常第一条(索引 0) 就是最新的一期
        # data[0] 的最后一个元素是开奖日期
        newest_date_scraped = data[0][-1]

        # 尝试将它们写入本地文件，得到本次新增条数
        new_count = append_data_to_file(filename, data)
        print(f"本次新增条数: {new_count}")

        # 如果 new_count > 0，则 newest_date_scraped 也就是最新一期的日期
        # 如果 new_count == 0，说明本次抓到的数据都已存在，但还是可以用 newest_date_scraped
        # 或者用 get_local_latest_date(filename)，二者应该是一致的
        if new_count == 0:
            # double check
            newest_date_local = get_local_latest_date(filename)
            if newest_date_local:
                return newest_date_local, 0
            else:
                return newest_date_scraped, 0
        else:
            # 有新写入，则 newest_date_scraped 肯定是最新的
            return newest_date_scraped, new_count
