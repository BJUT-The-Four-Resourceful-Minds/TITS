import pandas as pd
import re


def analyze_can_basic(file_path: str, window: float = 3.0) -> pd.DataFrame:
    """
    基础版CAN数据分析

    参数：
    file_path: 数据文件路径
    window: 时间窗口大小（秒），默认3秒

    返回：
    包含DLC统计和报文数量的合并DataFrame
    """
    # 数据解析
    pattern = r"Timestamp: (\d+\.\d+).*?ID: (\w+).*?DLC: (\d+)"
    data = []

    with open(file_path, "r") as f:
        for line in f:
            if match := re.search(pattern, line):
                data.append({
                    "Timestamp": float(match.group(1)),
                    "CAN_ID": match.group(2),
                    "DLC": int(match.group(3))
                })

    df = pd.DataFrame(data)

    # 时间窗口计算
    df["TimeWindow"] = (df["Timestamp"] // window).astype(int) * window

    # 基础统计
    dlc_stats = df.groupby(["TimeWindow", "CAN_ID"])["DLC"].agg(["mean", "std"]).reset_index()
    count_stats = df.groupby(["TimeWindow", "CAN_ID"]).size().reset_index(name="Count")

    # 合并结果
    return pd.merge(dlc_stats, count_stats, on=["TimeWindow", "CAN_ID"])


# 使用示例
result = analyze_can_basic("data/normal_run_data.txt")
print(result)
