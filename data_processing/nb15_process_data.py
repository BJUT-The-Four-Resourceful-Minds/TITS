import numpy as np
import pandas as pd


# 针对nb15 Dataset进行数据预处理与特征提取

def nb15_process_data(file_path):
    # 读取 CSV 文件
    file_csv = pd.read_csv(file_path, header=None, low_memory=False)
    # 选择需要的列
    file = file_csv.iloc[:, [0, 2, 6, 7, 9, 10, 14, 15, 16, 17, 28, 48]]

    file = file.dropna(axis=0)
    #对文件进行排序和重新编号
    file = file.sort_values(by=28)
    file = file.reset_index(drop=True)
    # 初始化结果列表
    result = []
    label = []
    # 获取起始时间和结束时间
    t = file.loc[0, 28]
    t_end = list(file.loc[:, 28])[-1]
    while True:
        # 创建时间窗口的掩码
        mask = (file.loc[:, 28] >= t) & (file.loc[:, 28] < t + 3)
        # 根据掩码选择数据
        temp_data = file[mask]
        # 进行分组求和操作
        result1 = temp_data.groupby(0)[14].sum()
        result2 = temp_data.groupby(0)[6].sum()
        result3 = temp_data.groupby([0, 2])[9].sum() + temp_data.groupby([0, 2])[10].sum()
        result4 = temp_data.groupby([0, 2])[14].sum() + temp_data.groupby([0, 2])[15].sum()
        result5 = temp_data.groupby([0, 2])[16].sum() + temp_data.groupby([0, 2])[17].sum()
        # 计算统计量

        temp_res = [np.mean(result1), np.std(result1),
                    np.mean(result2), np.std(result2),
                    np.mean(result3), np.std(result3),
                    np.mean(result4), np.std(result4),
                    np.mean(result5), np.std(result5)]

        # flag =1是正常0是攻击
        # 如果时间窗口内有一个为攻击则标记为flag=0
        # 我没有找到label,自己写了一个
        flag = 0 if (temp_data[48] == 1).any() else 1
        # 将当前时间窗口的统计结果添加到结果列表中
        if len(temp_data)>40:
            result.append(temp_res)
            label.append(flag)

        # 更新时间
        true_indices = np.where(mask)[0]
        next_indices = true_indices[-1] + 1
        t = file.loc[next_indices, 28]

        # 如果时间超过结束时间，则退出循环
        if t + 3 > t_end:
            break

    final_result = np.array(result)
    print('read over')
    print(len(final_result))
    return final_result, label
