import numpy as np
import pandas as pd


# 针对car_hacking Dataset的数据进行数据清洗
def clean_data(data, attack):
    # data = data.dropna(axis=0) // 该操作将有用信息全部删除了
    data = concatenate_columns(data)
    result = data[[0, 1, 2, 'new_column']].copy()
    if attack:
        col2_plus_2 = data[2] + 3
        result[3] = [data.loc[i, col] if col in data.columns else None
                     for i, col in enumerate(col2_plus_2)]
    result = result.dropna(axis=0)
    return result


def concatenate_columns(data):
    # 获取最大列数
    max_cols = data[2].astype(int) + 2
    max_col_overall = max_cols.max()

    # 创建一个空的结果Series
    result = pd.Series('', index=data.index)

    # 对每个可能的列进行向量化操作
    for col in range(3, max_col_overall + 1):
        if col in data.columns:
            # 只在需要该列的行中添加
            mask = max_cols > col - 1
            result[mask] += data[col][mask].astype(str)

    data['new_column'] = result
    return data


# 对car_hacking Dataset数据进行预处理和特征提取
def car_hacking_process_data(file_path):
    if 'txt' in file_path:
        file = pd.read_csv(file_path, header=None, sep=r'\s+')
        file = file.loc[:, [1, 3] + list(range(6, 15))]
        file = file[:-1]
        columns = range(0, 11)
        file.columns = columns
        attack = 0
    elif 'csv' in file_path:
        file = pd.read_csv(file_path)
        columns = range(0, 12)
        file.columns = columns
        attack = 1
    else:
        print("file_path不规范")
        return

    data = clean_data(file, attack)
    # print(data)

    result = []
    data_array = data.values  # 转换为 numpy 数组
    t = data_array[0][0]
    t_end = data_array[-1][0]
    index = 0
    flag = 1
    label = []
    num = []
    while t <= t_end:
        temp_data = []
        temp_dlc = []
        number = 0
        while index < len(data_array) and data_array[index][0] < t + 3:
            temp_data.append(data_array[index][1])
            # print(data_array[index])
            temp_dlc.append(data_array[index][3])
            if attack and data_array[index][-1] == 'T':
                flag = 0
            index += 1
            number += 1

        if temp_data and number >= 50:
            temp_data = np.array(temp_data)
            temp_dlc = pd.DataFrame(temp_dlc)
            temp_dlc = temp_dlc.groupby([0]).size()
            temp_res = [len(np.unique(temp_data)), np.mean(temp_dlc), np.std(temp_dlc)]
            result.append(temp_res)
            label.append(flag)
            flag = 1
            num.append(number)

        t += 3
    return result, label


if __name__ == '__main__':
    file_path_txt = r'../Car-Hacking Dataset/normal_run_data/normal_run_data.txt'
    file_path_csv = r"..\Car-Hacking Dataset\gear_dataset.csv"
    feature, label = car_hacking_process_data(file_path_csv)
    print(len(label))
    print(label)
