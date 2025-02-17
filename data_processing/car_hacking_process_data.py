import pandas as pd
import numpy as np


def clean_data(data):
    data = data.dropna(axis=0)
    data_new = data
    data = data.astype(str)

    try:
        data = data.loc[:, [1, 3, 4, 5, 6, 7, 8, 9, 10]].apply(lambda col: col.map(lambda x: int(x, 16)))
    except ValueError:
        print(data)
    data_new.loc[:, [1, 3, 4, 5, 6, 7, 8, 9, 10]] = data.loc[:, [1, 3, 4, 5, 6, 7, 8, 9, 10]]
    return data_new


def car_hacking_process_data(file_path):
    if 'txt' in file_path:
        file = pd.read_csv(file_path, header=None, sep=r'\s+')
        file = file.loc[:, [1, 3] + list(range(6, 15))]
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

    data = clean_data(file)
    result = []
    data_array = data.values  # 转换为 numpy 数组
    t = data_array[0][0]
    t_end = data_array[-1][0]
    index = 0
    flag = 1
    label = []
    while t <= t_end:
        temp_data = []
        while index < len(data_array) and data_array[index][0] < t + 3:
            temp_data.append(data_array[index][[1] + list(range(3, 3+int(data_array[index][2])))])
            if attack and data_array[index][-1] == 'T':
                flag = 0
            index += 1

        if temp_data:
            temp_data = np.array(temp_data)
            temp_mac = temp_data[:][1:].ravel()
            temp_res = [len(np.unique(temp_data[:][0])), np.mean(temp_mac), np.std(temp_mac)]
            result.append(temp_res)

            label.append(flag)
            flag = 1

        t += 3
    return result, label


if __name__ == '__main__':
    file_path_txt = r'../Car-Hacking Dataset/normal_run_data/normal_run_data.txt'
    file_path_csv = r"..\Car-Hacking Dataset\gear_dataset.csv"
    feature, label = car_hacking_process_data(file_path_csv)
    print(len(label))
    print(label)
