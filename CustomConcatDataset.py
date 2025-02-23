from torch.utils.data import ConcatDataset
#在ConcatDataset基础上定义新类，实现了获得形如(X,label)的数据
class SimpleConcatDataset(ConcatDataset):
    def __init__(self, datasets):
        super().__init__(datasets)
        self.sub_datasets = datasets  # 显式保存子数据集列表

    def get_test_sample(self, index):#我想实现默认不输入时返回整个列表，但是不传入index时总是报错
        # 遍历子数据集，找到 idx 对应的那个
        for dataset in self.sub_datasets:
            if index < len(dataset):
                return dataset.get_test_sample(index)  # 调用子数据集的方法
            index -= len(dataset)
        raise IndexError("索引超出范围")

# 自定义子集类
