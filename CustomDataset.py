from torch.utils.data import Dataset
#在Dataset基础上定义新类，#没实现#从dataloader获得形如(X,label)的数据
class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx=None):#我想实现默认不输入时返回整个列表，但是不传入index时总是报错
        if(idx==None):
            return self.data, self.labels
        else:
            # 可以根据需求修改返回的内容
            sample = self.data[idx]
            label = self.labels[idx]
            return sample, label