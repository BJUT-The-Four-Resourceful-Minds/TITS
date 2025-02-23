from torch.utils.data import Subset
#在Subset基础上定义新类，实现了获得形如(X,label)的数据
class SimpleSubset(Subset):
    def get_test_sample(self, subset_idx=None):#我想实现默认不输入时返回整个列表，但是不传入index时总是报错
        if subset_idx==None:
            return self.dataset.get_test_sample()
        else:
            original_idx = self.indices[subset_idx]
            return self.dataset.get_test_sample(original_idx)

