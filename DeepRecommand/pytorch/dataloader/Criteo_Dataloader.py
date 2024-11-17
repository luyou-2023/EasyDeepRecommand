import torch
from torch.utils.data import Dataset
import numpy as np
from torch.utils.data import DataLoader

class CriteoDataset(Dataset):
    def __init__(self, data_path, features_name, label_name):
        self.data = np.load(data_path)
        self.features_name = features_name
        self.label_name = label_name
        
    def __len__(self):
        return len(self.data[self.label_name])

    def __getitem__(self, idx):
        sample = {key: torch.tensor(self.data[key][idx]) for key in self.features_name}
        sample[self.label_name] = torch.tensor(self.data[self.label_name][idx])
        return sample


class CriteoDataloader(DataLoader):
    def __init__(self, data_path, 
                 features_name=['I' + str(i) for i in range(1,14)] + ['C' + str(i) for i in range(1, 27)], 
                 label_name='label', 
                 batch_size=4, shuffle=False, num_workers=0):
        """
        初始化CriteoDataloader
        Args:
            data_path: string, .npz文件的路径
            features_name: list, 数据所有特征的名, 如：['I' + str(i) for i in range(1,14)] + ['C' + str(i) for i in range(1, 27)]
            batch_size: int, batch_size值, Defaults to 4.
            shuffle: 数据集顺序是否打乱. Defaults to False.
            num_workers: 加载数据时，使用几个线程一起加载. 有时使用多线程加载数据会爆错，默认为0。Defaults to 0.
        Retures:
            无返回，实例话数据后，就可以使用enumerate遍历批次数据，每个批次里面都是一个tensor类型的dict,
            dict中的keys为各个特征名，每个value为list,对应特征数据，如：{'label': [1, 0, 1], 'I1': [1.0,2.0, 5.0] ...} 
        """
        self.dataset = CriteoDataset(data_path=data_path, 
                                     features_name=features_name, 
                                     label_name=label_name)
        super(CriteoDataloader, self).__init__(
            dataset=self.dataset, 
            batch_size=batch_size, 
            shuffle=shuffle, 
            num_workers=num_workers)
        
        self.batchs = np.ceil(self.dataset.__len__() / self.batch_size)


if __name__ == '__main__':
    # 测试数据
    data_path = '/Users/ctb/WorkSpace/EasyDeepRecommend/Dataset/criteo/process_sample.npz'
    features_name = ['I' + str(i) for i in range(1,14)] + ['C' + str(i) for i in range(1, 27)]
    data_loader = CriteoDataloader(data_path=data_path, 
                                   features_name=features_name, 
                                   label_name='label', 
                                   batch_size=512, 
                                   shuffle=False)

    print("batchs = ", data_loader.batchs)
    for i, batch in enumerate(data_loader):
        print(f"第 {i} 批：")
        print(batch['label'], batch['I1'], batch['I1'].shape)
        break  # 只打印第一个批次的数据

