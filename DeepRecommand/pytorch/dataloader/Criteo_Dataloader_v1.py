import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class CriteoDataset(Dataset):
    def __init__(self, data_path, features_name, label_name):
        """
        Args:
            data_path: string, .npz 文件路径
            features_name: list[str], 包含所有特征列名
            label_name: string, label列名
        """
        self.data = np.load(data_path)
        self.features_name = features_name
        self.label_name = label_name
        
    def __len__(self):
        return len(self.data[self.label_name])

    def __getitem__(self, idx):
        sample = {}
        for key in self.features_name:
            sample[key] = torch.tensor(self.data[key][idx], dtype=torch.long)
        # label 用 float
        sample[self.label_name] = torch.tensor(self.data[self.label_name][idx], dtype=torch.float)
        return sample


class CriteoDataloader(DataLoader):
    def __init__(
        self, 
        data_path, 
        features_name=None, 
        label_name='label', 
        batch_size=1024, 
        shuffle=False, 
        num_workers=0,
        pin_memory=True
    ):
        """
        初始化CriteoDataloader
        Args:
            data_path: string, .npz 文件路径
            features_name: list, 特征列名, 例如：['I1'...'I13', 'C1'...'C26']
            label_name: string, label 列名
            batch_size: int, 默认 1024
            shuffle: bool, 是否打乱样本顺序
            num_workers: int, 多少个worker线程来加载数据
            pin_memory: bool, 是否使用pin memory, 加快数据拷贝到GPU的速度
        """
        if features_name is None:
            # 默认取 13 数值 + 26 类别
            features_name = [f'I{i}' for i in range(1,14)] + [f'C{i}' for i in range(1,27)]

        self.dataset = CriteoDataset(data_path=data_path, 
                                     features_name=features_name, 
                                     label_name=label_name)
        super(CriteoDataloader, self).__init__(
            dataset=self.dataset, 
            batch_size=batch_size, 
            shuffle=shuffle, 
            num_workers=num_workers,
            pin_memory=pin_memory
        )
