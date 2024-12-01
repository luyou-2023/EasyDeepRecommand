"""
针对Criteo数据集预处理: 
空值填充，分桶、数据集划分、特征整理
"""
import os
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OrdinalEncoder
import json
import numpy as np
import math

class CriteoPreprocess():
    def __init__(self, 
                 data_dir = '../../Dataset/criteo/',        # 原始数据所在目录的相对路径
                 data_path='sample.csv',                    # 原始数据明
                 output_save_dir = '../../Dataset/criteo/', # 数据输出的存放目录
                 is_number_bucket=False                     # 数值型特征是否分桶
                 ):
        super(CriteoPreprocess, self).__init__()
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(current_dir, data_dir)                 # 通过相对路径存放文件
        self.data_path = os.path.join(self.data_dir, data_path)             # 样本路径
        self.output_save_dir = os.path.join(current_dir, output_save_dir)   # 数据输出的存放目录
        self.numeric_features = ['I' + str(i) for i in range(1, 14)]        # 数值型特征    
        self.categerical_features = ['C' + str(i) for i in range(1, 27)]    # 列别型特征
        self.label = 'label'
        self.is_numeric_bucket = False                                      # 数值特征是否分桶
        if self.is_numeric_bucket:
            self.numeric_features_dim = [4] * 13                            # 定义每个特征后续的embedding维度
        else:
            self.numeric_features_dim = [1] * 13
        self.categerical_features_dim = [4] * 26                            # 定义每个特征后续的embedding维度
    

    def categerical_null_fill(self, df, columns, n_neighbors=10):
        """
        对类别型特征进行空值填充
        Args:
            df: 需要填充的pd提取的数据
            columns: 需要填充的列，list型
            n_neighbors: KNN中的K. Defaults to 5.
        Returns:
            空值填充好的pd型数据
        """
        encoder = OrdinalEncoder()                                                          # 创建OrdinalEncoder对象
        df[columns] = encoder.fit_transform(df[columns])                                    # 使用OrdinalEncoder进行整数编码
        imputer = KNNImputer(n_neighbors=n_neighbors)                                       # 创建KNNImputer对象
        df_filled = imputer.fit_transform(df)                                               # 使用KNNImputer进行填充
        df_filled = pd.DataFrame(df_filled, columns=df.columns)                             # 将数组转换为DataFrame
        df_filled[columns] = encoder.inverse_transform(df_filled[columns].astype(int))      # 将编码转换回原始类别
        return df_filled


    def get_data_and_null_fill(self):
        """
        获取样本数据并进行空值填充
        Returns:
            空值填充好的样本数据
        """
        print("get data starting ...")
        raw_data = pd.read_csv(self.data_path)
        print("查看数据基本信息：")
        print(f"data.shape: {raw_data.shape} \n")
        print(f"{raw_data.info()}")
        print(f"\n缺失值情况: \n{raw_data.isnull().sum()}")
        num_features = ['I' + str(i) for i in range(1,14)]
        cate_features = ['C' + str(i) for i in range(1, 27)]
        raw_data[num_features] = raw_data[num_features].interpolate(method='linear', limit_direction='both')    # 数值型特征缺失值处理：使用前后均值
        raw_data = self.categerical_null_fill(df=raw_data, columns=cate_features, n_neighbors=10)               # 类别型特征缺失值处理：使用KNN聚类
        print(f"\n缺失值处理后情况: \n{raw_data.isnull().sum()}")
        print('get data end !!!')
        return raw_data


    def get_numeric_bucket_threshold(self, df, bucket_threshold_save_dir, threshold_value=56):
        """
        获取数值型特征的分桶阈值，同时获取每个特征的分桶数量
        Args:
            df: 样本数据
            numeric_features: 数值特征列表
            bucket_threshold_save_dir: 存放阈值的目录
            threshold_value: 当桶内的数量超过阈值才算有效区间. Defaults to 56.
        Returns:
            feature_map: 保存每个特征的分桶信息
        """
        feature_map = {}
        for i, feature in enumerate(self.numeric_features):
            # 计算最小值和最大值
            min_val = df[feature].min()
            max_val = df[feature].max()
            
            # 按数值排序并统计各数值出现次数
            value_counts = df[feature].value_counts().sort_index()
            # 初始化区间
            start = min_val
            count = 0
            buckets = []
            bucket_idx = 1
            buckets.append((-999999, min_val, 0))   # 防止后续出现小于当前数据集最小值的情况
            new_start = True
            for index, val in value_counts.items():
                if new_start:
                    count = val
                    if count >= threshold_value:
                        buckets.append((start, index, bucket_idx))
                        bucket_idx += 1
                        count = 0
                        new_start = True
                        start = index
                    else:
                        new_start = False
                elif count + val >= threshold_value:
                    # 当累积计数超过阈值时，记录桶区间，区间的含义是：除第一个和最后一个桶，其余当 a<x<=b时，x就属于（a,b]桶
                    buckets.append((start, index, bucket_idx))
                    count = 0
                    bucket_idx += 1
                    new_start = True
                    start = index
                else:
                    count += val
                    new_start = False
            
            # 如果还有未记录的区间
            if start is not None:
                buckets.append((start, max_val, bucket_idx))

            buckets.append((max_val, 999999, bucket_idx+1)) # 防止后续出现大于当前数据集最大值的情况

            # 保存到文件
            output_path = os.path.join(bucket_threshold_save_dir, f'numeric_{feature}.txt')
            with open(output_path, 'w') as f:
                for start, end, idx in buckets:
                    f.write(f'{start}\t{end}\t{idx}\n')
            
            # 更新特征映射
            feature_map[feature] = {
                "type": "numeric",
                "vocab_size": len(buckets),
                "feature_dim": self.numeric_features_dim[i]
            }
        
        return feature_map
    

    def get_categorical_features(self, df, categorical_save_dir, threshold_value=10):
        """
        获取特征型特征的列别，同时获取每个特征的分桶数量
        Args:
            df: 样本数据
            categorical_features: 列别特征列表
            categorical_save_dir: 存放类别的目录
            threshold_value: 当类别数量超过阈值才算有效类别,低于阈值的全部算others类. Defaults to 10  
        Returns:
            feature_map: 保存每个特征的类别信息
        """
        feature_map = {}
        feature_filter = {}
        for C_i, feature in enumerate(self.categerical_features):
            categories_count = df[feature].value_counts()   # 统计出现的类别，以及每一个类别出现的次数
            filter_count = 0

            output_path = os.path.join(categorical_save_dir, f'categorical_{feature}.txt')
            with open(output_path, 'w') as f:
                idx = 0
                for i, num in enumerate(categories_count):
                    if num >= threshold_value:                              # 小于阈值的类别不单独算一类，全部算others类，
                        f.write(f'{categories_count.index[i]}\t{idx}\n')
                        idx = idx + 1
                    else:
                        filter_count = filter_count + 1
                f.write(f'others\t{idx}')                                   # 添加一个"others"类别, 后续出现当前数据集不存在的类别或当前数据集中小于阈值的类别，全部归入others
            
            # 更新特征数据
            feature_map[feature] = {
                "type": "categorical",                              # 特征类型
                "vocab_size": idx+1,                                # 有效类别
                "feature_dim": self.categerical_features_dim[C_i]   # 每个特征的embedding维度
            }

            feature_filter[feature] = f"{feature}:\t类别总数:{filter_count+idx+1},\t过滤类别:{filter_count},\t超过阈值{threshold_value}的类别:{idx+1}"

        for k, v in feature_filter.items():
            print(v)

        return feature_map
    

    def get_all_features_buckets(self, df, bucket_dir, numeric_threshold_value=56):
        """
        对数值型特征进行分桶，统计类别型特征的所有类别
        Args:
            bucket_dir: 存放特征分桶的路径
            numeric_threshold_value: 数值特征分组的阈值. Defaults to 56.
            is_numeric_bucket: 是否对数值型特征进行分桶. Defaults to True，如果为False，则直接log_2(x)
        Others:
            criteo_feature_info: 保存criteo数据集的基本信息
        """
        numeric_feature_map = {}
        if self.is_numeric_bucket:
            numeric_feature_map = self.get_numeric_bucket_threshold(df=df, 
                                                                    bucket_threshold_save_dir=bucket_dir, 
                                                                    threshold_value=numeric_threshold_value)
        categorical_feature_map = self.get_categorical_features(df=df,
                                                                categorical_save_dir=bucket_dir)
        features_map = {**numeric_feature_map, **categorical_feature_map}    # 合并特征映射

        # 保存数据集的所有基本信息
        criteo_feature_info = {}
        criteo_feature_info['dataset_name'] = 'criteo'
        criteo_feature_info['is_numeric_bucket'] = self.is_numeric_bucket
        criteo_feature_info['numeric_feature'] = self.numeric_features
        criteo_feature_info['categorical_feature'] = self.categerical_features
        criteo_feature_info['label'] = self.label
        criteo_feature_info['numeric_feature_len'] = sum(self.numeric_features_dim)
        criteo_feature_info['categorical_feature_len'] = sum(self.categerical_features_dim)
        criteo_feature_info['sample_len'] = sum(self.numeric_features_dim) + sum(self.categerical_features_dim) # 每条样本经过embedding之后的长度
        criteo_feature_info['features_map'] = features_map
 
        # 保存feature_map
        feature_map_output_path = os.path.join(self.data_dir, 'feature_map.json')
        with open(feature_map_output_path, 'w') as f:
            json.dump(criteo_feature_info, f, indent=4)


    def original_data_to_bucket(self, df, bucket_dir, output_npz, output_csv):
        """
        将原始数据根据numeric型特征分桶映射为桶序，将类别型特征按类别映射为类别序号
        Args:
            df: 原始样本
            bucket_dir: 特征分桶的目录
            output_npz: 原始数据映射后的保存路径
            output_csv: 原始数据映射后的前100行保存路径(方便查看)
        """
        def load_bucket_info(bucket_dir, feature_name, is_numeric):
            bucket_file = os.path.join(bucket_dir, f"{feature_name}.txt")
            buckets = []
            with open(bucket_file, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if is_numeric:
                        buckets.append((float(parts[0]), float(parts[1]), int(parts[2])))
                    else:
                        buckets.append((parts[0], int(parts[1])))
            return buckets

        def map_numeric(value, buckets):
            if value < buckets[0][0]:
                return buckets[0][2]
            if value > buckets[-1][1]:
                return buckets[-1][2]
            for a, b, y in buckets[1:-1]:
                if (a < value <= b) or (a == b and value == b):
                    return y
            return buckets[-1][2]
        
        def norm_numeric(value):
            if value > 2:
                return np.floor(math.log(value, 2))
            return 1

        def map_categorical(value, buckets):
            for cate_str, y in buckets[:-1]:
                if value == cate_str:
                    return y
            return buckets[-1][1]
        
        processed_data = df.copy()
        
        # 处理 numeric features
        for i in range(1, 14):
            feature_name = f'I{i}'
            if self.is_numeric_bucket:
                buckets = load_bucket_info(bucket_dir, f'numeric_{feature_name}', True)
                processed_data[feature_name] = df[feature_name].apply(lambda x: map_numeric(x, buckets))
            else:
                processed_data[feature_name] = df[feature_name].apply(lambda x: norm_numeric(x))

        # 处理 categorical features
        for i in range(1, 27): 
            feature_name = f'C{i}'
            buckets = load_bucket_info(bucket_dir, f'categorical_{feature_name}', False)
            processed_data[feature_name] = df[feature_name].apply(lambda x: map_categorical(x, buckets))

        np.savez(output_npz, **{col: processed_data[col].to_numpy() for col in processed_data.columns}) # 将所有数据保存为npz文件，因为读取效率高。各文件类型读取效率可参考：https://juejin.cn/post/7429336556928876607        
        processed_data.head(100).to_csv(output_csv, index=False)                                        # 将前100行数据保存为csv,方便查看

    
    def process_all_operation(self, dataset_save_name="process_sample.npz"):
        """
        数据处理集合
        Args:
            dataset_save_name: 数据集处理后的最终命名
        """
        print("Step1: 开始读取数据，并进行空值填充...")
        raw_data = self.get_data_and_null_fill()

        print("Step2: 开始进行分桶处理...")
        bucket_dir = self.output_save_dir + 'bucket'
        self.get_all_features_buckets(df=raw_data, bucket_dir=bucket_dir, numeric_threshold_value=56)
        
        print("Step3: 将原始样本值按桶/类别映射为桶序/类别序号...")
        process_sample_path = self.output_save_dir + dataset_save_name
        process_sample_head_100_path = self.output_save_dir + "process_sample_head_100.csv"     # 将前100行数据保存为csv，方便查看
        self.original_data_to_bucket(df=raw_data, 
                                     bucket_dir=bucket_dir, 
                                     output_npz=process_sample_path, 
                                     output_csv=process_sample_head_100_path)



if __name__ == '__main__':
    preprocess = CriteoPreprocess(
        data_dir = '../../Dataset/criteo/',      # 原始数据所在目录的相对路径
        data_path='sample.csv',                       # 原始数据明
        output_save_dir = '../../Dataset/criteo/'   # 数据输出的存放目录
    )                 
    preprocess.process_all_operation(
        dataset_save_name="process_sample.npz"                # 数据处理后的文件名
    )