import os
import pandas as pd
import numpy as np
import math
import time
from collections import defaultdict
import json
from memory_profiler import profile  # 检测内存占用


class CriteoPreprocess:
    def __init__(
        self,
        data_dir='../../Dataset/criteo/',  # 原始数据所在目录的相对路径
        data_path='sample.csv',  # 原始数据文件名
        output_save_dir='../../Dataset/criteo/',  # 输出数据的保存目录
        dataset_save_name='process_sample.npz',  # 输出数据文件名
        is_number_bucket=False  # 数值型特征是否分桶
    ):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(current_dir, data_dir)  # 通过相对路径存放文件
        self.data_path = os.path.join(self.data_dir, data_path)  # 样本路径
        self.output_save_dir = os.path.join(current_dir, output_save_dir)  # 数据输出的存放目录
        self.dataset_save_name = dataset_save_name  # 输出数据文件名
        self.numeric_features = ['I' + str(i) for i in range(1, 14)]  # 数值型特征
        self.categorical_features = ['C' + str(i) for i in range(1, 27)]  # 类别型特征
        self.label = 'label'
        self.is_numeric_bucket = is_number_bucket  # 数值特征是否分桶
        if self.is_numeric_bucket:
            self.numeric_features_dim = [4] * 13  # 定义每个特征后续的embedding维度
        else:
            self.numeric_features_dim = [1] * 13
        self.categorical_features_dim = [4] * 26  # 定义每个特征后续的embedding维度

    def categorical_null_fill(self, df, columns, K=10):
        """
        使用前向填充和后向填充结合众数进行类别型特征的空值填充
        Args:
            df: 需要填充的pd.DataFrame
            columns: 需要填充的列，list型
            K: 填充时考虑的邻近范围（这里暂未使用，保留参数以保持接口一致）
        Returns:
            空值填充好的pd.DataFrame
        """
        for col in columns:
            ffill = df[col].fillna(method='ffill')
            bfill = df[col].fillna(method='bfill')
            mode = df[col].mode()[0] if not df[col].mode().empty else 'unknown'
            df[col] = ffill.combine(bfill, lambda x, y: x if pd.notnull(x) else y)
            df[col].fillna(mode, inplace=True)
        return df

    def get_data_and_null_fill(self):
        """
        获取样本数据并进行空值填充
        Returns:
            空值填充好的样本数据
        """
        print("get data starting ...")
        raw_data = pd.read_csv(self.data_path)
        print(f"data.shape: {raw_data.shape} \n")

        if not raw_data.isnull().sum().any():
            print("数据中不存在空值，直接返回")
            return raw_data

        print(f"\n缺失值情况: \n{raw_data.isnull().sum()}")
        raw_data[self.numeric_features] = raw_data[self.numeric_features].interpolate(method='linear', limit_direction='both')
        raw_data = self.categorical_null_fill(df=raw_data, columns=self.categorical_features)

        print(f"\n缺失值处理后情况: \n{raw_data.isnull().sum()}")
        print('get data end !!!')
        return raw_data

    def get_numeric_bucket_threshold(self, df, bucket_threshold_save_dir, threshold_value=56):
        """
        获取数值型特征的分桶阈值，同时获取每个特征的分桶数量
        Args:
            df: 样本数据
            bucket_threshold_save_dir: 存放阈值的目录
            threshold_value: 当桶内的数量超过阈值才算有效区间. Defaults to 56.
        Returns:
            feature_map: 保存每个特征的分桶信息
        """
        feature_map = {}
        for i, feature in enumerate(self.numeric_features):
            min_val = df[feature].min()
            max_val = df[feature].max()

            value_counts = df[feature].value_counts().sort_index()
            cumulative_counts = value_counts.cumsum()

            bucket_edges = [min_val]
            last_idx = 0
            for idx, cumulative_count in enumerate(cumulative_counts):
                if cumulative_count - cumulative_counts.iloc[last_idx] >= threshold_value:
                    bucket_edges.append(value_counts.index[idx])
                    last_idx = idx
            bucket_edges.append(max_val)

            buckets = [(-np.inf, bucket_edges[0], 0)]
            for b in range(1, len(bucket_edges)):
                buckets.append((bucket_edges[b-1], bucket_edges[b], b))
            buckets.append((bucket_edges[-1], np.inf, len(bucket_edges)))

            output_path = os.path.join(bucket_threshold_save_dir, f'numeric_{feature}.txt')
            with open(output_path, 'w') as f:
                for start, end, idx in buckets:
                    f.write(f'{start}\t{end}\t{idx}\n')

            feature_map[feature] = {
                "type": "numeric",
                "vocab_size": len(buckets),
                "feature_dim": self.numeric_features_dim[i]
            }

        return feature_map

    def get_categorical_features(self, df, categorical_save_dir, threshold_value=10):
        """
        获取类别型特征的类别，同时获取每个特征的分桶数量
        Args:
            df: 样本数据
            categorical_save_dir: 存放类别的目录
            threshold_value: 当类别数量超过阈值才算有效类别,低于阈值的全部算others类. Defaults to 10
        Returns:
            feature_map: 保存每个特征的类别信息
        """
        feature_map = {}
        feature_filter = {}
        for C_i, feature in enumerate(self.categorical_features):
            categories_count = df[feature].value_counts()
            valid_categories = categories_count[categories_count >= threshold_value].index.tolist()

            category_mapping = {category: idx for idx, category in enumerate(valid_categories)}
            category_mapping['others'] = len(valid_categories)

            output_path = os.path.join(categorical_save_dir, f'categorical_{feature}.txt')
            with open(output_path, 'w') as f:
                for category, idx in category_mapping.items():
                    f.write(f'{category}\t{idx}\n')

            feature_map[feature] = {
                "type": "categorical",
                "vocab_size": len(category_mapping),
                "feature_dim": self.categorical_features_dim[C_i]
            }

            feature_filter[feature] = f"{feature}:\t\t类别总数:{len(categories_count)},\t\t过滤类别:{len(categories_count) - len(valid_categories)},\t\t超过阈值{threshold_value}的类别:{len(valid_categories)}"

        for k, v in feature_filter.items():
            print(v)

        return feature_map

    def get_all_features_buckets(self, df, bucket_dir, numeric_threshold_value=56):
        """
        对数值型特征进行分桶，统计类别型特征的所有类别
        Args:
            bucket_dir: 存放特征分桶的路径
            numeric_threshold_value: 数值特征分组的阈值. Defaults to 56.
        """
        numeric_feature_map = {}
        if self.is_numeric_bucket:
            numeric_feature_map = self.get_numeric_bucket_threshold(
                df=df,
                bucket_threshold_save_dir=bucket_dir,
                threshold_value=numeric_threshold_value
            )
        categorical_feature_map = self.get_categorical_features(
            df=df,
            categorical_save_dir=bucket_dir
        )
        features_map = {**numeric_feature_map, **categorical_feature_map}

        criteo_feature_info = {
            'dataset_name': 'criteo',
            'is_numeric_bucket': self.is_numeric_bucket,
            'numeric_feature': self.numeric_features,
            'categorical_feature': self.categorical_features,
            'label': self.label,
            'numeric_feature_len': sum(self.numeric_features_dim),
            'categorical_feature_len': sum(self.categorical_features_dim),
            'sample_len': sum(self.numeric_features_dim) + sum(self.categorical_features_dim),
            'features_map': features_map
        }

        feature_map_output_path = os.path.join(self.data_dir, 'feature_map.json')
        with open(feature_map_output_path, 'w') as f:
            json.dump(criteo_feature_info, f, indent=4)

    def load_bucket_info(self, bucket_dir, feature_name, is_numeric):
        """
        加载分桶信息
        Args:
            bucket_dir: 分桶文件目录
            feature_name: 特征名称
            is_numeric: 是否为数值型特征
        Returns:
            buckets: 分桶信息
        """
        bucket_file = os.path.join(bucket_dir, f"{'numeric' if is_numeric else 'categorical'}_{feature_name}.txt")
        buckets = []
        with open(bucket_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if is_numeric:
                    buckets.append((float(parts[0]), float(parts[1]), int(parts[2])))
                else:
                    buckets.append((parts[0], int(parts[1])))
        return buckets

    def original_data_to_bucket(self, df, bucket_dir, output_npz, output_csv):
        """
        将原始数据根据numeric型特征分桶映射为桶序，将类别型特征按类别映射为类别序号
        Args:
            df: 原始样本
            bucket_dir: 特征分桶的目录
            output_npz: 原始数据映射后的保存路径
            output_csv: 原始数据映射后的前100行保存路径(方便查看)
        """
        processed_data = pd.DataFrame()

        for i in range(1, 14):
            feature_name = f'I{i}'
            if self.is_numeric_bucket:
                buckets = self.load_bucket_info(bucket_dir, feature_name, True)
                thresholds = [b[1] for b in buckets[:-1]]
                processed_data[feature_name] = np.searchsorted(thresholds, df[feature_name], side='right')
            else:
                processed_data[feature_name] = df[feature_name].apply(lambda x: int(math.floor(math.log(x, 2))) if x > 2 else 1)

        for i in range(1, 27):
            feature_name = f'C{i}'
            buckets = self.load_bucket_info(bucket_dir, feature_name, False)
            mapping_dict = {k: v for k, v in buckets}
            processed_data[feature_name] = df[feature_name].map(mapping_dict).fillna(mapping_dict.get('others')).astype(int)

        processed_data[self.label] = df[self.label].astype(int)

        np.savez(output_npz, **{col: processed_data[col].to_numpy() for col in processed_data.columns})

        processed_data.head(100).to_csv(output_csv, index=False)

    def process_all_operation(self):
        """
        数据处理集合
        """
        print("\nStep1: 开始读取数据，并进行空值填充...")
        start_time = time.time()
        raw_data = self.get_data_and_null_fill()
        end_time = time.time()
        one_step_time = end_time - start_time

        print("\nStep2: 开始进行分桶处理...")
        start_time = time.time()
        bucket_dir = os.path.join(self.output_save_dir, 'bucket')
        os.makedirs(bucket_dir, exist_ok=True)
        self.get_all_features_buckets(df=raw_data, bucket_dir=bucket_dir, numeric_threshold_value=56)
        end_time = time.time()
        two_step_time = end_time - start_time

        print("\nStep3: 将原始样本值按桶/类别映射为桶序/类别序号...")
        start_time = time.time()
        process_sample_path = os.path.join(self.output_save_dir, self.dataset_save_name)
        process_sample_head_100_path = os.path.join(self.output_save_dir, "process_sample_head_100.csv")
        self.original_data_to_bucket(
            df=raw_data,
            bucket_dir=bucket_dir,
            output_npz=process_sample_path,
            output_csv=process_sample_head_100_path
        )
        end_time = time.time()
        three_step_time = end_time - start_time

        print(f"step1_cost_time: {one_step_time:.2f} seconds")
        print(f"step2_cost_time: {two_step_time:.2f} seconds")
        print(f"step3_cost_time: {three_step_time:.2f} seconds")
        print(f"total_time: {one_step_time + two_step_time + three_step_time:.2f} seconds")


if __name__ == '__main__':
    preprocess = CriteoPreprocess(
        data_dir='../../Dataset/criteo/',  # 原始数据所在目录的相对路径
        data_path='sample.csv',  # 原始数据名
        output_save_dir='../../Dataset/criteo/',  # 输出数据的存放目录
        dataset_save_name='process_sample.npz',  # 输出数据文件名
        is_number_bucket=False  # 数值数据是否分桶
    )
    preprocess.process_all_operation()
