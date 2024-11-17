# Criteo数据预处理详细说明

## 1. Criteo数据集简介

该数据集来自Criteo展示广告的Kaggle挑战。Criteo是一家个性化重新定位公司，与互联网零售商合作，向消费者提供个性化的在线展示广告。这个Kaggle挑战的目标是预测展示广告的点击率。它提供了来自Criteo流量的一周数据。在为期7天的标记训练集中，每一行对应Criteo提供的一个展示广告。样本按时间顺序排列。正负样本都以不同的速率进行了二次采样，以减小数据集的大小。有13个数值特征和26个分类特征，这些特征的语义未公开，一些特征缺少值。

数据集下载可参考：https://github.com/reczoo/Datasets/tree/main/Criteo

以下是Criteo数据示例：

（1）label: 第一行是表示label，1表示点击，0表示不点击

（2）I1～I13: 表示数值型特征

（3）C1~C26: 表示分类型特征

![Criteo数据示例](Critreo数据预处理.assets/image-20241019215056868-9345871.png)



下面将对EasyDeepRecommand项目中的一些重点进行讲解，详情可以见：

[EasyDeepRecommand](https://github.com/Iamctb/EasyDeepRecommand) 中的项目文件[criteo_preprocess.py](https://github.com/Iamctb/EasyDeepRecommand/blob/main/DataProcess/criteo/criteo_preprocess.py)



## 2. 缺失值处理

可参考：[缺失值填补、缺失值填充方法汇总](https://gitcode.csdn.net/65e935fb1a836825ed78d63c.html?dp_token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpZCI6NDQwMzAzLCJleHAiOjE3MzAwMjU4NjUsImlhdCI6MTcyOTQyMTA2NSwidXNlcm5hbWUiOiJxcV80MTkxNTYyMyJ9.dHQruCdSigc47DHVDROf4L3iobUe6NRbz2yNGnVnHkM&spm=1001.2101.3001.6650.15&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7Eactivity-15-103306928-blog-90599619.235%5Ev43%5Epc_blog_bottom_relevance_base5&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7Eactivity-15-103306928-blog-90599619.235%5Ev43%5Epc_blog_bottom_relevance_base5&utm_relevant_index=23)

**Step1: 读取数据**

```
import pandas as pd
raw_data = pd.read_csv(self.data_path)	# 上图那种表格的路径
print("查看数据基本信息：")
print(f"data.shape: {raw_data.shape} \n")
print(f"{raw_data.info()}")
print(f"\n缺失值情况: \n{raw_data.isnull().sum()}")
```

打印结果如下：

```
查看数据基本信息：
data.shape: (10000, 40) 

<class 'pandas.core.frame.DataFrame'>
RangeIndex: 10000 entries, 0 to 9999
Data columns (total 40 columns):
 #   Column  Non-Null Count  Dtype  
---  ------  --------------  -----  
 0   label   10000 non-null  int64  
 1   I1      5519 non-null   float64
 2   I2      10000 non-null  int64  
 3   I3      7963 non-null   float64
 4   I4      8022 non-null   float64
 5   I5      9506 non-null   float64
 6   I6      7489 non-null   float64
 7   I7      9508 non-null   float64
 8   I8      9990 non-null   float64
 9   I9      9508 non-null   float64
 10  I10     5519 non-null   float64
 11  I11     9508 non-null   float64
 12  I12     2265 non-null   float64
 13  I13     8022 non-null   float64
 14  C1      10000 non-null  object 
 15  C2      10000 non-null  object 
 16  C3      9643 non-null   object 
 17  C4      9643 non-null   object 
 18  C5      10000 non-null  object 
 19  C6      8668 non-null   object 
 20  C7      10000 non-null  object 
 21  C8      10000 non-null  object 
 22  C9      10000 non-null  object 
 23  C10     10000 non-null  object 
 24  C11     10000 non-null  object 
 25  C12     9643 non-null   object 
 26  C13     10000 non-null  object 
 27  C14     10000 non-null  object 
 28  C15     10000 non-null  object 
 29  C16     9643 non-null   object 
 30  C17     10000 non-null  object 
 31  C18     10000 non-null  object 
 32  C19     5504 non-null   object 
 33  C20     5504 non-null   object 
 34  C21     9643 non-null   object 
 35  C22     1818 non-null   object 
 36  C23     10000 non-null  object 
 37  C24     9643 non-null   object 
 38  C25     5504 non-null   object 
 39  C26     5504 non-null   object 
dtypes: float64(12), int64(2), object(26)
memory usage: 3.1+ MB
None

缺失值情况: 
label       0
I1       4481
I2          0
I3       2037
I4       1978
I5        494
I6       2511
I7        492
I8         10
I9        492
I10      4481
I11       492
I12      7735
I13      1978
C1          0
C2          0
C3        357
C4        357
C5          0
C6       1332
C7          0
C8          0
C9          0
C10         0
C11         0
C12       357
C13         0
C14         0
C15         0
C16       357
C17         0
C18         0
C19      4496
C20      4496
C21       357
C22      8182
C23         0
C24       357
C25      4496
C26      4496
dtype: int64
```

从上图中也可以看出，无论是数值型特征还是类别型特征，都存在很多空值，针对这些空值，采用如下策略：

**Step2: **：对 **数值型特征** 采用 **前后线性插值填充 **

```
# 数值型特征缺失值处理：使用前后均值
raw_data[num_features] = raw_data[num_features].interpolate(method='linear', limit_direction='both')    
```

**Step3: ** 对 **类别型特征** 采用 **KNN聚类填充**

```
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
        # 创建OrdinalEncoder对象
        encoder = OrdinalEncoder()

        # 使用OrdinalEncoder进行整数编码
        df[columns] = encoder.fit_transform(df[columns])

        # 创建KNNImputer对象
        imputer = KNNImputer(n_neighbors=n_neighbors)
        # 使用KNNImputer进行填充
        df_filled = imputer.fit_transform(df)
        # 将数组转换为DataFrame
        df_filled = pd.DataFrame(df_filled, columns=df.columns)

        # 将编码转换回原始类别
        df_filled[columns] = encoder.inverse_transform(df_filled[columns].astype(int))

        return df_filled

# 类别型特征缺失值处理：使用KNN聚类
raw_data = self.categerical_null_fill(df=raw_data, columns=cate_features, n_neighbors=10)             
```

查看填充后的数据：

```
print(f"\n缺失值处理后情况: \n{raw_data.isnull().sum()}")
```

打印结果如下：

```
缺失值处理后情况: 
label    0
I1       0
I2       0
I3       0
I4       0
I5       0
I6       0
I7       0
I8       0
I9       0
I10      0
I11      0
I12      0
I13      0
C1       0
C2       0
C3       0
C4       0
C5       0
C6       0
C7       0
C8       0
C9       0
C10      0
C11      0
C12      0
C13      0
C14      0
C15      0
C16      0
C17      0
C18      0
C19      0
C20      0
C21      0
C22      0
C23      0
C24      0
C25      0
C26      0
dtype: int64
```



## 3. 分桶处理

**（1）针对数值型特征的分桶处理**

将现有数据从小到大排列，设定桶数量的阈值numeric_threshold_value，从最小值遍历，遍历的数值放到一个桶里面，当一个桶的数量超过等于numeric_threshold_value这个阈值时，就将这个桶里面的最小值和最大值记为桶的上下界；然后再根据此方法获取下一个桶的上下界。特殊地，（1）最后一个桶数量不满numeric_threshold_value时，也划分为同一个桶；（2）将小于当前数据集最小时的值域和大于当前最大值的值域单独划分为一个桶，防止以后在推理时遇到不在当前数据集值域的特殊样本，而无法找到对应的embedding，从而增强模型的稳定性。如：

现在有个数值特征，共有10个样本，样本值分别为：0, 1, 2, 3, 4, 5, 6, 7, 8, 9，设numeric_threshold_value=3；则按上述规则产生如下桶：

bucket_0:[-∞, 0)，表示未来一个样本值没有在当前数据集中，当它小于当前数据集的最小值时，就分为第0桶（第一个桶）；

bucket_1: [0, 2]，表示在值属于[0,2]区间的划分到桶1；

bucket_2: [3, 5]，表示在值属于[3, 5]区间的划分到桶2；

bucket_3: [6, 8]，表示在值属于[6,8]区间的划分到桶3；

bucket_4: [10]，表示在值属于[10]区间的划分到桶4；	

bucket_5: （10，+∞]，表示未来一个样本值没有在当前数据集中，当它大于当前数据集的最小值时，就分到第5桶（最后一个桶）。

**（2）针对类别型特征的分桶处理**

将当前所有出现的类别进行归类，类别数即为分桶数，同时在最后添加一个“others”类，如：
当前有一个类别型特征，共有5个样本，样本值分别为：A，A，B，B，C，则按上述规则，将其划分为：

bucket_0: "A"

bucket_1: "B"

bucket_2: "C"

Bucket_3: "others", 表示未来一个样本值如果没有在当前数据集中，当将它分到“others”桶（最后一个桶）。



## 4. 特征收集

将上述分桶处理的信息进行记录到 [feature_map.json](https://github.com/Iamctb/EasyDeepRecommand/blob/main/Dataset/criteo/feature_map.json) 文件中，记录如下：

```
{
    "dataset_name": "criteo",   // 数据集名称
    "numeric_feature": [				// 数值型特征有哪些
        "I1",
        "I2",
        "I3",
        "I4",
        "I5",
        "I6",
        "I7",
        "I8",
        "I9",
        "I10",
        "I11",
        "I12",
        "I13"
    ],
    "categorical_feature": [		// 列别型特征有哪些
        "C1",
        "C2",
        "C3",
        "C4",
        "C5",
        "C6",
        "C7",
        "C8",
        "C9",
        "C10",
        "C11",
        "C12",
        "C13",
        "C14",
        "C15",
        "C16",
        "C17",
        "C18",
        "C19",
        "C20",
        "C21",
        "C22",
        "C23",
        "C24",
        "C25",
        "C26"
    ],
    "label": "label",								// label在数据集中的列名
    "numeric_feature_len": 52,			// 数值型特征经过embedding后的维度
    "categorical_feature_len": 104,	// 列别型特征经过embedding后的维度
    "sample_len": 156,							// 样本经过embedding后总维度
    "features_map": {								// 所有特征的信息集合
        "I1": {											// 表示I1特征
            "type": "numeric",			// 表示特征类型
            "vocab_size": 46,				// 表示该特征的桶数，后面初始化embedding需要用到
            "feature_dim": 4				// 表示特征embedding后的维度，后面初始化embedding需要用到。 下同，不重复解释
        },
        "I2": {
            "type": "numeric",
            "vocab_size": 73,
            "feature_dim": 4
        },
        "I3": {
            "type": "numeric",
            "vocab_size": 67,
            "feature_dim": 4
        },
        "I4": {
            "type": "numeric",
            "vocab_size": 45,
            "feature_dim": 4
        },
        "I5": {
            "type": "numeric",
            "vocab_size": 168,
            "feature_dim": 4
        },
        "I6": {
            "type": "numeric",
            "vocab_size": 128,
            "feature_dim": 4
        },
        "I7": {
            "type": "numeric",
            "vocab_size": 46,
            "feature_dim": 4
        },
        "I8": {
            "type": "numeric",
            "vocab_size": 52,
            "feature_dim": 4
        },
        "I9": {
            "type": "numeric",
            "vocab_size": 121,
            "feature_dim": 4
        },
        "I10": {
            "type": "numeric",
            "vocab_size": 21,
            "feature_dim": 4
        },
        "I11": {
            "type": "numeric",
            "vocab_size": 21,
            "feature_dim": 4
        },
        "I12": {
            "type": "numeric",
            "vocab_size": 49,
            "feature_dim": 4
        },
        "I13": {
            "type": "numeric",
            "vocab_size": 51,
            "feature_dim": 4
        },
        "C1": {
            "type": "categorical",
            "vocab_size": 175,
            "feature_dim": 4
        },
        "C2": {
            "type": "categorical",
            "vocab_size": 386,
            "feature_dim": 4
        },
        "C3": {
            "type": "categorical",
            "vocab_size": 5520,
            "feature_dim": 4
        },
        "C4": {
            "type": "categorical",
            "vocab_size": 4032,
            "feature_dim": 4
        },
        "C5": {
            "type": "categorical",
            "vocab_size": 56,
            "feature_dim": 4
        },
        "C6": {
            "type": "categorical",
            "vocab_size": 7,
            "feature_dim": 4
        },
        "C7": {
            "type": "categorical",
            "vocab_size": 3184,
            "feature_dim": 4
        },
        "C8": {
            "type": "categorical",
            "vocab_size": 93,
            "feature_dim": 4
        },
        "C9": {
            "type": "categorical",
            "vocab_size": 3,
            "feature_dim": 4
        },
        "C10": {
            "type": "categorical",
            "vocab_size": 2986,
            "feature_dim": 4
        },
        "C11": {
            "type": "categorical",
            "vocab_size": 2084,
            "feature_dim": 4
        },
        "C12": {
            "type": "categorical",
            "vocab_size": 5283,
            "feature_dim": 4
        },
        "C13": {
            "type": "categorical",
            "vocab_size": 1725,
            "feature_dim": 4
        },
        "C14": {
            "type": "categorical",
            "vocab_size": 24,
            "feature_dim": 4
        },
        "C15": {
            "type": "categorical",
            "vocab_size": 2035,
            "feature_dim": 4
        },
        "C16": {
            "type": "categorical",
            "vocab_size": 4723,
            "feature_dim": 4
        },
        "C17": {
            "type": "categorical",
            "vocab_size": 9,
            "feature_dim": 4
        },
        "C18": {
            "type": "categorical",
            "vocab_size": 1149,
            "feature_dim": 4
        },
        "C19": {
            "type": "categorical",
            "vocab_size": 546,
            "feature_dim": 4
        },
        "C20": {
            "type": "categorical",
            "vocab_size": 3,
            "feature_dim": 4
        },
        "C21": {
            "type": "categorical",
            "vocab_size": 5036,
            "feature_dim": 4
        },
        "C22": {
            "type": "categorical",
            "vocab_size": 7,
            "feature_dim": 4
        },
        "C23": {
            "type": "categorical",
            "vocab_size": 12,
            "feature_dim": 4
        },
        "C24": {
            "type": "categorical",
            "vocab_size": 2524,
            "feature_dim": 4
        },
        "C25": {
            "type": "categorical",
            "vocab_size": 39,
            "feature_dim": 4
        },
        "C26": {
            "type": "categorical",
            "vocab_size": 1938,
            "feature_dim": 4
        }
    }
}
```

自此，Criteo数据集重点介绍完毕，详见代码：[EasyDeepRecommand](https://github.com/Iamctb/EasyDeepRecommand) 中的项目文件[criteo_preprocess.py](https://github.com/Iamctb/EasyDeepRecommand/blob/main/DataProcess/criteo/criteo_preprocess.py)

