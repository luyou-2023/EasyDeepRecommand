<div align="center">
<img src="https://github.com/Iamctb/EasyDeepRecommand/blob/main/ResyDeepRecommand_logo.png" alt="Logo" width="360"/>
</div>

<div align="center">
<a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.8%2B-blue" style="max-width: 100%;" alt="python version"></a >
<a href="https://pytorch.org/"><img src="https://img.shields.io/badge/pytorch-1.13%2B-blue" style="max-width: 100%;" alt="pytorch version"></a >
<a href="https://www.tensorflow.org/?hl=zh-cn"><img src="https://img.shields.io/badge/tensorflow-2.1+-blue" style="max-width: 100%;" alt="tensorflow version"></a >
<a href="https://blog.csdn.net/qq_41915623/article/details/138839827?fromshare=blogdetail&sharetype=blogdetail&sharerId=138839827&sharerefer=PC&sharesource=qq_41915623&sharefrom=from_link"><img src="https://img.shields.io/badge/CSDN-Blog-red" style="max-width: 100%;" alt="Downloads"></a >
<a href="https://juejin.cn/post/7424903278063140898"><img src="https://img.shields.io/badge/JueJin-Blog-red" style="max-width: 100%;" alt="Downloads"></a >
<a href="https://www.apache.org/licenses/LICENSE-2.0.html"><img src="https://img.shields.io/badge/lisence-Apache_2.0-green" style="max-width: 100%;" alt="License"></a >
</div>
<hr/>

<div align="center">
<a href="https://github.com/Iamctb/EasyDeepRecommand/stargazers"><img src="http://bytecrank.com/nastyox/reporoster/php/stargazersSVG.php?user=Iamctb&repo=EasyDeepRecommand" width="600"/><a/>
</div>


一个通俗易懂的开源推荐系统（A user-friendly open-source project for recommendation systems）.

本项目将结合：**代码、数据流转图、博客、模型发展史** 等多个方面通俗易懂地讲解经典推荐模型，让读者通过一个项目了解推荐系统概况！

持续更新中..., 欢迎star🌟, 第一时间获取更新!!!

## Features

- 分类解析推荐模型：特征交叉模型、多任务模型、行为序列模型等
- 通过blog详细解释模型/论文
- 提供模型间的关系递进图：介绍模型前后的因果关系
- 代码有详细注释，包含详细的数据处理模块


## Dataset

| Name   | Preprocess_url                                               | Download                                                     | Progress |
| ------ | ------------------------------------------------------------ | ------------------------------------------------------------ | -------- |
| Criteo | [criteo_preprocess.py](https://github.com/Iamctb/EasyDeepRecommand/blob/main/DataProcess/criteo/criteo_preprocess.py): 预处理源代码 | [Download_URL](https://github.com/reczoo/Datasets/tree/main/Criteo) | ✅ |
|        | [预处理说明](https://github.com/Iamctb/EasyDeepRecommand/blob/main/DataProcess/criteo/readme_about_criteo_preprocess.md) |                                                              |          |



## Model_Zoo

| No.  | Publication | Model    | Blog                                                         | Paper                                                        | Version |
| ---- | ----------- | -------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------- |
| 1    | DLRS'16     | WideDeep | [📝WideDeep](https://blog.csdn.net/qq_41915623/article/details/138839827?fromshare=blogdetail&sharetype=blogdetail&sharerId=138839827&sharerefer=PC&sharesource=qq_41915623&sharefrom=from_link) | [Wide & Deep Learning for Recommender Systems](https://arxiv.org/pdf/1606.07792.pdf), **Google** | ✅ torch   |
| 2    | ADKDD'17    | DCN      |                                                              | [Deep & Cross Network for Ad Click Predictions](https://arxiv.org/abs/1708.05123), **Google** | ✅ torch   |



## Dependencies

本项目环境主要有：

- python=3.8.20
- pytorch=1.13.0

其余安装包可以使用下面命令安装：

```
pip install -r requirements.txt
```



## Quick Start

以Criteo数据集和WideDeep举例：

***Step1:***  数据预处理

```python
cd DataProcess/criteo
python criteo_preprocess.py
```

样本数据是使用的Criteo一万条数据作为示例，在执行命令过程中，需要注意 **数据集的路径**

***Step2:*** 训练模型

在 [data_config.json](https://github.com/Iamctb/EasyDeepRecommand/blob/main/ModelZoo/WideDeep/WideDeep_torch/config/data_config.json) 中配置数据集路径；

在 [model_config.json](https://github.com/Iamctb/EasyDeepRecommand/blob/main/ModelZoo/WideDeep/WideDeep_torch/config/model_config.json) 中配置模型信息；

然后运行下面命令即可：

```python
cd ModelZoo/WideDeep/WideDeep_torch
python train.py
```

## 最后
如果你觉得还不错的话，请帮忙点个star🌟吧，感谢感谢！！！
If you think it's good, please help out with a star🌟, thank you !!!
