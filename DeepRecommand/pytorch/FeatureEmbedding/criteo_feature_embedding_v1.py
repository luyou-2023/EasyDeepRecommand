import torch
import torch.nn as nn

class CriteoFeatureEmbedding(nn.Module):
    def __init__(self, feature_map):
        super(CriteoFeatureEmbedding, self).__init__()
        self.feature_map = feature_map
        # 用于创建 embedding
        self.embedding_layers = nn.ModuleDict()
        self._build_embedding()

    def _build_embedding(self):
        features_map = self.feature_map['features_map']
        for feature, feature_conf in features_map.items():
            if feature == "label":
                continue
            # 如果是 numeric 且不分桶，可以直接用1维 embedding 或者跳过
            # 这里保留原逻辑：numeric也走embedding(如果 is_numeric_bucket=True)
            self.embedding_layers[feature] = nn.Embedding(
                num_embeddings=feature_conf["vocab_size"],
                embedding_dim=feature_conf['feature_dim']
            )

    def forward(self, X):
        """
        Args:
            X: dict[str, Tensor], 每个 key 对应一个特征的 index/id
        Returns:
            concat_embedding: Tensor, shape=(batch_size, sum_of_feature_dim)
        """
        data_embedding = []
        for key, value in X.items():
            if key == "label":
                continue
            # 注意：value 可能是 long dtype, embedding lookup时需要 long
            embed = self.embedding_layers[key](value)
            data_embedding.append(embed)
        # dim=1 拼接：例如所有Embedding都以 [batch, dim] 形式拼在第二维
        concat_embedding = torch.cat(data_embedding, dim=1)  
        return concat_embedding
