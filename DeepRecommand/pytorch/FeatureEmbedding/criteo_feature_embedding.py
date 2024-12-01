import torch
import torch.nn as nn

class CriteoFeatureEmbedding(nn.Module):
    def __init__(self, feature_map):
        super(CriteoFeatureEmbedding, self).__init__()

        self.feature_map = feature_map
        self.embedding_layers = self.get_embedding_layer()
        

    def get_embedding_layer(self):
        features_map = self.feature_map['features_map']
        embedding_layers = nn.ModuleDict()
        for feature, feature_conf in features_map.items():
            if feature == "label":
                continue
            embedding_layers[feature] = nn.Embedding(num_embeddings=feature_conf["vocab_size"],
                                                     embedding_dim=feature_conf['feature_dim'])
        return embedding_layers
    

    def forward(self, X):
        """
        _summary_
        Args:
            X: dict, 样本数据，形如：{"I1":[], "I2":[], ...}
        Returns:
            _description_
        """
        data_embedding = []
        for key, indecs in X.items():
            if key == "label":
                continue
            if 'I' in key and not self.feature_map["is_numeric_bucket"]:
                embeddings = indecs.unsqueeze(1).float()
            else:
                embeddings = self.embedding_layers[key](indecs)
            data_embedding.append(embeddings)

        concat_embedding = torch.cat(data_embedding, dim=1)
        return concat_embedding
    

if __name__ == "__main__":
    feature_map = {}
    features_map = {
        "I1": {"vocab_size": 3, "feature_dim": 2},
        "I2": {"vocab_size": 3, "feature_dim": 2}
    }
    feature_map["features_map"] = features_map

    X = {
        "label": [0, 1, 0],
        "I1": [1, 2, 0],
        "I2": [0, 2, 1]
    }

    CriteoFeatureEmbedding = CriteoFeatureEmbedding(feature_map=feature_map)
    res = CriteoFeatureEmbedding(X=X)
    print(res)
    print(res.shape)