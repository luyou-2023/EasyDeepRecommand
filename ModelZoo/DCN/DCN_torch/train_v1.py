import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import roc_auc_score
from datetime import datetime
from tqdm import tqdm
import random

# 将当前路径添加到sys.path，方便引用同目录/子目录的模块
current_dir = os.path.dirname(__file__)                                 # 获取当前脚本所在的目录
project_root = os.path.abspath(os.path.join(current_dir, "../../.."))   # 定位项目根目录
sys.path.append(project_root)                                           # 添加项目根目录到 sys.path

# 引入我们前面定义的 Dataloader 和 DCN
from DeepRecommand.pytorch.dataloader.Criteo_Dataloader_v1 import CriteoDataloader
from model.DCN_v1 import DCN

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def train_and_valid(data_config, feature_map, model_config):
    """
    data_config: dict, 包含数据相关配置，如 train_data/valid_data 路径
    feature_map: dict, 读取 feature_map.json 后得到的信息
    model_config: dict, 训练配置，如 batch_size, epoch, lr等
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 构建 DataLoader
    print("\n[1] 构建训练/验证 Dataloader ...")
    train_loader = CriteoDataloader(
        data_path=data_config['train_data'],
        features_name=feature_map['numeric_feature'] + feature_map['categorical_feature'],
        label_name=feature_map['label'],
        batch_size=model_config['batch_size'],
        shuffle=True,         # 建议训练集打乱
        num_workers=4,        # 可按CPU核心数调整
        pin_memory=True
    )
    valid_loader = CriteoDataloader(
        data_path=data_config['valid_data'],
        features_name=feature_map['numeric_feature'] + feature_map['categorical_feature'],
        label_name=feature_map['label'],
        batch_size=model_config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # 构建模型
    print("[2] 构建 DCN 模型 ...")
    model = DCN(feature_map=feature_map, model_config=model_config).to(device)
    print(model)

    # 构建损失和优化器
    criterion = nn.BCELoss()
    if model_config['optimizer'].lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=model_config['learning_rate'])
    else:
        optimizer = optim.SGD(model.parameters(), lr=model_config['learning_rate'], momentum=0.9)
    # 学习率调度器 (可选)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

    # 混合精度的自动类型转换器
    scaler = torch.cuda.amp.GradScaler(enabled=model_config.get("use_amp", False))

    num_epochs = model_config['epoch']
    best_val_auc = 0.0

    print("\n[3] 开始训练 ...")
    for epoch in range(num_epochs):
        # ------- 训练阶段 -------
        model.train()
        train_labels_gpu = []
        train_preds_gpu = []
        total_train_loss = 0.0
        
        # tqdm仅显示每个epoch的进度，减少刷新频率
        for batch_data in tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] - Training", mininterval=10):
            features = {k: v.to(device, non_blocking=True) 
                        for k, v in batch_data.items() if k != 'label'}
            labels = batch_data['label'].to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=model_config.get("use_amp", False)):
                outputs = model(features).squeeze(-1)
                loss = criterion(outputs, labels)

            # 后向
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_train_loss += loss.item()
            train_preds_gpu.append(outputs.detach())
            train_labels_gpu.append(labels.detach())

        # epoch 结束 => 计算 train AUC
        train_preds_gpu = torch.cat(train_preds_gpu, dim=0)
        train_labels_gpu = torch.cat(train_labels_gpu, dim=0)
        # 移动到 CPU 计算 AUC
        train_preds_cpu = train_preds_gpu.cpu().numpy()
        train_labels_cpu = train_labels_gpu.cpu().numpy()
        train_auc = roc_auc_score(train_labels_cpu, train_preds_cpu)
        train_loss_avg = total_train_loss / len(train_loader)

        # ------- 验证阶段 -------
        model.eval()
        val_labels_gpu = []
        val_preds_gpu = []
        val_loss_total = 0.0
        with torch.no_grad():
            for batch_data in tqdm(valid_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] - Validation", mininterval=10):
                features = {k: v.to(device, non_blocking=True) 
                            for k, v in batch_data.items() if k != 'label'}
                labels = batch_data['label'].to(device, non_blocking=True)

                with torch.cuda.amp.autocast(enabled=model_config.get("use_amp", False)):
                    outputs = model(features).squeeze(-1)
                    loss = criterion(outputs, labels)
                val_loss_total += loss.item()

                val_preds_gpu.append(outputs.detach())
                val_labels_gpu.append(labels.detach())

        val_preds_gpu = torch.cat(val_preds_gpu, dim=0)
        val_labels_gpu = torch.cat(val_labels_gpu, dim=0)
        val_preds_cpu = val_preds_gpu.cpu().numpy()
        val_labels_cpu = val_labels_gpu.cpu().numpy()
        val_auc = roc_auc_score(val_labels_cpu, val_preds_cpu)
        val_loss_avg = val_loss_total / len(valid_loader)

        # 调度学习率
        scheduler.step(val_loss_avg)

        # 打印/记录
        print(f"Epoch [{epoch+1}/{num_epochs}] => "
              f"Train Loss: {train_loss_avg:.4f}, Train AUC: {train_auc:.4f}, "
              f"Val Loss: {val_loss_avg:.4f}, Val AUC: {val_auc:.4f}")

        # 保存最好 AUC 的模型
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_save_path = os.path.join(model_config["model_save_dir"],
                                           f"DCN_best_epoch{epoch+1}_auc_{val_auc:.5f}_{current_time}.pt")
            torch.save(model.state_dict(), model_save_path)
            print(f"  ==> Best model saved at epoch {epoch+1}, val_auc={val_auc:.5f}")

    print("\n训练结束。")

if __name__ == '__main__':
    set_seed(2024)

    # 示例：从同目录 config 里读 json
    data_config_path = os.path.join(current_dir, "config/data_config.json")
    model_config_path = os.path.join(current_dir, "config/model_config.json")

    with open(data_config_path, 'r') as file:
        data_config = json.load(file)

    with open(model_config_path, 'r') as file:
        model_config = json.load(file)

    # 获取配置中的路径，和项目路径合并后重新赋值，防止因为路径导致程序出错
    data_config['feature_map'] = project_root + data_config['feature_map']
    data_config['train_data'] = project_root + data_config['train_data']
    data_config['valid_data'] = project_root + data_config['valid_data']
    data_config['test_data'] = project_root + data_config['test_data']
    model_config['model_save_dir'] = project_root + model_config['model_save_dir']

    print("data_config: \n", json.dumps(data_config, indent=4))
    print("model_config: \n", json.dumps(model_config, indent=4))

    # 读取 feature_map.json
    feature_map_path = data_config['feature_map']
    with open(feature_map_path, 'r') as file:
        feature_map = json.load(file)

    # 若想使用混合精度加速，在 model_config.json 中加入 "use_amp": true
    train_and_valid(data_config=data_config,
                    feature_map=feature_map,
                    model_config=model_config)


