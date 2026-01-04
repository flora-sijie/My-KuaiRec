import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import scipy.sparse as sp
import os
import time

# ================= 配置与超参数 =================
class Config:
    # 路径配置
    dataset_path = './processed_data/dataset.pkl'
    
    # 训练参数
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    epochs = 50
    batch_size = 2048
    learning_rate = 0.001
    weight_decay = 1e-4
    
    # 模型参数 (XSimGCL)
    emb_dim = 64
    n_layers = 3
    cl_rate = 0.2    # 对比学习权重 lambda
    eps = 0.1        # 噪声比例 epsilon
    temp = 0.2       # 温度系数 tau
    
    # 评估参数
    top_k = 20

config = Config()
print(f"🚀 运行环境: {config.device}")

# ================= 1. 数据集加载与图构建 =================
class RecDataset(Dataset):
    def __init__(self, conf):
        print(f"Loading data from {conf.dataset_path}...")
        with open(conf.dataset_path, 'rb') as f:
            data = pickle.load(f)
            
        self.num_users = data['num_users']
        self.num_items = data['num_items']
        self.num_authors = data['num_authors']
        
        # 核心映射：Item -> Author (numpy -> tensor)
        self.item2author = torch.LongTensor(data['item2author']).to(conf.device)
        
        # 训练数据
        self.train_pairs = np.array(data['train_pairs'])
        self.test_pairs = data['test_pairs']
        
        # 转换成 User-Item 字典供快速查找（用于负采样和测试）
        self.train_dict = {}
        for u, i in self.train_pairs:
            if u not in self.train_dict: self.train_dict[u] = []
            self.train_dict[u].append(i)
            
        self.test_dict = {}
        for u, i in self.test_pairs:
            if u not in self.test_dict: self.test_dict[u] = []
            self.test_dict[u].append(i)
            
        # 构建稀疏邻接矩阵 (用于 GNN)
        self.graph = self._build_sparse_graph()

    def _build_sparse_graph(self):
        print("构建稀疏邻接矩阵 (Normalized Adjacency Matrix)...")
        # 1. 构建交互矩阵 R (User x Item)
        u_ids = self.train_pairs[:, 0]
        i_ids = self.train_pairs[:, 1]
        values = np.ones(len(self.train_pairs), dtype=np.float32)
        
        R = sp.coo_matrix((values, (u_ids, i_ids)), shape=(self.num_users, self.num_items))
        
        # 2. 构建大邻接矩阵 A
        # [0, R]
        # [R.T, 0]
        vals = np.concatenate([values, values])
        rows = np.concatenate([u_ids, i_ids + self.num_users])
        cols = np.concatenate([i_ids + self.num_users, u_ids])
        
        adj_shape = self.num_users + self.num_items
        adj = sp.coo_matrix((vals, (rows, cols)), shape=(adj_shape, adj_shape))
        
        # 3. 归一化 D^-0.5 * A * D^-0.5
        rowsum = np.array(adj.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        
        norm_adj = d_mat_inv.dot(adj).dot(d_mat_inv)
        
        # 4. 转为 PyTorch Sparse Tensor
        norm_adj = norm_adj.tocoo()
        indices = np.vstack((norm_adj.row, norm_adj.col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(norm_adj.data)
        shape = norm_adj.shape
        
        graph = torch.sparse.FloatTensor(i, v, torch.Size(shape)).to(config.device)
        return graph

    def __len__(self):
        return len(self.train_pairs)
    
    def __getitem__(self, idx):
        # 简单的正样本提取，负采样在 collate_fn 或 训练循环里做更高效
        # 这里为了配合 DataLoader，我们返回 user, pos_item
        u, i = self.train_pairs[idx]
        return u, i

# ================= 2. 模型定义 (CreatorXSimGCL) =================
class CreatorXSimGCL(nn.Module):
    def __init__(self, dataset, conf):
        super(CreatorXSimGCL, self).__init__()
        self.num_users = dataset.num_users
        self.num_items = dataset.num_items
        self.num_authors = dataset.num_authors
        self.graph = dataset.graph
        
        self.emb_dim = conf.emb_dim
        self.n_layers = conf.n_layers
        self.eps = conf.eps
        self.item2author_map = dataset.item2author
        
        # Embeddings
        self.user_emb = nn.Embedding(self.num_users, self.emb_dim)
        self.item_emb = nn.Embedding(self.num_items, self.emb_dim)
        self.author_emb = nn.Embedding(self.num_authors, self.emb_dim)
        
        # 初始化
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)
        nn.init.xavier_uniform_(self.author_emb.weight)

    def forward(self, perturbed=False):
        # 1. 融合创作者信息 (Part A 核心创新)
        # A：Creator-Aware
        # item_emb = item_id_emb + author_id_emb
        #author_feats = self.author_emb(self.item2author_map) # [num_items, dim]
        #mixed_item_emb = self.item_emb.weight + author_feats
        
        # B：baseline-XSimGCL
        mixed_item_emb = self.item_emb.weight  # 直接用 ID Embedding
        
        # 2. 拼接初始特征
        ego_embeddings = torch.cat([self.user_emb.weight, mixed_item_emb], dim=0)
        all_embeddings = []
        
        # 3. 图卷积传播
        for k in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(self.graph, ego_embeddings)
            
            # XSimGCL 核心：加入随机噪声
            if perturbed:
                noise = torch.rand_like(ego_embeddings).to(ego_embeddings.device)
                # sign(E) * noise * eps
                ego_embeddings += torch.sign(ego_embeddings) * F.normalize(noise, dim=-1) * self.eps
                
            all_embeddings.append(ego_embeddings)
            
        # 4. 聚合层 (Mean Pooling)
        final_embeddings = torch.stack(all_embeddings, dim=1)
        final_embeddings = torch.mean(final_embeddings, dim=1)
        
        # 5. 拆分回 User 和 Item
        users, items = torch.split(final_embeddings, [self.num_users, self.num_items])
        return users, items

# ================= 3. 工具函数 (Loss & Evaluation) =================
def cal_bpr_loss(user_emb, pos_item_emb, neg_item_emb):
    pos_score = torch.mul(user_emb, pos_item_emb).sum(dim=1)
    neg_score = torch.mul(user_emb, neg_item_emb).sum(dim=1)
    loss = -torch.log(torch.sigmoid(pos_score - neg_score) + 1e-8)
    return torch.mean(loss)

def cal_infonce_loss(view1, view2, temperature):
    # InfoNCE Loss: L = -log( exp(sim(v1, v2)/t) / sum(exp(sim(v1, all)/t)) )
    # 为了简化计算，通常使用 Batch 内负采样
    view1 = F.normalize(view1, dim=1)
    view2 = F.normalize(view2, dim=1)
    
    pos_score = (view1 * view2).sum(dim=1) / temperature
    pos_score = torch.exp(pos_score)
    
    # 矩阵乘法计算 Batch 内所有相似度
    ttl_score = torch.matmul(view1, view2.transpose(0, 1)) / temperature
    ttl_score = torch.exp(ttl_score).sum(dim=1)
    
    loss = -torch.log(pos_score / ttl_score + 1e-8)
    return torch.mean(loss)

def evaluate(model, dataset, top_k=20):
    model.eval()
    NDCG, RECALL = [], []
    test_users = list(dataset.test_dict.keys())
    
    with torch.no_grad():
        # 获取最终的 User 和 Item Embedding (不加噪声)
        all_users, all_items = model(perturbed=False)
        
        # 分批次测试防止显存爆炸
        batch_size = 100
        for start in range(0, len(test_users), batch_size):
            end = min(start + batch_size, len(test_users))
            batch_u_ids = test_users[start:end]
            
            # 获取当前 Batch User 的向量
            batch_u_emb = all_users[batch_u_ids]
            
            # 计算所有 Item 的分数
            scores = torch.matmul(batch_u_emb, all_items.transpose(0, 1))
            
            # Mask 掉训练集中已经看过的物品 (防止作弊)
            for i, u_id in enumerate(batch_u_ids):
                train_pos = dataset.train_dict.get(u_id, [])
                scores[i, train_pos] = -1e9 # 设置为极小值
            
            # Top-K 排序
            _, indices = torch.topk(scores, top_k, dim=1)
            indices = indices.cpu().numpy()
            
            # 计算指标
            for i, u_id in enumerate(batch_u_ids):
                ground_truth = set(dataset.test_dict[u_id])
                hit = 0
                idcg = 0
                dcg = 0
                
                for j, item_idx in enumerate(indices[i]):
                    if item_idx in ground_truth:
                        hit += 1
                        dcg += 1.0 / np.log2(j + 2)
                    idcg += 1.0 / np.log2(j + 2)
                
                # 只有当 ground_truth 里的数量少于 k 时，IDCG 才是部分和
                # 准确的 IDCG 应该是前 min(len(gt), k) 个位置为 1
                real_idcg = 0
                for j in range(min(len(ground_truth), top_k)):
                    real_idcg += 1.0 / np.log2(j + 2)
                    
                RECALL.append(hit / len(ground_truth))
                NDCG.append(dcg / real_idcg if real_idcg > 0 else 0)
                
    return np.mean(RECALL), np.mean(NDCG)

# ================= 4. 主训练循环 (Main) =================
if __name__ == '__main__':
    # 1. 初始化
    dataset = RecDataset(config)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    
    model = CreatorXSimGCL(dataset, config).to(config.device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    
    print(f"🔥 开始训练 (Epochs={config.epochs})...")
    
    best_recall = 0
    
    for epoch in range(config.epochs):
        model.train()
        total_loss = 0
        
        start_time = time.time()
        
        for batch_u, batch_pos_i in dataloader:
            batch_u = batch_u.to(config.device)
            batch_pos_i = batch_pos_i.to(config.device)
            
            # 2. 负采样 (简单的随机采样)
            batch_neg_i = torch.randint(0, dataset.num_items, batch_pos_i.shape).to(config.device)
            # (严谨的做法应该检查 neg 是否在 train_dict 中，这里为了速度略过)
            
            # 3. 计算推荐 Loss (Clean View)
            users_emb, items_emb = model(perturbed=False)
            
            u_e = users_emb[batch_u]
            pos_i_e = items_emb[batch_pos_i]
            neg_i_e = items_emb[batch_neg_i]
            
            rec_loss = cal_bpr_loss(u_e, pos_i_e, neg_i_e)
            
            # 4. 计算对比学习 Loss (XSimGCL)
            # 生成两个有噪声的视图
            users_view1, items_view1 = model(perturbed=True)
            users_view2, items_view2 = model(perturbed=True)
            
            # 只计算当前 Batch 涉及节点的 CL Loss，减少计算量
            # User CL
            cl_loss_u = cal_infonce_loss(users_view1[batch_u], users_view2[batch_u], config.temp)
            # Item CL (Pos Items)
            cl_loss_i = cal_infonce_loss(items_view1[batch_pos_i], items_view2[batch_pos_i], config.temp)
            
            cl_loss = config.cl_rate * (cl_loss_u + cl_loss_i)
            
            # 5. 反向传播
            loss = rec_loss + cl_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        # 每个 Epoch 结束后评估
        recall, ndcg = evaluate(model, dataset, config.top_k)
        print(f"Epoch {epoch+1:2d} | Loss: {total_loss/len(dataloader):.4f} | "
              f"Recall@{config.top_k}: {recall:.4f} | NDCG@{config.top_k}: {ndcg:.4f} | "
              f"Time: {time.time()-start_time:.1f}s")
        
        if recall > best_recall:
            best_recall = recall
            # 保存模型 (可选)
            # torch.save(model.state_dict(), 'best_model.pth')

    print(f"\n✅ 训练完成！Best Recall: {best_recall:.4f}")