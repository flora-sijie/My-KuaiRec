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

# ================= 配置区 =================
class Config:
    dataset_path = './processed_data/dataset.pkl'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 基础训练参数
    epochs = 50
    batch_size = 2048
    lr_worker = 0.001
    lr_manager = 0.005 # Manager 通常需要更大的更新幅度
    weight_decay = 1e-4
    
    # Worker 参数
    emb_dim = 64
    n_layers = 3
    cl_rate = 0.2
    eps = 0.1
    temp = 0.2
    
    # Manager 参数 (生态调节)
    eco_lambda = 0.5   # 生态 Loss 的权重 (论文中的 lambda)
    target_exposure = 0.3 # 目标长尾曝光率 (论文中的 tau)
    
    # 评估
    top_k = 20

config = Config()
print(f"🚀 Phase 2: Full Model Training on {config.device}")

# ================= 1. 数据集 (Dataset) =================
class RecDataset(Dataset):
    def __init__(self, conf):
        with open(conf.dataset_path, 'rb') as f:
            data = pickle.load(f)
            
        self.num_users = data['num_users']
        self.num_items = data['num_items']
        self.num_authors = data['num_authors']
        self.num_active_levels = data['num_active_levels']
        
        # 核心映射
        self.item2author = torch.LongTensor(data['item2author']).to(conf.device)
        self.author_groups = torch.LongTensor(data['author_groups']).to(conf.device)
        self.user_active = torch.LongTensor(data['user_active_feature']).to(conf.device)
        
        # 生成 Item-level 的 Tail Mask (方便查询)
        # author_groups: 0=Tail, 1=Mid, 2=Head
        # 我们需要知道每个 item 是不是 Tail
        # 逻辑: item -> author -> group
        item_groups = self.author_groups[self.item2author]
        self.is_tail_item = (item_groups == 0).float() # [num_items] 1.0 if tail
        
        self.train_pairs = np.array(data['train_pairs'])
        self.test_pairs = data['test_pairs']
        
        self.train_dict = {}
        for u, i in self.train_pairs:
            if u not in self.train_dict: self.train_dict[u] = []
            self.train_dict[u].append(i)
        self.test_dict = {}
        for u, i in self.test_pairs:
            if u not in self.test_dict: self.test_dict[u] = []
            self.test_dict[u].append(i)
            
        self.graph = self._build_sparse_graph()

    def _build_sparse_graph(self):
        u_ids = self.train_pairs[:, 0]
        i_ids = self.train_pairs[:, 1]
        values = np.ones(len(self.train_pairs), dtype=np.float32)
        R = sp.coo_matrix((values, (u_ids, i_ids)), shape=(self.num_users, self.num_items))
        
        vals = np.concatenate([values, values])
        rows = np.concatenate([u_ids, i_ids + self.num_users])
        cols = np.concatenate([i_ids + self.num_users, u_ids])
        adj_shape = self.num_users + self.num_items
        adj = sp.coo_matrix((vals, (rows, cols)), shape=(adj_shape, adj_shape))
        
        rowsum = np.array(adj.sum(1))
        d_inv = np.power(rowsum + 1e-9, -0.5).flatten() # Fix divide by zero
        d_mat_inv = sp.diags(d_inv)
        norm_adj = d_mat_inv.dot(adj).dot(d_mat_inv)
        
        norm_adj = norm_adj.tocoo()
        i = torch.LongTensor(np.vstack((norm_adj.row, norm_adj.col)))
        v = torch.FloatTensor(norm_adj.data)
        return torch.sparse_coo_tensor(i, v, torch.Size(norm_adj.shape)).to(config.device)

    def __len__(self): return len(self.train_pairs)
    def __getitem__(self, idx): return self.train_pairs[idx][0], self.train_pairs[idx][1]

# ================= 2. 模型定义 (Worker + Manager) =================

# --- Part B: The Manager ---
class ManagerNetwork(nn.Module):
    def __init__(self, num_active_levels, emb_dim=32):
        super(ManagerNetwork, self).__init__()
        # Input: User Active Level (Embedding)
        self.active_emb = nn.Embedding(num_active_levels, emb_dim)
        
        # MLP: State -> Weight
        self.net = nn.Sequential(
            nn.Linear(emb_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
            nn.Sigmoid() # 输出 0~1 之间的扶持力度
        )
        
    def forward(self, active_level_ids):
        # active_level_ids: [batch_size]
        emb = self.active_emb(active_level_ids)
        weight = self.net(emb) # [batch_size, 1]
        return weight

# --- Part A: The Worker (CreatorXSimGCL) ---
class CreatorXSimGCL(nn.Module):
    def __init__(self, dataset, conf):
        super(CreatorXSimGCL, self).__init__()
        self.num_users = dataset.num_users
        self.num_items = dataset.num_items
        self.num_authors = dataset.num_authors
        self.graph = dataset.graph
        self.eps = conf.eps
        self.n_layers = conf.n_layers
        
        self.item2author_map = dataset.item2author
        
        self.user_emb = nn.Embedding(self.num_users, conf.emb_dim)
        self.item_emb = nn.Embedding(self.num_items, conf.emb_dim)
        self.author_emb = nn.Embedding(self.num_authors, conf.emb_dim)
        
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)
        nn.init.xavier_uniform_(self.author_emb.weight)

    def forward(self, perturbed=False):
        # Creator-Aware Fusion
        author_feats = self.author_emb(self.item2author_map)
        mixed_item_emb = self.item_emb.weight + author_feats
        
        ego_embeddings = torch.cat([self.user_emb.weight, mixed_item_emb], dim=0)
        all_embeddings = []
        
        for k in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(self.graph, ego_embeddings)
            if perturbed:
                noise = torch.rand_like(ego_embeddings)
                ego_embeddings += torch.sign(ego_embeddings) * F.normalize(noise, dim=-1) * self.eps
            all_embeddings.append(ego_embeddings)
            
        final_embeddings = torch.stack(all_embeddings, dim=1).mean(dim=1)
        return torch.split(final_embeddings, [self.num_users, self.num_items])

# ================= 3. 联合训练框架 =================
class JointModel(nn.Module):
    def __init__(self, dataset, conf):
        super(JointModel, self).__init__()
        self.worker = CreatorXSimGCL(dataset, conf)
        self.manager = ManagerNetwork(dataset.num_active_levels)
        self.dataset = dataset
        self.conf = conf
        
    def predict(self, u_batch, i_batch, use_manager=True):
        # 1. Worker Score (Base)
        users_emb, items_emb = self.worker() # No perturbation during inference
        
        u_e = users_emb[u_batch]
        i_e = items_emb[i_batch]
        base_score = (u_e * i_e).sum(dim=1)
        
        if not use_manager:
            return base_score
        
        # 2. Manager Boost
        # 获取用户活跃度
        u_active = self.dataset.user_active[u_batch]
        boost_weight = self.manager(u_active).squeeze() # [batch]
        
        # 获取物品是否为 Tail
        is_tail = self.dataset.is_tail_item[i_batch] # [batch] 0 or 1
        
        # 最终得分 = 基础分 + 扶持分
        # 注意：这里我们让 Manager 决定"是否"扶持以及"扶持多少"
        # 只有当物品是 Tail 时，扶持才生效
        final_score = base_score + (boost_weight * is_tail)
        
        return final_score

# ================= 4. Loss Functions =================
def cal_bpr_loss(scores_pos, scores_neg):
    # 基础 BPR: log(sigmoid(pos - neg))
    loss = -torch.log(torch.sigmoid(scores_pos - scores_neg) + 1e-8)
    return loss.mean()

def cal_infonce_loss(view1, view2, temp):
    view1 = F.normalize(view1, dim=1)
    view2 = F.normalize(view2, dim=1)
    pos_score = (view1 * view2).sum(dim=1) / temp
    pos_score = torch.exp(pos_score)
    ttl_score = torch.matmul(view1, view2.transpose(0, 1)) / temp
    ttl_score = torch.exp(ttl_score).sum(dim=1)
    return -torch.log(pos_score / ttl_score + 1e-8).mean()

def cal_eco_loss(scores_all_items, is_tail_mask, target_exposure, k=20):
    # 这是一个简化的生态 Loss 实现
    # 我们希望 Top-K 中 Tail 的比例接近 target_exposure
    # 但由于 argmax/topk 不可导，我们通常用 Softmax 近似
    
    # 这里的实现比较 Trick：我们只惩罚"Manager没有给Tail足够分值"的情况
    # 但在 Batch 训练中，计算全局 TopK 太慢。
    # 替代方案：最大化 (Manager_Weight * Tail_Items) 的均值，直至达到阈值
    # 或者：使用 Pairwise 思想，如果 pos 是 tail，neg 是 head，且 score_pos < score_neg，则大力惩罚
    
    # 这里我们采用论文中提到的：Expected Exposure Loss
    # prob = softmax(scores / temp)
    # exposure = sum(prob * is_tail)
    # loss = max(0, target - exposure)
    
    # 为了显存，我们只在一个随机采样的小子集上算 softmax
    probs = F.softmax(scores_all_items, dim=1)
    expected_tail_exposure = (probs * is_tail_mask).sum(dim=1).mean()
    
    loss = F.relu(target_exposure - expected_tail_exposure)
    return loss

# ================= 5. 训练与评估 Loop =================
def evaluate(model, dataset, top_k=20):
    model.eval()
    RECALL, NDCG, TAIL_RATIO = [], [], []
    test_users = list(dataset.test_dict.keys())
    
    with torch.no_grad():
        # 预计算所有 User 和 Item Embedding
        all_users, all_items = model.worker()
        
        for start in range(0, len(test_users), 100):
            end = min(start + 100, len(test_users))
            batch_u_ids = torch.LongTensor(test_users[start:end]).to(config.device)
            
            # --- Inference Logic ---
            # 1. Base Scores
            batch_u_emb = all_users[batch_u_ids]
            scores = torch.matmul(batch_u_emb, all_items.transpose(0, 1))
            
            # 2. Manager Boost (广播机制)
            u_active = dataset.user_active[batch_u_ids]
            weights = model.manager(u_active) # [batch, 1]
            is_tail = dataset.is_tail_item.unsqueeze(0) # [1, n_items]
            
            # Final Scores
            scores = scores + (weights * is_tail)
            
            # Mask train
            for i, u_id in enumerate(batch_u_ids.cpu().numpy()):
                train_pos = dataset.train_dict.get(u_id, [])
                scores[i, train_pos] = -1e9
            
            # TopK
            _, indices = torch.topk(scores, top_k, dim=1)
            indices = indices.cpu().numpy()
            
            # Metrics
            for i, u_id in enumerate(batch_u_ids.cpu().numpy()):
                ground_truth = set(dataset.test_dict[u_id])
                hit = 0
                dcg = 0
                idcg = 0
                tail_cnt = 0
                
                for j, item_idx in enumerate(indices[i]):
                    if item_idx in ground_truth:
                        hit += 1
                        dcg += 1.0 / np.log2(j + 2)
                    # 统计推荐列表里的长尾含量
                    if dataset.is_tail_item[item_idx] == 1:
                        tail_cnt += 1
                        
                    if j < len(ground_truth):
                        idcg += 1.0 / np.log2(j + 2)
                        
                RECALL.append(hit / len(ground_truth))
                NDCG.append(dcg / idcg if idcg > 0 else 0)
                TAIL_RATIO.append(tail_cnt / top_k)
                
    return np.mean(RECALL), np.mean(NDCG), np.mean(TAIL_RATIO)

if __name__ == '__main__':
    dataset = RecDataset(config)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    
    model = JointModel(dataset, config).to(config.device)
    optimizer = optim.Adam([
        {'params': model.worker.parameters(), 'lr': config.lr_worker},
        {'params': model.manager.parameters(), 'lr': config.lr_manager}
    ], weight_decay=config.weight_decay)
    
    print("🔥 Start Joint Training (Worker + Manager)...")
    
    for epoch in range(config.epochs):
        model.train()
        total_loss = 0
        
        for batch_u, batch_pos_i in dataloader:
            batch_u = batch_u.to(config.device)
            batch_pos_i = batch_pos_i.to(config.device)
            batch_neg_i = torch.randint(0, dataset.num_items, batch_pos_i.shape).to(config.device)
            
            # --- 1. Forward Pass (Accuracy) ---
            # 这里的 BPR Loss 需要基于"最终得分"计算，也就是 Manager 干预后的得分
            score_pos = model.predict(batch_u, batch_pos_i, use_manager=True)
            score_neg = model.predict(batch_u, batch_neg_i, use_manager=True)
            
            acc_loss = cal_bpr_loss(score_pos, score_neg)
            
            # --- 2. Contrastive Loss (Worker Only) ---
            u_v1, i_v1 = model.worker(perturbed=True)
            u_v2, i_v2 = model.worker(perturbed=True)
            cl_loss = config.cl_rate * (
                cal_infonce_loss(u_v1[batch_u], u_v2[batch_u], config.temp) +
                cal_infonce_loss(i_v1[batch_pos_i], i_v2[batch_pos_i], config.temp)
            )
            
            # --- 3. Ecosystem Loss (Manager Only) ---
            # 为了让 Manager 真的工作，我们需要惩罚它如果不推长尾
            # 我们随机采样一些用户，看他们的 TopK 推荐里长尾够不够
            # (由于计算量大，这里简化为：如果 Positive Sample 是 Tail，则给额外奖励)
            # 或者直接用 Weights 的 L2 正则，防止它变得无限大
            
            # 简易版 Eco Loss: 强迫 Manager 输出的 weight 均值接近 0.5 (表示至少要有一半力度)
            # 或者是根据 batch 内实际的 tail 曝光来算
            
            # 这里我们用一个基于 Margin 的 Loss：
            # 如果 pos item 是 Tail，我们希望 score_pos 越大越好
            is_tail_pos = dataset.is_tail_item[batch_pos_i]
            # eco_loss = -torch.mean(score_pos * is_tail_pos) * 0.1 # 简单激励
            
            # 更高级：Hinge Loss
            # 确保 Tail Items 的得分有一个底线
            eco_loss = 0
            if is_tail_pos.sum() > 0:
                eco_loss = F.relu(1.0 - score_pos[is_tail_pos.bool()]).mean() * config.eco_lambda
            
            loss = acc_loss + cl_loss + eco_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        # Eval
        recall, ndcg, tail_ratio = evaluate(model, dataset)
        print(f"Epoch {epoch+1:2d} | Loss: {total_loss/len(dataloader):.4f} | "
              f"Recall: {recall:.4f} | NDCG: {ndcg:.4f} | "
              f"TailRatio: {tail_ratio:.4f}") # 关注这个指标！