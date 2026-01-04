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

# ================= 0. 配置 (Config) =================
class Config:
    dataset_path = './processed_data/dataset.pkl'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 训练参数
    epochs = 50
    batch_size = 2048
    lr_worker = 0.001
    lr_manager = 0.001 
    weight_decay = 1e-4
    
    # Worker 参数
    emb_dim = 64
    n_layers = 3
    cl_rate = 0.2
    eps = 0.1
    temp = 0.2
    
    # Manager & Social 参数
    eco_lambda = 0.5    # 生态 Loss 权重
    manager_scale = 0.1 # Part B: 防止 Manager 过度干预
    social_scale = 2.0  # [新增] Part C: 社交加分权重 (给得高一点，确保挤进 Top-20)
    
    # 评估
    top_k = 20

config = Config()
print(f"🚀 Full Model (Part A+B+C) running on {config.device}")

# ================= 1. 数据集 (Dataset) =================
class RecDataset(Dataset):
    def __init__(self, conf):
        with open(conf.dataset_path, 'rb') as f:
            data = pickle.load(f)
            
        self.num_users = data['num_users']
        self.num_items = data['num_items']
        self.num_authors = data['num_authors']
        self.num_active_levels = data['num_active_levels']
        
        # 核心数据转 Tensor
        self.item2author = torch.LongTensor(data['item2author']).to(conf.device)
        self.author_groups = torch.LongTensor(data['author_groups']).to(conf.device)
        self.user_active = torch.LongTensor(data['user_active_feature']).to(conf.device)
        
        # 标记长尾物品 (Group 0)
        item_groups = self.author_groups[self.item2author]
        self.is_tail_item = (item_groups == 0).float() 
        
        # 训练/测试集
        self.train_pairs = np.array(data['train_pairs'])
        self.test_pairs = data['test_pairs']
        
        # 快速查询字典
        self.train_dict = {}
        for u, i in self.train_pairs:
            if u not in self.train_dict: self.train_dict[u] = []
            self.train_dict[u].append(i)
        self.test_dict = {}
        for u, i in self.test_pairs:
            if u not in self.test_dict: self.test_dict[u] = []
            self.test_dict[u].append(i)
            
        # 构建图
        self.graph = self._build_sparse_graph()
        
        # ============ [核心修复] 构建社交矩阵 for Part C ============
        # 1. 社交 Set (用于评估 FanR)
        self.social_set = set()
        if 'social_edges' in data and len(data['social_edges']) > 0:
            for u, a in data['social_edges']:
                self.social_set.add((u, a))
        else:
            print("⚠️ Warning: No social_edges found.")

        # 2. 社交 Matrix (用于模型预测 predict 加分)
        # 这是一个 [Num_Users, Num_Authors] 的 Dense 矩阵
        # 因为 Small Matrix 很小，Dense 矩阵存得下且查询最快
        self.social_matrix = torch.zeros((self.num_users, self.num_authors)).to(conf.device)
        
        if len(self.social_set) > 0:
            # 将 list 转 tensor 索引
            edges = torch.tensor(data['social_edges'], dtype=torch.long).to(conf.device)
            # edges shape: [N, 2], col 0 is user, col 1 is author
            # 赋值为 1.0
            self.social_matrix[edges[:, 0], edges[:, 1]] = 1.0
            print(f"✅ Social Matrix 构建完成! Shape: {self.social_matrix.shape}, Edges: {len(edges)}")

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
        d_inv = np.power(rowsum + 1e-9, -0.5).flatten()
        d_mat_inv = sp.diags(d_inv)
        norm_adj = d_mat_inv.dot(adj).dot(d_mat_inv)
        
        norm_adj = norm_adj.tocoo()
        i = torch.LongTensor(np.vstack((norm_adj.row, norm_adj.col)))
        v = torch.FloatTensor(norm_adj.data)
        return torch.sparse_coo_tensor(i, v, torch.Size(norm_adj.shape)).to(config.device)

    def __len__(self): return len(self.train_pairs)
    def __getitem__(self, idx): return self.train_pairs[idx][0], self.train_pairs[idx][1]

# ================= 2. 模型定义 (Part A, B, C) =================

# --- Part B: Manager ---
class ManagerNetwork(nn.Module):
    def __init__(self, num_active_levels, emb_dim=32):
        super(ManagerNetwork, self).__init__()
        self.active_emb = nn.Embedding(num_active_levels, emb_dim)
        self.net = nn.Sequential(
            nn.Linear(emb_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    def forward(self, active_level_ids):
        emb = self.active_emb(active_level_ids)
        return self.net(emb)

# --- Part A: Worker ---
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

# --- Joint Model (Putting it all together) ---
class JointModel(nn.Module):
    def __init__(self, dataset, conf):
        super(JointModel, self).__init__()
        self.worker = CreatorXSimGCL(dataset, conf)
        self.manager = ManagerNetwork(dataset.num_active_levels)
        self.dataset = dataset
        self.conf = conf
        
    def predict(self, u_batch, i_batch, use_manager=True):
        # 1. Base Score (Part A)
        users_emb, items_emb = self.worker() 
        u_e = users_emb[u_batch]
        i_e = items_emb[i_batch]
        base_score = (u_e * i_e).sum(dim=1)
        
        if not use_manager:
            return base_score
        
        # 2. Manager Boost (Part B)
        u_active = self.dataset.user_active[u_batch]
        boost_weight = self.manager(u_active).squeeze() 
        is_tail = self.dataset.is_tail_item[i_batch]
        
        # 3. Social Boost (Part C: Social Residual) --- [核心新增]
        # 查找这批 item 对应的 author
        batch_authors = self.dataset.item2author[i_batch] # [batch_size]
        # 查表: user 是否关注了这些 author?
        # u_batch shape: [2048], batch_authors shape: [2048]
        # indexing: social_matrix[u, a] -> returns 1.0 or 0.0
        social_bonus = self.dataset.social_matrix[u_batch, batch_authors]
        
        # 4. Final Combination
        final_score = base_score + \
                      (boost_weight * is_tail * self.conf.manager_scale) + \
                      (social_bonus * self.conf.social_scale)
                      
        return final_score

# ================= 3. Loss & Evaluate =================
def cal_bpr_loss(scores_pos, scores_neg):
    return -torch.log(torch.sigmoid(scores_pos - scores_neg) + 1e-8).mean()

def cal_infonce_loss(view1, view2, temp):
    view1 = F.normalize(view1, dim=1)
    view2 = F.normalize(view2, dim=1)
    pos_score = torch.exp((view1 * view2).sum(dim=1) / temp)
    ttl_score = torch.matmul(view1, view2.transpose(0, 1)) / temp
    ttl_score = torch.exp(ttl_score).sum(dim=1)
    return -torch.log(pos_score / ttl_score + 1e-8).mean()

def evaluate_h3(model, dataset, top_k=20):
    model.eval()
    NDCG_list, FanR_list = [], []
    recommended_item_counts = {}
    
    test_users = list(dataset.test_dict.keys())
    item2author_np = dataset.item2author.cpu().numpy()
    
    with torch.no_grad():
        all_users, all_items = model.worker()
        
        for start in range(0, len(test_users), 100):
            end = min(start + 100, len(test_users))
            batch_u_ids = torch.LongTensor(test_users[start:end]).to(config.device)
            
            # --- Inference (User x All Items) ---
            batch_u_emb = all_users[batch_u_ids]
            # 1. Base
            scores = torch.matmul(batch_u_emb, all_items.transpose(0, 1))
            
            # 2. Manager
            u_active = dataset.user_active[batch_u_ids]
            weights = model.manager(u_active) # [100, 1]
            is_tail = dataset.is_tail_item.unsqueeze(0) # [1, n_items]
            
            # 3. Social (Part C) - Vectorized for User x All Items
            # matrix slicing: [100 users] x [all authors]
            # social_matrix: [N_users, N_authors]
            # batch_social: [100, N_authors]
            batch_social_authors = dataset.social_matrix[batch_u_ids] 
            # item2author mapping expansion
            # We need [100, n_items]. 
            # batch_social_authors [100, n_authors] -> map cols via item2author -> [100, n_items]
            # user_social_score = batch_social_authors[:, dataset.item2author] -- this is huge
            # efficient way: gather
            item_authors = dataset.item2author # [n_items]
            # expand item_authors to match batch size is inefficient, 
            # instead we just use the pre-computed matrix logic or loop
            # For speed in evaluation, simple gather:
            social_bonus = batch_social_authors[:, item_authors]
            
            # Final Score Sum
            scores = scores + \
                     (weights * is_tail * config.manager_scale) + \
                     (social_bonus * config.social_scale)
            
            # Mask Train
            for i, u_id in enumerate(batch_u_ids.cpu().numpy()):
                train_pos = dataset.train_dict.get(u_id, [])
                scores[i, train_pos] = -1e9
            
            _, indices = torch.topk(scores, top_k, dim=1)
            indices = indices.cpu().numpy()
            
            # --- Metrics ---
            for i, u_id in enumerate(batch_u_ids.cpu().numpy()):
                # NDCG
                ground_truth = set(dataset.test_dict[u_id])
                dcg, idcg = 0, 0
                for j, item_idx in enumerate(indices[i]):
                    if item_idx in ground_truth:
                        dcg += 1.0 / np.log2(j + 2)
                    if j < len(ground_truth):
                        idcg += 1.0 / np.log2(j + 2)
                NDCG_list.append(dcg / idcg if idcg > 0 else 0)
                
                # FanR (基于 Social Set 验证)
                fan_cnt = 0
                for item_idx in indices[i]:
                    author_idx = item2author_np[item_idx]
                    if (u_id, author_idx) in dataset.social_set:
                        fan_cnt += 1
                FanR_list.append(fan_cnt / top_k)
                
                # Gini
                for item_idx in indices[i]:
                    recommended_item_counts[item_idx] = recommended_item_counts.get(item_idx, 0) + 1

    avg_ndcg = np.mean(NDCG_list)
    avg_fanr = np.mean(FanR_list)
    
    # Gini
    if len(recommended_item_counts) == 0:
        gini = 0.0
    else:
        all_counts = np.zeros(dataset.num_items)
        for i, c in recommended_item_counts.items():
            all_counts[i] = c
        all_counts.sort()
        n = len(all_counts)
        cum_counts = np.cumsum(all_counts)
        sum_counts = cum_counts[-1]
        if sum_counts > 0:
            index = np.arange(1, n + 1)
            gini = ((2 * index - n - 1) * all_counts).sum() / (n * sum_counts)
        else:
            gini = 0.0

    # H3 Score
    epsilon = 1e-6
    h3_score = 3.0 / (1.0/(avg_ndcg + epsilon) + 1.0/((1.0 - gini) + epsilon) + 1.0/(avg_fanr + epsilon))
    
    return avg_ndcg, gini, avg_fanr, h3_score

# ================= 4. 主循环 =================
if __name__ == '__main__':
    dataset = RecDataset(config)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    
    model = JointModel(dataset, config).to(config.device)
    optimizer = optim.Adam([
        {'params': model.worker.parameters(), 'lr': config.lr_worker},
        {'params': model.manager.parameters(), 'lr': config.lr_manager}
    ], weight_decay=config.weight_decay)
    
    print("🔥 Start Training (Part A/B/C Activated)...")
    
    for epoch in range(config.epochs):
        model.train()
        total_loss = 0
        
        for batch_u, batch_pos_i in dataloader:
            batch_u = batch_u.to(config.device)
            batch_pos_i = batch_pos_i.to(config.device)
            batch_neg_i = torch.randint(0, dataset.num_items, batch_pos_i.shape).to(config.device)
            
            # 1. Accuracy Loss
            score_pos = model.predict(batch_u, batch_pos_i, use_manager=True)
            score_neg = model.predict(batch_u, batch_neg_i, use_manager=True)
            acc_loss = cal_bpr_loss(score_pos, score_neg)
            
            # 2. Contrastive Loss
            u_v1, i_v1 = model.worker(perturbed=True)
            u_v2, i_v2 = model.worker(perturbed=True)
            cl_loss = config.cl_rate * (
                cal_infonce_loss(u_v1[batch_u], u_v2[batch_u], config.temp) +
                cal_infonce_loss(i_v1[batch_pos_i], i_v2[batch_pos_i], config.temp)
            )
            
            # 3. Ecosystem Loss
            is_tail_pos = dataset.is_tail_item[batch_pos_i]
            eco_loss = 0
            if is_tail_pos.sum() > 0:
                eco_loss = F.relu(1.0 - score_pos[is_tail_pos.bool()]).mean() * config.eco_lambda
            
            loss = acc_loss + cl_loss + eco_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        # 评估
        ndcg, gini, fanr, h3 = evaluate_h3(model, dataset)
        print(f"Epoch {epoch+1:2d} | Loss: {total_loss/len(dataloader):.4f} | "
              f"NDCG: {ndcg:.4f} | Gini: {gini:.4f} | FanR: {fanr:.4f} | "
              f"H3-Score: {h3:.4f}")

    print(f"\n✅ Finished! Final FanR: {fanr:.4f}, H3: {h3:.4f}")