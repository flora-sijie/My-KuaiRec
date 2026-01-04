import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import scipy.sparse as sp
import os

# ================= 0. 配置 (Baseline: XSimGCL) =================
class Config:
    # 必须使用相同的数据集，保证评估指标公平
    dataset_path = './processed_data/dataset_hybrid.pkl' 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 训练参数
    epochs = 30
    batch_size = 10240 
    
    # 学习率
    lr_worker = 0.001  # Baseline 通常用标准的学习率
    weight_decay = 1e-4
    
    # XSimGCL 参数
    emb_dim = 64
    n_layers = 3
    cl_rate = 0.2
    eps = 0.1
    temp = 0.2
    
    # [核心修改] 将所有"你的创新点"权重归零
    eco_lambda = 0.0      # 无生态 Loss
    manager_scale = 0.0   # 无 Manager 干预
    social_scale = 0.0    # 无 Social 干预
    
    top_k = 20

config = Config()
print(f"🚀 Running Baseline (XSimGCL) on {config.device}")
print(f"⚠️  Note: Manager & Social modules are DISABLED.")

# ================= 1. 数据集 (不变) =================
class RecDataset(Dataset):
    def __init__(self, conf):
        print("正在加载数据集...")
        with open(conf.dataset_path, 'rb') as f:
            data = pickle.load(f)
            
        self.num_users = data['num_users']
        self.num_items = data['num_items']
        self.num_authors = data['num_authors']
        self.num_active_levels = data['num_active_levels']
        
        self.item2author = torch.LongTensor(data['item2author']).to(conf.device)
        self.author_groups = torch.LongTensor(data['author_groups']).to(conf.device)
        self.user_active = torch.LongTensor(data['user_active_feature']).to(conf.device)
        
        item_groups = self.author_groups[self.item2author]
        self.is_tail_item = (item_groups == 0).float() 
        
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
        
        # 依然加载 Social Set 用于"评估"，但训练不用它
        self.social_set = set()
        if 'social_edges' in data and len(data['social_edges']) > 0:
            edges = data['social_edges']
            for u, a in edges:
                self.social_set.add((u, a))

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

# ================= 2. 模型 (Baseline Mode) =================
# 虽然保留了类结构，但 Manager 不参与计算
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
        mixed_item_emb = self.item_emb.weight + self.author_emb(self.item2author_map)
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

class JointModel(nn.Module):
    def __init__(self, dataset, conf):
        super(JointModel, self).__init__()
        self.worker = CreatorXSimGCL(dataset, conf)
        self.dataset = dataset
        self.conf = conf
        
    def predict(self, u_batch, i_batch):
        # [Baseline 核心逻辑]
        # 只计算 Base Score，没有任何加权
        users_emb, items_emb = self.worker() 
        base_score = (users_emb[u_batch] * items_emb[i_batch]).sum(dim=1)
        return base_score

# ================= 3. 评估与循环 =================
def evaluate_metrics(model, dataset, top_k=20):
    model.eval()
    NDCG_list, Recall_list, FanR_list = [], [], []
    recommended_item_counts = {}
    test_users = list(dataset.test_dict.keys())
    item2author_np = dataset.item2author.cpu().numpy()
    
    with torch.no_grad():
        all_users, all_items = model.worker()
        
        for start in range(0, len(test_users), 200):
            end = min(start + 200, len(test_users))
            batch_u_ids = torch.LongTensor(test_users[start:end]).to(config.device)
            
            batch_u_emb = all_users[batch_u_ids]
            scores = torch.matmul(batch_u_emb, all_items.transpose(0, 1))
            
            # Baseline: 没有 Manager/Social 的加分步骤
            
            for i, u_id in enumerate(batch_u_ids.cpu().numpy()):
                train_pos = dataset.train_dict.get(u_id, [])
                scores[i, train_pos] = -1e9
            
            _, indices = torch.topk(scores, top_k, dim=1)
            indices = indices.cpu().numpy()
            
            for i, u_id in enumerate(batch_u_ids.cpu().numpy()):
                ground_truth = set(dataset.test_dict[u_id])
                
                # NDCG
                dcg, idcg = 0, 0
                for j, item_idx in enumerate(indices[i]):
                    if item_idx in ground_truth: dcg += 1.0 / np.log2(j + 2)
                    if j < len(ground_truth): idcg += 1.0 / np.log2(j + 2)
                NDCG_list.append(dcg / idcg if idcg > 0 else 0)
                
                # Recall
                hit_num = 0
                for item_idx in indices[i]:
                    if item_idx in ground_truth: hit_num += 1
                Recall_list.append(hit_num / len(ground_truth) if len(ground_truth) > 0 else 0)
                
                # FanR (即便 Baseline 为 0 也要算，展示 Gap)
                fan_cnt = 0
                for item_idx in indices[i]:
                    if (u_id, item2author_np[item_idx]) in dataset.social_set: fan_cnt += 1
                FanR_list.append(fan_cnt / top_k)
                
                # Gini
                for item_idx in indices[i]:
                    recommended_item_counts[item_idx] = recommended_item_counts.get(item_idx, 0) + 1

    avg_ndcg = np.mean(NDCG_list)
    avg_recall = np.mean(Recall_list)
    avg_fanr = np.mean(FanR_list)
    
    if len(recommended_item_counts) == 0: gini = 0.0
    else:
        all_counts = np.zeros(dataset.num_items)
        for i, c in recommended_item_counts.items(): all_counts[i] = c
        all_counts.sort()
        n = len(all_counts)
        cum_counts = np.cumsum(all_counts)
        gini = ((2 * np.arange(1, n + 1) - n - 1) * all_counts).sum() / (n * cum_counts[-1]) if cum_counts[-1] > 0 else 0

    epsilon = 1e-6
    h3_score = 3.0 / (1.0/(avg_ndcg + epsilon) + 1.0/((1.0 - gini) + epsilon) + 1.0/(avg_fanr + epsilon))
    return avg_ndcg, avg_recall, gini, avg_fanr, h3_score

def cal_bpr_loss(scores_pos, scores_neg): return -torch.log(torch.sigmoid(scores_pos - scores_neg) + 1e-8).mean()
def cal_infonce_loss(view1, view2, temp):
    view1 = F.normalize(view1, dim=1); view2 = F.normalize(view2, dim=1)
    pos_score = torch.exp((view1 * view2).sum(dim=1) / temp)
    ttl_score = torch.matmul(view1, view2.transpose(0, 1)) / temp
    ttl_score = torch.exp(ttl_score).sum(dim=1)
    return -torch.log(pos_score / ttl_score + 1e-8).mean()

if __name__ == '__main__':
    dataset = RecDataset(config)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    
    # 注意：这里不需要初始化 Manager Network，因为 Baseline 不用
    # 但为了代码兼容 JointModel，我们保留结构，但不优化它
    model = JointModel(dataset, config).to(config.device)
    
    # [核心修改] 优化器只优化 Worker (GNN)
    optimizer = optim.Adam([
        {'params': model.worker.parameters(), 'lr': config.lr_worker}
    ], weight_decay=config.weight_decay)
    
    print("🔥 Start Training Baseline (XSimGCL)...")
    for epoch in range(config.epochs):
        model.train()
        total_loss = 0
        for batch_u, batch_pos_i in dataloader:
            batch_u = batch_u.to(config.device); batch_pos_i = batch_pos_i.to(config.device)
            batch_neg_i = torch.randint(0, dataset.num_items, batch_pos_i.shape).to(config.device)
            
            # Baseline Predict (Pure Worker)
            score_pos = model.predict(batch_u, batch_pos_i)
            score_neg = model.predict(batch_u, batch_neg_i)
            
            acc_loss = cal_bpr_loss(score_pos, score_neg)
            
            # Contrastive Loss
            u_v1, i_v1 = model.worker(True); u_v2, i_v2 = model.worker(True)
            cl_loss = config.cl_rate * (cal_infonce_loss(u_v1[batch_u], u_v2[batch_u], config.temp) + cal_infonce_loss(i_v1[batch_pos_i], i_v2[batch_pos_i], config.temp))
            
            # [核心修改] 无 Eco Loss
            loss = acc_loss + cl_loss 
            
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item()
            
        ndcg, recall, gini, fanr, h3 = evaluate_metrics(model, dataset)
        print(f"Epoch {epoch+1:2d} | Loss: {total_loss/len(dataloader):.4f} | NDCG: {ndcg:.4f} | Recall: {recall:.4f} | Gini: {gini:.4f} | FanR: {fanr:.4f} | H3: {h3:.4f}")