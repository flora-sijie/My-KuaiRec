import pandas as pd
import numpy as np
import os
import pickle
from sklearn.preprocessing import LabelEncoder
from ast import literal_eval

# ================= 配置区 =================
DATA_DIR = './data' # 确保路径正确
OUTPUT_DIR = './processed_data'
SOURCE_FILE = 'small_matrix.csv' # 之后改 big_matrix.csv
WATCH_RATIO_THRESHOLD = 0.5 

# 分层比例
HEAD_RATIO = 0.2
MID_RATIO = 0.3

RANDOM_SEED = 2023
np.random.seed(RANDOM_SEED)

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def process_data_final():
    print(f"🚀 开始最终版数据处理 (Target: 捕获那50条有效边)...")

    # 1. 读取交互
    print(f"--- [1/6] 读取交互矩阵 ({SOURCE_FILE}) ---")
    df_inter = pd.read_csv(os.path.join(DATA_DIR, SOURCE_FILE))
    df_inter = df_inter[df_inter['watch_ratio'] >= WATCH_RATIO_THRESHOLD].copy()
    
    # 2. 读取作者信息 (关联 video_id -> author_id)
    print("--- [2/6] 构建 Item-Author 映射 ---")
    feat_path = os.path.join(DATA_DIR, 'item_daily_features.csv')
    # 只读两列
    df_item_feat = pd.read_csv(feat_path, usecols=['video_id', 'author_id'])
    # 去重
    item2author_raw = df_item_feat.drop_duplicates('video_id')[['video_id', 'author_id']]
    
    # 过滤：交互矩阵里的视频必须有作者信息
    valid_video_ids = set(item2author_raw.video_id)
    df_inter = df_inter[df_inter['video_id'].isin(valid_video_ids)].copy()
    
    # 3. ID Encoding
    print("--- [3/6] ID 编码 ---")
    
    # User
    user_encoder = LabelEncoder()
    df_inter['user_idx'] = user_encoder.fit_transform(df_inter['user_id'])
    user_list = user_encoder.classes_
    num_users = len(user_list)
    
    # Item
    item_encoder = LabelEncoder()
    df_inter['item_idx'] = item_encoder.fit_transform(df_inter['video_id'])
    item_list = item_encoder.classes_
    num_items = len(item_list)
    
    # Author
    # 逻辑：先建立 map: video_id -> author_id
    raw_vid2aid = dict(zip(item2author_raw.video_id, item2author_raw.author_id))
    # 找到所有 items 对应的 raw author id
    relevant_authors_raw = [raw_vid2aid[vid] for vid in item_list]
    
    author_encoder = LabelEncoder()
    author_mapped_ids = author_encoder.fit_transform(relevant_authors_raw) # item对应的author_idx数组
    num_authors = len(author_encoder.classes_)
    
    # 关键数组: item_idx -> author_idx
    item2author_array = author_mapped_ids
    
    # 关键集合: 用于社交过滤的“有效作者原始ID”集合
    # 只有在这个集合里的人，才算“生产者”
    valid_author_raw_set = set(author_encoder.classes_)
    
    print(f"   统计: Users={num_users}, Items={num_items}, Authors={num_authors}")

    # 4. User Features (活跃度)
    print("--- [4/6] 用户活跃度特征 ---")
    df_user_feat = pd.read_csv(os.path.join(DATA_DIR, 'user_features.csv'))
    # 填充空值
    df_user_feat['user_active_degree'] = df_user_feat['user_active_degree'].fillna('unknown')
    # 建立映射
    u_feat_map = df_user_feat.set_index('user_id')['user_active_degree'].to_dict()
    
    active_encoder = LabelEncoder()
    # 收集所有可能的标签并 fit
    all_labels = list(df_user_feat['user_active_degree'].unique())
    if 'unknown' not in all_labels: all_labels.append('unknown')
    active_encoder.fit(all_labels)
    num_active_levels = len(active_encoder.classes_)
    unknown_code = active_encoder.transform(['unknown'])[0]
    
    user_active_feature = np.full(num_users, unknown_code, dtype=int)
    for i, u_raw in enumerate(user_list):
        if u_raw in u_feat_map:
            user_active_feature[i] = active_encoder.transform([u_feat_map[u_raw]])[0]

    # 5. 社交关系 (Social Edges) - 核心修正部分
    print("--- [5/6] 提取社交关系 (User -> Author) ---")
    df_social = pd.read_csv(os.path.join(DATA_DIR, 'social_network.csv'))
    
    social_edges = []
    # 这里的 user_list 是 LabelEncoder 里的 classes_，即 raw user ids
    valid_user_raw_set = set(user_list)
    
    for _, row in df_social.iterrows():
        u_raw = row['user_id']
        # 1. 关注者必须在我们的用户集里
        if u_raw not in valid_user_raw_set:
            continue
            
        try:
            friend_list = literal_eval(row['friend_list'])
        except:
            continue
            
        # 获取编码后的 user_idx
        u_idx = user_encoder.transform([u_raw])[0]
        
        for f_raw in friend_list:
            # 2. 被关注者必须在我们的作者集里 (这样才算关注了生产者)
            if f_raw in valid_author_raw_set:
                # 获取编码后的 author_idx
                a_idx = author_encoder.transform([f_raw])[0]
                social_edges.append([u_idx, a_idx])
                
    social_edges = np.array(social_edges)
    print(f"✅ 成功提取社交边: {len(social_edges)} 条 (预期应接近 50)")

    # 6. 创作者分层
    print("--- [6/6] 创作者分层 ---")
    author_heat = np.zeros(num_authors, dtype=int)
    # 统计交互
    # 遍历所有交互，找到对应的 item_idx -> 找到 author_idx -> 加热度
    # 更快的方法：
    item_counts = df_inter['item_idx'].value_counts()
    for i_idx, cnt in item_counts.items():
        a_idx = item2author_array[i_idx]
        author_heat[a_idx] += cnt
        
    sorted_idx = np.argsort(author_heat)[::-1]
    n_head = int(num_authors * HEAD_RATIO)
    n_mid = int(num_authors * MID_RATIO)
    
    author_groups = np.zeros(num_authors, dtype=int)
    author_groups[sorted_idx[n_head:n_head+n_mid]] = 1 # Mid
    author_groups[sorted_idx[:n_head]] = 2 # Head
    
    # 7. 保存
    indices = np.arange(len(df_inter))
    np.random.shuffle(indices)
    split = int(len(indices) * 0.8)
    
    dataset = {
        'num_users': int(num_users),
        'num_items': int(num_items),
        'num_authors': int(num_authors),
        'num_active_levels': int(num_active_levels),
        'item2author': item2author_array,
        'user_active_feature': user_active_feature,
        'author_groups': author_groups,
        'social_edges': social_edges, # 这里存的一定是 [user_idx, author_idx]
        'train_pairs': df_inter.iloc[indices[:split]][['user_idx', 'item_idx']].values,
        'test_pairs': df_inter.iloc[indices[split:]][['user_idx', 'item_idx']].values
    }
    
    with open(os.path.join(OUTPUT_DIR, 'dataset.pkl'), 'wb') as f:
        pickle.dump(dataset, f)
    print(f"🎉 处理完成！dataset.pkl 已生成。")

if __name__ == '__main__':
    process_data_final()