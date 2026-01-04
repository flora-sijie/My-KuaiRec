import pandas as pd
import numpy as np
import os
import pickle
from sklearn.preprocessing import LabelEncoder
from ast import literal_eval
import gc # 引入垃圾回收，防止内存爆掉

# ================= 配置区 (Big Matrix) =================
DATA_DIR = './data'
OUTPUT_DIR = './processed_data'
SOURCE_FILE = 'big_matrix.csv' # 👈 这里改成了全量数据
OUTPUT_FILE = 'dataset_big.pkl' # 👈 这里改名了，不会覆盖 dataset.pkl

# 过滤阈值
WATCH_RATIO_THRESHOLD = 0.5 
HEAD_RATIO = 0.2
MID_RATIO = 0.3
RANDOM_SEED = 2023
np.random.seed(RANDOM_SEED)

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def process_data_big():
    print(f"🚀 开始全量数据处理 (Big Matrix)...")
    print(f"⚠️ 注意：全量数据量大，请关注 Colab 内存使用情况")

    # 1. 读取交互 (分块读取或只读必要列以节省内存)
    print(f"--- [1/6] 读取交互矩阵 ({SOURCE_FILE}) ---")
    file_path = os.path.join(DATA_DIR, SOURCE_FILE)
    
    # 优化：只读取需要的列
    use_cols = ['user_id', 'video_id', 'watch_ratio']
    df_inter = pd.read_csv(file_path, usecols=use_cols)
    
    # 过滤
    original_len = len(df_inter)
    df_inter = df_inter[df_inter['watch_ratio'] >= WATCH_RATIO_THRESHOLD].copy()
    print(f"    - 原始: {original_len} -> 过滤后: {len(df_inter)}")
    
    # 2. Item -> Author 映射
    print("--- [2/6] 构建 Item-Author 映射 ---")
    feat_path = os.path.join(DATA_DIR, 'item_daily_features.csv')
    # 只读两列
    df_item_feat = pd.read_csv(feat_path, usecols=['video_id', 'author_id'])
    item2author_raw = df_item_feat.drop_duplicates('video_id')[['video_id', 'author_id']]
    
    # 过滤：交互中的视频必须有作者
    valid_video_ids = set(item2author_raw.video_id)
    df_inter = df_inter[df_inter['video_id'].isin(valid_video_ids)].copy()
    
    # 释放不再需要的内存
    del df_item_feat
    gc.collect()

    # 3. ID Encoding
    print("--- [3/6] ID 编码 ---")
    
    # User
    user_encoder = LabelEncoder()
    df_inter['user_idx'] = user_encoder.fit_transform(df_inter['user_id'])
    num_users = len(user_encoder.classes_)
    
    # Item
    item_encoder = LabelEncoder()
    df_inter['item_idx'] = item_encoder.fit_transform(df_inter['video_id'])
    num_items = len(item_encoder.classes_)
    
    # Author
    raw_vid2aid = dict(zip(item2author_raw.video_id, item2author_raw.author_id))
    # 只转换在 df_inter 里出现过的 item 对应的 author
    relevant_authors_raw = [raw_vid2aid[vid] for vid in item_encoder.classes_]
    
    author_encoder = LabelEncoder()
    author_mapped_ids = author_encoder.fit_transform(relevant_authors_raw)
    num_authors = len(author_encoder.classes_)
    
    item2author_array = author_mapped_ids
    valid_author_raw_set = set(author_encoder.classes_) # 用于社交过滤
    
    print(f"   统计: Users={num_users}, Items={num_items}, Authors={num_authors}")

    # 4. User Features
    print("--- [4/6] 用户活跃度特征 ---")
    df_user_feat = pd.read_csv(os.path.join(DATA_DIR, 'user_features.csv'))
    df_user_feat['user_active_degree'] = df_user_feat['user_active_degree'].fillna('unknown')
    u_feat_map = df_user_feat.set_index('user_id')['user_active_degree'].to_dict()
    
    active_encoder = LabelEncoder()
    all_labels = list(df_user_feat['user_active_degree'].unique())
    if 'unknown' not in all_labels: all_labels.append('unknown')
    active_encoder.fit(all_labels)
    num_active_levels = len(active_encoder.classes_)
    unknown_code = active_encoder.transform(['unknown'])[0]
    
    user_active_feature = np.full(num_users, unknown_code, dtype=int)
    # 这里要小心：user_encoder.classes_ 是 int 还是 str？BigMatrix 里通常是 int
    # 确保类型匹配
    sample_feat_key = list(u_feat_map.keys())[0]
    sample_enc_key = user_encoder.classes_[0]
    
    # 类型转换检测
    need_str_convert = isinstance(sample_feat_key, str) and not isinstance(sample_enc_key, str)
    
    for i, u_raw in enumerate(user_encoder.classes_):
        key = str(u_raw) if need_str_convert else u_raw
        if key in u_feat_map:
            user_active_feature[i] = active_encoder.transform([u_feat_map[key]])[0]

    # 5. 社交关系 (真实全量)
    print("--- [5/6] 提取社交关系 (Real Data Only) ---")
    # 这里我们坚决不造假数据，只提取真实的
    df_social = pd.read_csv(os.path.join(DATA_DIR, 'social_network.csv'))
    
    social_edges = []
    valid_user_raw_set = set(user_encoder.classes_)
    
    # 优化循环速度
    df_social_filtered = df_social[df_social['user_id'].isin(valid_user_raw_set)]
    
    print(f"    正在扫描 {len(df_social_filtered)} 个用户的关注列表...")
    
    for _, row in df_social_filtered.iterrows():
        u_raw = row['user_id']
        try:
            friend_list = literal_eval(row['friend_list'])
        except:
            continue
            
        u_idx = user_encoder.transform([u_raw])[0]
        
        for f_raw in friend_list:
            if f_raw in valid_author_raw_set:
                a_idx = author_encoder.transform([f_raw])[0]
                social_edges.append([u_idx, a_idx])
                
    social_edges = np.array(social_edges)
    # 去重
    if len(social_edges) > 0:
        social_edges = np.unique(social_edges, axis=0)
        
    print(f"✅ 成功提取真实社交边: {len(social_edges)} 条")

    # 6. 分层与保存
    print("--- [6/6] 分层与保存 ---")
    author_heat = np.zeros(num_authors, dtype=int)
    item_counts = df_inter['item_idx'].value_counts()
    for i_idx, cnt in item_counts.items():
        author_heat[item2author_array[i_idx]] += cnt
        
    sorted_idx = np.argsort(author_heat)[::-1]
    n_head = int(num_authors * HEAD_RATIO)
    n_mid = int(num_authors * MID_RATIO)
    
    author_groups = np.zeros(num_authors, dtype=int)
    author_groups[sorted_idx[n_head:n_head+n_mid]] = 1
    author_groups[sorted_idx[:n_head]] = 2
    
    # 切分
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
        'social_edges': social_edges,
        'train_pairs': df_inter.iloc[indices[:split]][['user_idx', 'item_idx']].values,
        'test_pairs': df_inter.iloc[indices[split:]][['user_idx', 'item_idx']].values
    }
    
    with open(os.path.join(OUTPUT_DIR, OUTPUT_FILE), 'wb') as f:
        pickle.dump(dataset, f)
        
    print(f"🎉 全量数据处理完成！保存为: {OUTPUT_FILE}")

if __name__ == '__main__':
    process_data_big()