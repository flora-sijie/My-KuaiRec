import pandas as pd
import numpy as np
import os
import pickle
from sklearn.preprocessing import LabelEncoder
from ast import literal_eval
import gc

# ================= 0. 配置区 =================
DATA_DIR = './data'
OUTPUT_DIR = './processed_data'
SOURCE_FILE = 'big_matrix.csv' # 使用全量数据
OUTPUT_FILE = 'dataset_hybrid.pkl' # 输出文件名

# --- 核心挖掘参数 ---
# 1. 隐式关注的数量上限：每个用户最多挖掘 15 个
IMPLICIT_TOP_K = 15  

# 2. 最低互动阈值：只有跟作者互动(观看) >= 2 次才算"隐式关注"
# 这能有效过滤掉偶然点击的噪点，保证提取出的 Top-20 都是用户真正感兴趣的
MIN_INTERACT_COUNT = 2 

# --- 其他参数 ---
WATCH_RATIO_THRESHOLD = 0.5 
HEAD_RATIO = 0.2
MID_RATIO = 0.3
RANDOM_SEED = 2023
np.random.seed(RANDOM_SEED)

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def process_data_hybrid():
    print(f"🚀 开始全量数据处理 (Hybrid: 显式 + 隐式挖掘)...")
    print(f"   策略: Top-{IMPLICIT_TOP_K} 且 互动次数>={MIN_INTERACT_COUNT}")

    # ================= 1. 读取交互 (Big Matrix) =================
    print(f"--- [1/6] 读取交互矩阵 ({SOURCE_FILE}) ---")
    # 优化内存：只读需要的列
    use_cols = ['user_id', 'video_id', 'watch_ratio']
    df_inter = pd.read_csv(os.path.join(DATA_DIR, SOURCE_FILE), usecols=use_cols)
    
    # 过滤无效交互
    original_len = len(df_inter)
    df_inter = df_inter[df_inter['watch_ratio'] >= WATCH_RATIO_THRESHOLD].copy()
    print(f"    原始数据: {original_len} -> 过滤后有效交互: {len(df_inter)}")

    # ================= 2. 关联作者信息 =================
    print("--- [2/6] 关联作者信息 ---")
    feat_path = os.path.join(DATA_DIR, 'item_daily_features.csv')
    # 只读两列，节省内存
    df_item_feat = pd.read_csv(feat_path, usecols=['video_id', 'author_id'])
    
    # 去重得到 video -> author 映射
    item2author_raw = df_item_feat.drop_duplicates('video_id')[['video_id', 'author_id']]
    
    # 过滤：交互表里的视频必须有作者信息
    valid_videos = set(item2author_raw.video_id)
    df_inter = df_inter[df_inter['video_id'].isin(valid_videos)].copy()
    print(f"    关联作者后剩余交互: {len(df_inter)}")

    # 释放内存
    del df_item_feat
    gc.collect()

    # ================= 3. ID 编码 (User/Item/Author) =================
    print("--- [3/6] ID 编码 ---")
    
    # User
    user_encoder = LabelEncoder()
    df_inter['user_idx'] = user_encoder.fit_transform(df_inter['user_id'])
    
    # Item
    item_encoder = LabelEncoder()
    df_inter['item_idx'] = item_encoder.fit_transform(df_inter['video_id'])
    
    # Author
    # 逻辑: video_id(str) -> author_id(str) -> author_idx(int)
    raw_vid2aid = dict(zip(item2author_raw.video_id, item2author_raw.author_id))
    
    # 只为在交互中出现的 Item 对应的 Author 进行编码
    relevant_authors_raw = [raw_vid2aid[vid] for vid in item_encoder.classes_]
    
    author_encoder = LabelEncoder()
    author_mapped_ids = author_encoder.fit_transform(relevant_authors_raw)
    
    # 核心映射数组: item_idx -> author_idx
    item2author_array = author_mapped_ids 
    
    num_users = len(user_encoder.classes_)
    num_items = len(item_encoder.classes_)
    num_authors = len(author_encoder.classes_)
    print(f"    Users: {num_users}, Items: {num_items}, Authors: {num_authors}")

    # ================= 4. 混合社交关系构建 (Hybrid) =================
    print(f"--- [4/6] 构建混合社交网络 ---")

    # ----- A. 挖掘隐式交互 (Implicit Mining) -----
    print(f"    A. 挖掘隐式交互 (Top-{IMPLICIT_TOP_K}, MinCount>={MIN_INTERACT_COUNT})...")
    
    # 给交互表打上 author_idx
    df_inter['author_idx'] = item2author_array[df_inter['item_idx'].values]
    
    # 聚合：统计 (User, Author) 的互动次数
    # 这一步在全量数据上可能稍慢，请耐心等待
    print("       正在聚合 User-Author 交互频次...")
    user_author_counts = df_inter.groupby(['user_idx', 'author_idx']).size().reset_index(name='count')
    
    # [关键步骤] 过滤掉偶然交互 (只保留互动 >= 2次的)
    valid_interactions = user_author_counts[user_author_counts['count'] >= MIN_INTERACT_COUNT].copy()
    print(f"       过滤低频交互后，剩余候选对: {len(valid_interactions)}")
    
    # 排序：按互动次数降序
    valid_interactions = valid_interactions.sort_values(['user_idx', 'count'], ascending=[True, False])
    
    # 截断：每个用户取 Top-K
    # 注意：如果用户有效交互不足 K 个，这里会自动取实际个数，不会报错
    top_k_social = valid_interactions.groupby('user_idx').head(IMPLICIT_TOP_K)
    
    implicit_edges = top_k_social[['user_idx', 'author_idx']].values
    print(f"       >>> 挖掘出隐式边数: {len(implicit_edges)}")

    # ----- B. 提取显式关注 (Explicit Extraction) -----
    print(f"    B. 提取显式关注 (social_network.csv)...")
    df_social = pd.read_csv(os.path.join(DATA_DIR, 'social_network.csv'))
    explicit_edges = []
    
    valid_user_set = set(user_encoder.classes_)
    valid_author_set = set(author_encoder.classes_)
    
    # 只处理有效用户
    df_social = df_social[df_social['user_id'].isin(valid_user_set)]
    
    print(f"       正在扫描 {len(df_social)} 个用户的关注列表...")
    for _, row in df_social.iterrows():
        try:
            u_raw = row['user_id']
            friend_list = literal_eval(row['friend_list']) # 解析列表字符串
            
            u_idx = user_encoder.transform([u_raw])[0]
            
            for f_raw in friend_list:
                # 只有关注了发过视频的作者，才算有效边
                if f_raw in valid_author_set:
                    a_idx = author_encoder.transform([f_raw])[0]
                    explicit_edges.append([u_idx, a_idx])
        except:
            continue
            
    explicit_edges = np.array(explicit_edges)
    print(f"       >>> 提取出显式边数: {len(explicit_edges)}")

    # ----- C. 合并与去重 (Merge) -----
    print(f"    C. 合并去重...")
    if len(explicit_edges) > 0 and len(implicit_edges) > 0:
        social_edges = np.vstack([implicit_edges, explicit_edges])
    elif len(implicit_edges) > 0:
        social_edges = implicit_edges
    elif len(explicit_edges) > 0:
        social_edges = explicit_edges
    else:
        social_edges = np.empty((0, 2), dtype=int)
        
    # 去重：如果同一对关系在显式和隐式里都出现了，只留一条
    social_edges = np.unique(social_edges, axis=0)
    
    print(f"✅ 混合社交网络构建完成！")
    print(f"   最终总边数: {len(social_edges)}")
    print(f"   平均每人关注: {len(social_edges)/num_users:.2f} 个作者")

    # ================= 5. 特征处理 =================
    print("--- [5/6] 特征处理 ---")
    
    # 用户活跃度
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
    
    # 类型匹配检测 (防止 str vs int 导致匹配失败)
    sample_key = list(u_feat_map.keys())[0]
    need_str_convert = isinstance(sample_key, str) and not isinstance(user_encoder.classes_[0], str)
    
    for i, u_raw in enumerate(user_encoder.classes_):
        k = str(u_raw) if need_str_convert else u_raw
        if k in u_feat_map:
            user_active_feature[i] = active_encoder.transform([u_feat_map[k]])[0]

    # 作者热度分层
    author_heat = np.zeros(num_authors, dtype=int)
    # 使用所有交互统计热度
    total_author_counts = df_inter.groupby('author_idx').size()
    for aid, cnt in total_author_counts.items():
        author_heat[aid] = cnt
        
    sorted_idx = np.argsort(author_heat)[::-1]
    n_head = int(num_authors * HEAD_RATIO)
    n_mid = int(num_authors * MID_RATIO)
    
    author_groups = np.zeros(num_authors, dtype=int)
    author_groups[sorted_idx[n_head:n_head+n_mid]] = 1 # Mid
    author_groups[sorted_idx[:n_head]] = 2 # Head

    # ================= 6. 切分与保存 =================
    print("--- [6/6] 切分数据集与保存 ---")
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
        'social_edges': social_edges, # 混合后的边
        'train_pairs': df_inter.iloc[indices[:split]][['user_idx', 'item_idx']].values,
        'test_pairs': df_inter.iloc[indices[split:]][['user_idx', 'item_idx']].values
    }
    
    with open(os.path.join(OUTPUT_DIR, OUTPUT_FILE), 'wb') as f:
        pickle.dump(dataset, f)
        
    print(f"🎉 处理完成！文件已保存至: {os.path.join(OUTPUT_DIR, OUTPUT_FILE)}")

if __name__ == '__main__':
    process_data_hybrid()