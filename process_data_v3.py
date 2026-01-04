import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import LabelEncoder

# ================= 配置区 =================
# 确保这里是你的文件路径
RAW_DATA_PATH = '/content/drive/MyDrive/MyCode/data/KuaiRec/data' 
OUTPUT_PATH = './processed_data/dataset.pkl'

def process_data_v3():
    print("🚀 开始数据处理 v3 (修复社交关系 ID 对齐问题)...")
    
    # 1. 加载原始数据
    # Big Matrix (全量) 或者 Small Matrix (开发)
    # 建议直接用 big_matrix，如果内存够的话；为了复现问题先用 small
    df_matrix = pd.read_csv(os.path.join(RAW_DATA_PATH, 'small_matrix.csv')) 
    df_social = pd.read_csv(os.path.join(RAW_DATA_PATH, 'social_network.csv'))
    
    # 2. 提取核心列
    # 假设 matrix 列名: user_id, video_id, author_id, ...
    # 假设 social 列名: user_id, friend_id (注意：friend_id 就是被关注者)
    
    # 3. 构建 ID Encoders
    print("--- 构建 ID 映射 ---")
    
    # User Encoder: 基于交互矩阵里的所有用户
    user_le = LabelEncoder()
    all_users = df_matrix['user_id'].unique()
    user_le.fit(all_users)
    
    # Item Encoder
    item_le = LabelEncoder()
    item_le.fit(df_matrix['video_id'].unique())
    
    # Author Encoder: 基于交互矩阵里的所有作者
    author_le = LabelEncoder()
    all_authors = df_matrix['author_id'].unique()
    author_le.fit(all_authors)
    
    print(f"Users: {len(user_le.classes_)}")
    print(f"Items: {len(item_le.classes_)}")
    print(f"Authors: {len(author_le.classes_)}")
    
    # 4. 转换交互矩阵 ID
    print("--- 转换交互矩阵 ---")
    df_matrix['u_idx'] = user_le.transform(df_matrix['user_id'])
    df_matrix['i_idx'] = item_le.transform(df_matrix['video_id'])
    df_matrix['a_idx'] = author_le.transform(df_matrix['author_id'])
    
    # 5. 构建 Item -> Author 映射表 (模型需要)
    # 逻辑: 每个 Item 只有一个 Author
    item2author_df = df_matrix[['i_idx', 'a_idx']].drop_duplicates().sort_values('i_idx')
    # 确保 item ID 是连续的，可以直接用 array 索引
    item2author_map = np.zeros(len(item_le.classes_), dtype=np.int64)
    item2author_map[item2author_df['i_idx'].values] = item2author_df['a_idx'].values
    
    # 6. 处理社交关系 (关键修复点!!!)
    print("--- 处理社交关系 (Alignment) ---")
    
    # 过滤 1: 只保留在我们 dataset 用户列表里的 follower
    valid_followers = df_social['user_id'].isin(user_le.classes_)
    # 过滤 2: 只保留在我们 dataset 作者列表里的 followee (被关注者)
    # 关键：我们只关心“关注了在这个数据集里发视频的人”
    valid_followees = df_social['friend_id'].isin(author_le.classes_)
    
    df_social_valid = df_social[valid_followers & valid_followees].copy()
    
    if len(df_social_valid) == 0:
        print("⚠️ 警告: Small Matrix 太小，过滤后没有剩余社交关系。建议换 Big Matrix。")
        social_edges = []
    else:
        # 核心映射：
        # Follower -> User ID Space
        # Followee -> Author ID Space
        u_social = user_le.transform(df_social_valid['user_id'])
        a_social = author_le.transform(df_social_valid['friend_id'])
        
        social_edges = list(zip(u_social, a_social))
        print(f"✅ 成功提取社交边: {len(social_edges)} 条")
        print(f"   (格式: User {u_social[0]} -> Author {a_social[0]})")

    # 7. 处理创作者分层 (Head/Tail) - 为了 Manager
    print("--- 计算创作者分层 ---")
    author_counts = df_matrix['a_idx'].value_counts()
    # 这里的阈值可以按分位数定，比如后 50% 是 Tail
    tail_threshold = author_counts.quantile(0.5) 
    
    # 0: Tail, 1: Mid, 2: Head
    # 先默认全为 0
    author_groups = np.zeros(len(author_le.classes_), dtype=np.int64)
    
    for aid, count in author_counts.items():
        if count <= tail_threshold:
            author_groups[aid] = 0 # Tail
        elif count <= author_counts.quantile(0.8):
            author_groups[aid] = 1 # Mid
        else:
            author_groups[aid] = 2 # Head
            
    print(f"Tail Authors: {(author_groups==0).sum()}")

    # 8. 处理用户活跃度 (User Active Level)
    # 简单起见，按交互数量分 4 档
    user_counts = df_matrix['u_idx'].value_counts()
    user_active_feature = np.zeros(len(user_le.classes_), dtype=np.int64)
    # 分位数: 0-40%, 40-70%, 70-90%, 90-100%
    q40 = user_counts.quantile(0.4)
    q70 = user_counts.quantile(0.7)
    q90 = user_counts.quantile(0.9)
    
    for uid, count in user_counts.items():
        if count <= q40: user_active_feature[uid] = 0
        elif count <= q70: user_active_feature[uid] = 1
        elif count <= q90: user_active_feature[uid] = 2
        else: user_active_feature[uid] = 3

    # 9. 切分 Train/Test
    # 简单 Leave-one-out 或 8:2
    print("--- 切分数据集 ---")
    # 这里简单做随机切分，实际可用时间戳
    all_indices = np.random.permutation(len(df_matrix))
    train_size = int(len(df_matrix) * 0.8)
    train_idx = all_indices[:train_size]
    test_idx = all_indices[train_size:]
    
    train_pairs = list(zip(df_matrix['u_idx'].values[train_idx], df_matrix['i_idx'].values[train_idx]))
    test_pairs = list(zip(df_matrix['u_idx'].values[test_idx], df_matrix['i_idx'].values[test_idx]))

    # 10. 保存
    data = {
        'num_users': len(user_le.classes_),
        'num_items': len(item_le.classes_),
        'num_authors': len(author_le.classes_),
        'num_active_levels': 4,
        'item2author': item2author_map,
        'author_groups': author_groups,
        'user_active_feature': user_active_feature,
        'train_pairs': train_pairs,
        'test_pairs': test_pairs,
        'social_edges': social_edges  # 这里的边已经是 User->Author 格式了
    }
    
    if not os.path.exists(os.path.dirname(OUTPUT_PATH)):
        os.makedirs(os.path.dirname(OUTPUT_PATH))
        
    with open(OUTPUT_PATH, 'wb') as f:
        pickle.dump(data, f)
        
    print(f"✅ 数据处理完成！保存至: {OUTPUT_PATH}")

if __name__ == '__main__':
    process_data_v3()