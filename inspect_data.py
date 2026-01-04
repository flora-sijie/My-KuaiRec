import pandas as pd
import os

# 设置数据目录
data_dir = './data'

def inspect_all_data(directory):
    # 1. 获取目录下所有 CSV 文件
    all_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    
    if not all_files:
        print(f"❌ 在 {directory} 下没有找到 CSV 文件，请检查路径。")
        return

    print(f"🔎 发现 {len(all_files)} 个数据文件，准备逐一检查...\n")
    
    # 2. 遍历检查
    for file_name in sorted(all_files): # 排序一下，看着整齐
        file_path = os.path.join(directory, file_name)
        print(f"{'='*30}")
        print(f"📄 文件名: {file_name}")
        
        try:
            # 只读取前 3 行，极速预览
            df = pd.read_csv(file_path, nrows=3)
            
            # 打印列名（这对我们最重要，用来推测文件用途）
            print(f"📌 列名 ({len(df.columns)}列):")
            print(list(df.columns))
            
            # 打印少量数据样本
            print(f"👀 数据预览:")
            print(df.to_string(index=False)) # to_string 防止打印太宽被折叠
            
            # ------------------------------------------------------
            # 智能提示：根据你的模型需求，自动高亮关键字段
            # ------------------------------------------------------
            cols = set(df.columns)
            
            # 1. 找作者 (For Part C & GNN)
            if any(x in cols for x in ['author_id', 'uploader_id', 'owner_id']):
                print(f"   ✅ [关键] 发现疑似【创作者ID】字段！")
                
            # 2. 找社交关系 (For Part C)
            if 'friend_id' in cols or 'follow' in cols:
                print(f"   ✅ [关键] 发现疑似【社交关系】字段！")
                
            # 3. 找用户活跃度特征 (For Manager Input)
            if any(x in cols for x in ['active_level', 'view_count', 'interaction_count']):
                print(f"   ✅ [关键] 发现疑似【用户活跃度】特征，可用于 Manager 输入！")

            # 4. 找文本/类别特征 (For GNN Init)
            if any(x in cols for x in ['caption', 'title', 'category', 'tags']):
                print(f"   ✅ [关键] 发现疑似【内容语义】特征，可用于初始化 Item Embedding！")
                
        except Exception as e:
            print(f"⚠️ 读取失败: {e}")
        
        print("\n")

# 执行
inspect_all_data(data_dir)