import pandas as pd
import os

data_dir = './data'
file_path = os.path.join(data_dir, 'user_features.csv')

print(f"🕵️ 正在深度检查: {file_path}")
df = pd.read_csv(file_path, nrows=10) # 看前10行

# 1. 检查所有包含 'follow' 的列
follow_cols = [c for c in df.columns if 'follow' in c]
print(f"\n📌 包含 'follow' 关键词的列: {follow_cols}")

# 2. 打印这些列的具体值，看看是 List 还是 Int
print("\n👀 关键列前 5 行预览:")
print(df[follow_cols + ['user_active_degree']].head(5))

# 3. 验证数据类型
print("\nDataType 检查:")
for col in follow_cols:
    sample_val = df[col].iloc[0]
    print(f" - {col}: 值示例 '{sample_val}' (类型: {type(sample_val)})")

print("\n------------------------------------------------")
print("结论预测：")
print("如果值是 '5' (int)，那它只是统计数，不能用来建图。")
print("如果值是 '[123, 456]' (str/list)，那它才是我们需要的一度人脉关系。")