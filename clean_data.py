import pandas as pd

# 1. 读取数据
file_path = r"my_robot_data\data\chunk-000\file-000.parquet" 
df = pd.read_parquet(file_path)
original_length = len(df)

# 2. 提取 action
df['action_x'] = df['action'].apply(lambda x: x[0])
df['action_y'] = df['action'].apply(lambda x: x[1])

# 3. 按 episode 分组计算差值（避免首尾相接造成的假突变）
df['dx'] = df.groupby('episode_index')['action_x'].diff().abs()
df['dy'] = df.groupby('episode_index')['action_y'].diff().abs()

# 4. 动态计算阈值（利用刚才看到的 99.9% 分位数）
threshold_x = df['dx'].quantile(0.999)
threshold_y = df['dy'].quantile(0.999)

# 5. 真正的数据清洗：只保留正常数据（同时要保留每个 episode 的第一帧，因为它的 dx 是 NaN）
clean_df = df[
    ((df['dx'] <= threshold_x) | (df['dx'].isna())) & 
    ((df['dy'] <= threshold_y) | (df['dy'].isna()))
].copy()

# 6. 删掉辅助列，恢复原样
clean_df = clean_df.drop(columns=['action_x', 'action_y', 'dx', 'dy'])

# 7. 保存干净的数据
output_path = r"my_robot_data\data\chunk-000\cleaned_data.parquet"
clean_df.to_parquet(output_path, index=False)

# 8. 打印完美报告
print("=== 数据清洗完成 ===")
print(f"原始数据行数: {original_length}")
print(f"清洗后行数: {len(clean_df)}")
print(f"剔除了异常尖峰: {original_length - len(clean_df)} 行")
print(f"数据保留率: {len(clean_df) / original_length * 100:.2f}%")