import pandas as pd

# 注意：路径里用了你新建的文件夹名，并且用 r"" 来防止转义
file_path = r"my_robot_data\data\chunk-000\file-000.parquet" 

# 如果你的文件名不叫 file-000.parquet，请在上面替换成真实的名字

df = pd.read_parquet(file_path)

print("行数:", len(df))
print("列数:", len(df.columns))
print("\n列名:")
for col in df.columns:
    print("-", col)

# 额外加一段：看看动作和状态的数值长什么样
print("\n前3行数据（部分列）:")
print(df[['timestamp', 'observation.state', 'action']].head(3))