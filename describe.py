import pandas as pd

# 1. 读取清洗前的原始数据
file_path = r"my_robot_data\data\chunk-000\file-000.parquet" 
df = pd.read_parquet(file_path)

# 2. 把 X 和 Y 坐标从数组里抠出来
df['action_x'] = df['action'].apply(lambda x: x[0])
df['action_y'] = df['action'].apply(lambda x: x[1])

# 3. 按局（episode）分组，计算相邻帧的变化幅度（差值），并取绝对值
df['dx'] = df.groupby('episode_index')['action_x'].diff().abs()
df['dy'] = df.groupby('episode_index')['action_y'].diff().abs()

# 4. 核心：用 describe() 打印统计分布
print("=== 看看真实的数据差值分布 ===")
print("dx (X轴变化值) 的统计描述:")
# percentiles 参数指定了我们要看哪几个百分位，比如 .999 就是 99.9%
print(df['dx'].describe(percentiles=[.50, .90, .95, .99, .999]))

print("\ndy (Y轴变化值) 的统计描述:")
print(df['dy'].describe(percentiles=[.50, .90, .95, .99, .999]))

# 5. 提取 99.9% 分位数作为我们要用的动态阈值
threshold_x = df['dx'].quantile(0.999)
threshold_y = df['dy'].quantile(0.999)

print(f"\n最终确定的动态清洗阈值: X轴>{threshold_x:.2f}, Y轴>{threshold_y:.2f} 认为是异常瞬移")