import pandas as pd
import matplotlib.pyplot as plt

# 读取你刚生成的干净数据
file_path = r"my_robot_data\data\chunk-000\cleaned_data.parquet" 
df = pd.read_parquet(file_path)

# 获取第一个 episode 的数据，方便画图
ep0 = df[df['episode_index'] == 0].copy()

# 提取 X 轴坐标
ep0['action_x'] = ep0['action'].apply(lambda x: x[0])
ep0['state_x'] = ep0['observation.state'].apply(lambda x: x[0])

# 画图：只画前 100 帧，不然线太密看不清
plt.figure(figsize=(10, 5))
plt.plot(ep0['timestamp'][:100], ep0['action_x'][:100], label='Action X (目标位置)', marker='.', linestyle='--')
plt.plot(ep0['timestamp'][:100], ep0['state_x'][:100], label='State X (实际位置)', alpha=0.7)
plt.xlabel('Timestamp')
plt.ylabel('X Coordinate')
plt.title('Robot Action vs State')
plt.legend()
plt.show()