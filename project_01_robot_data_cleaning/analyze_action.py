import pandas as pd
import matplotlib.pyplot as plt # 用来画图

# 1. 读取数据
file_path = r"my_robot_data\data\chunk-000\file-000.parquet" 
df = pd.read_parquet(file_path)

# 2. 我们只看第一个 episode（片段）
episode_0 = df[df['episode_index'] == 0].copy()

# 3. 看看时间戳和动作（action）长什么样
# action 是一个数组，包含了机器人各个关节要动的目标值，我们算一下它的变化幅度
# 假设 action 有多个维度，我们求一下当前帧和上一帧动作的差值的绝对值（这就是速度）
# 这里我们取 action 的第一维和第二维（通常是 x 和 y 轴）
episode_0['action_x'] = episode_0['action'].apply(lambda x: x[0])
episode_0['action_y'] = episode_0['action'].apply(lambda x: x[1])

# 4. 计算相邻两帧的动作变化量（寻找“尖峰”）
episode_0['dx'] = episode_0['action_x'].diff().abs()
episode_0['dy'] = episode_0['action_y'].diff().abs()

# 5. 找出异常突变（比如变化量大于某个阈值）
threshold = 0.05 # 这个阈值可以先随便设，后面可以调
spikes = episode_0[(episode_0['dx'] > threshold) | (episode_0['dy'] > threshold)]
print(f"在这个片段中，发现了 {len(spikes)} 处动作突变！")
print(spikes[['timestamp', 'action_x', 'action_y', 'dx', 'dy']].head(10))

# 6. 画个图看看（如果你装了 matplotlib）
plt.plot(episode_0['timestamp'], episode_0['action_x'], label='Action X')
plt.plot(episode_0['timestamp'], episode_0['action_y'], label='Action Y')
plt.legend()
plt.title("Robot Action Changes")
plt.show()