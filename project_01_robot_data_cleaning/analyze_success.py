import pandas as pd

file_path = r"my_robot_data\data\chunk-000\cleaned_data.parquet" 
df = pd.read_parquet(file_path)

print("=== 正在分析 206 个片段的业务数据 ===")

episode_stats = df.groupby('episode_index').agg(
    total_frames=('timestamp', 'count'),       
    max_reward=('next.reward', 'max')          
).reset_index()

# 【关键修复】我们把标准降到 0.90
THRESHOLD = 0.90
episode_stats['is_success'] = (episode_stats['max_reward'] >= THRESHOLD).astype(int)

success_count = episode_stats['is_success'].sum()
total_episodes = len(episode_stats)
success_rate = success_count / total_episodes * 100

print(f"\n[设定阈值] >= {THRESHOLD}")
print(f"[整体战绩] 总片段数: {total_episodes}")
print(f"[整体战绩] 成功片段数: {success_count}")
print(f"[整体战绩] 任务成功率: {success_rate:.2f}%")

avg_frames = episode_stats.groupby('is_success')['total_frames'].mean()
print("\n[耗时分析] 成功局的平均帧数:", int(avg_frames.get(1, 0)))
print("[耗时分析] 失败局的平均帧数:", int(avg_frames.get(0, 0)))
print("*(注：一般每秒有10帧，除以10大概就是真实秒数)*")