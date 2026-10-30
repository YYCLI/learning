import pandas as pd

def analyze_success_rate(file_path):
    print("\n开始分析任务成功率...")
    df = pd.read_parquet(file_path)
    
    episode_stats = df.groupby('episode_index').agg(
        total_frames=('timestamp', 'count'),       
        max_reward=('next.reward', 'max')          
    ).reset_index()
    
    THRESHOLD = 0.90
    episode_stats['is_success'] = (episode_stats['max_reward'] >= THRESHOLD).astype(int)
    
    success_count = episode_stats['is_success'].sum()
    total_episodes = len(episode_stats)
    success_rate = success_count / total_episodes * 100
    
    print(f"任务成功率: {success_rate:.2f}%")
    
    avg_frames = episode_stats.groupby('is_success')['total_frames'].mean()
    print(f"成功局平均帧数: {int(avg_frames.get(1, 0))}")
    print(f"失败局平均帧数: {int(avg_frames.get(0, 0))}")

if __name__ == "__main__":
    input_file = r"my_robot_data\data\chunk-000\cleaned_data.parquet"
    analyze_success_rate(input_file)