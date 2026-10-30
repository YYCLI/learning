import pandas as pd

def trim_idle_frames(episode_df, idle_threshold=3.0):
    if len(episode_df) == 0:
        return episode_df
    action_x = episode_df['action'].apply(lambda x: x[0])
    dx = action_x.diff().abs()
    active_indices = episode_df.index[dx > idle_threshold]
    if len(active_indices) == 0:
        return episode_df.iloc[0:0]
        
    start_idx = active_indices[0]
    end_idx = active_indices[-1]
    start_loc = episode_df.index.get_loc(start_idx)
    end_loc = episode_df.index.get_loc(end_idx)
    final_start = max(0, start_loc - 5)
    final_end = min(len(episode_df), end_loc + 6)
    return episode_df.iloc[final_start:final_end]

def extract_golden_data(file_path, output_path):
    print("\n开始提取黄金数据集(切除冗余帧)...")
    df = pd.read_parquet(file_path)
    golden_df = df.groupby('episode_index', group_keys=False).apply(trim_idle_frames)
    golden_df.to_parquet(output_path, index=False)
    print(f"切除了 {(len(df) - len(golden_df)) / len(df) * 100:.2f}% 的冗余数据。")
    print(f"黄金数据集已保存至 {output_path}")

if __name__ == "__main__":
    input_file = r"my_robot_data\data\chunk-000\cleaned_data.parquet"
    output_file = r"my_robot_data\data\chunk-000\golden_data.parquet"
    extract_golden_data(input_file, output_file)