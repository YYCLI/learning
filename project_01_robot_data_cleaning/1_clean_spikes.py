import pandas as pd

def clean_outliers(file_path, output_path):
    print("开始清洗异常突变数据...")
    df = pd.read_parquet(file_path)
    
    df['action_x'] = df['action'].apply(lambda x: x[0])
    df['action_y'] = df['action'].apply(lambda x: x[1])
    
    df['dx'] = df.groupby('episode_index')['action_x'].diff().abs()
    df['dy'] = df.groupby('episode_index')['action_y'].diff().abs()
    
    # 采用 99.9% 分位数作为动态阈值
    threshold_x = df['dx'].quantile(0.999)
    threshold_y = df['dy'].quantile(0.999)
    
    clean_df = df[
        ((df['dx'] <= threshold_x) | (df['dx'].isna())) & 
        ((df['dy'] <= threshold_y) | (df['dy'].isna()))
    ].copy()
    
    clean_df = clean_df.drop(columns=['action_x', 'action_y', 'dx', 'dy'])
    clean_df.to_parquet(output_path, index=False)
    print(f"清洗完成！原始 {len(df)} 帧，清洗后 {len(clean_df)} 帧。保存至 {output_path}")

if __name__ == "__main__":
    # 注意：这里的路径要改成你电脑里真实的路径
    input_file = r"my_robot_data\data\chunk-000\file-000.parquet"
    output_file = r"my_robot_data\data\chunk-000\cleaned_data.parquet"
    clean_outliers(input_file, output_file)