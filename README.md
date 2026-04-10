# 时间序列预测项目

这是一个用于时间序列预测的Python项目，支持特征工程、数据预处理、多种模型选择、训练和调参。训练按horizon分组进行，每个horizon训练单独的模型。

## 功能介绍

### 特征工程 (features.py)
- 差分 (differencing)：可选阶数
- 前n日均值 (rolling mean)
- 前n日标准差 (rolling std)
- 前n日最大值 (rolling max)
- 前n日最小值 (rolling min)
- 前n日正值累积 (positive cumulative)
- 前n日负值累积 (negative cumulative)
- 分组截面rank (grouped cross-sectional rank)
- 分组截面z_score (grouped cross-sectional z-score，可变分组参数)
- 分组截面相关性加权特征 (grouped cross-sectional correlation weighted feature)

### 预处理 (preprocessing.py)
- 数据精度转化 (reduce memory)：减少内存使用
- 添加数据对数平滑 (log smoothing)
- 极端值缩尾 (outlier capping)：std可变，默认为3
- 缺失值处理：缺失<20%的特征使用组内均值，>20%添加缺失标记
- 异常值标记 (anomaly marking)：同缺失值

### 模型选择和训练调参 (model_training.py)
支持模型：
- XGBoost
- LightGBM
- CatBoost
- LSTM
- GRU

调参方法：
- 先进行网格搜索 (grid search)
- 然后贝叶斯优化 (Bayesian optimization with TPE算法)

训练按horizon分组，每个horizon单独训练模型。

### 主文件 (main.py)
接受输入：
- 训练DataFrame
- 测试DataFrame
- 主键列
- 时间列
- 特征列列表
- 分组列列表
- horizon列
- 目标列
- 预处理选项
- 特征添加选项
- 模型选择

输出：主键和预测值两列的DataFrame

## 安装

1. 克隆项目：
   ```
   git clone <repository-url>
   cd 时间序列预测
   ```

2. 创建虚拟环境：
   ```
   python -m venv time
   .\time\Scripts\Activate.ps1  # Windows PowerShell
   ```

3. 安装依赖：
   ```
   pip install -r requirements.txt
   ```

## 使用说明

### 示例代码

```python
import pandas as pd
from src.main import predict_time_series

# 准备数据
train_df = pd.DataFrame({
    'primary_key': ['A1', 'A2', 'B1', 'B2'],
    'time_col': [1, 2, 1, 2],
    'group_col': ['G1', 'G1', 'G2', 'G2'],
    'feature1': [10, 12, 15, 18],
    'horizon': [1, 1, 2, 2],
    'target': [11, 13, 16, 19]
})

test_df = pd.DataFrame({
    'primary_key': ['A3', 'B3'],
    'time_col': [3, 3],
    'group_col': ['G1', 'G2'],
    'feature1': [14, 20],
    'horizon': [1, 2]
})

# 定义选项
preprocessing_options = {
    'reduce_memory': True,
    'log_smooth': ['feature1'],
    'cap_outliers': {'cols': ['feature1'], 'std': 3},
    'handle_missing': True,
    'mark_anomalies': True
}

feature_additions = [
    {'type': 'rolling_mean', 'col': 'feature1', 'n': 2},
    {'type': 'grouped_rank', 'col': 'feature1', 'group_cols': ['group_col']},
    {'type': 'grouped_zscore', 'col': 'feature1', 'group_cols': ['group_col']},
    {'type': 'grouped_correlation_weighted', 'col': 'feature1', 'group_cols': ['group_col']}
]

# 运行预测
result = predict_time_series(
    train_df, test_df, 'primary_key', 'time_col', ['feature1'], ['group_col'], 'horizon', 'target',
    preprocessing_options, feature_additions, 'xgboost', top_n_features=10, nn_params={'seq_length': 5, 'learning_rate': 0.01, 'batch_size': 64, 'dropout_rate': 0.3, 'l1': 0.001, 'l2': 0.001}
)

print(result)
```

### 参数说明

- `train_df`: 训练数据DataFrame，包含horizon列
- `test_df`: 测试数据DataFrame，包含horizon列
- `primary_key`: 主键列名
- `time_col`: 时间列名
- `feature_cols`: 特征列名列表
- `group_cols`: 分组列名列表
- `horizon_col`: horizon列名
- `target_col`: 目标列名
- `preprocessing_options`: 预处理选项字典
- `feature_additions`: 特征添加列表，每个元素为字典
- `model_choice`: 模型选择 ('xgboost', 'lightgbm', 'catboost', 'lstm', 'gru')
- `top_n_features`: (可选) 选择top n重要特征，仅用于树模型
- `nn_params`: (可选) 神经网络参数字典，包括 'seq_length', 'learning_rate', 'batch_size', 'dropout_rate', 'l1', 'l2'

### 输出
DataFrame with columns: primary_key, prediction

## 测试

运行测试：
```
python -m pytest tests/ -v
```

## 贡献

欢迎提交issue和pull request。

## 许可证

MIT License