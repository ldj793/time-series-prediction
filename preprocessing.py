import pandas as pd
import numpy as np

def reduce_memory_usage(df):
    """Reduce memory usage by converting data types."""
    for col in df.columns:
        if df[col].dtype == 'float64':
            df[col] = df[col].astype('float32')
        elif df[col].dtype == 'int64':
            df[col] = df[col].astype('int32')
    return df

def add_log_smoothing(df, cols):
    """Add log smoothing to specified columns."""
    for col in cols:
        df[f'{col}_log'] = np.log(df[col] + 1)
    return df

def cap_outliers(df, cols, std_threshold=3):
    """Cap outliers using std threshold."""
    for col in cols:
        mean = df[col].mean()
        std = df[col].std()
        upper = mean + std_threshold * std
        lower = mean - std_threshold * std
        df[col] = np.clip(df[col], lower, upper)
    return df

def handle_missing_values(df, cols, group_cols):
    """Handle missing values: group mean if <20% missing, else add flag."""
    for col in cols:
        missing_rate = df[col].isnull().mean()
        if missing_rate < 0.2:
            df[col] = df.groupby(group_cols)[col].transform(lambda x: x.fillna(x.mean()))
        else:
            df[f'{col}_missing'] = df[col].isnull().astype(int)
            df[col] = df[col].fillna(0)  # or some default
    return df

def mark_anomalies(df, cols):
    """Mark anomalies similar to missing."""
    for col in cols:
        # Simple anomaly detection, e.g., based on std
        mean = df[col].mean()
        std = df[col].std()
        df[f'{col}_anomaly'] = ((df[col] - mean).abs() > 3 * std).astype(int)
    return df