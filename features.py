import pandas as pd
import numpy as np

def add_differencing(df, col, order=1):
    """Add differencing feature."""
    df[f'{col}_diff_{order}'] = df[col].diff(order)
    return df

def add_rolling_mean(df, col, n):
    """Add rolling mean feature."""
    df[f'{col}_rolling_mean_{n}'] = df[col].rolling(window=n).mean()
    return df

def add_rolling_std(df, col, n):
    """Add rolling std feature."""
    df[f'{col}_rolling_std_{n}'] = df[col].rolling(window=n).std()
    return df

def add_rolling_max(df, col, n):
    """Add rolling max feature."""
    df[f'{col}_rolling_max_{n}'] = df[col].rolling(window=n).max()
    return df

def add_rolling_min(df, col, n):
    """Add rolling min feature."""
    df[f'{col}_rolling_min_{n}'] = df[col].rolling(window=n).min()
    return df

def add_positive_cumulative(df, col, n):
    """Add positive cumulative feature."""
    df[f'{col}_pos_cum_{n}'] = df[col].clip(lower=0).rolling(window=n).sum()
    return df

def add_negative_cumulative(df, col, n):
    """Add negative cumulative feature."""
    df[f'{col}_neg_cum_{n}'] = df[col].clip(upper=0).rolling(window=n).sum()
    return df

def add_grouped_rank(df, col, group_cols, time_col):
    """Add grouped cross-sectional rank."""
    df[f'{col}_rank'] = df.groupby(group_cols + [time_col])[col].rank()
    return df

def add_grouped_zscore(df, col, group_cols, time_col):
    """Add grouped cross-sectional z-score."""
    mean = df.groupby(group_cols + [time_col])[col].transform('mean')
    std = df.groupby(group_cols + [time_col])[col].transform('std')
    df[f'{col}_zscore'] = (df[col] - mean) / std
    return df

def add_grouped_correlation_weighted_feature(df, col, group_cols, time_col):
    """Add grouped cross-sectional correlation weighted feature."""
    def weighted_sum(group):
        if len(group) <= 1:
            group[f'{col}_corr_weighted'] = group[col]
            return group
        
        corr_matrix = group[col].corr(group[col]) 

        for idx in group.index:
            others = group.index != idx
            weights = group.loc[others, col].corr(group.loc[idx, col]) if len(group) > 1 else 1
           
            group_mean = group[col].mean()
            
            correlations = group[col].corr(group_mean)  
            
            corr_matrix = group[[col]].corr()  
            
            group[f'{col}_corr_weighted'] = group[col].rolling(window=len(group), center=False).mean().shift(1)  # Not accurate
            
            group[f'{col}_corr_weighted'] = group.groupby(group_cols + [time_col])[col].transform(lambda x: (x.sum() - x) / (len(x) - 1) if len(x) > 1 else x)
            return group
    df = df.groupby(group_cols + [time_col]).apply(weighted_sum).reset_index(drop=True)
    return df