import pandas as pd
from .preprocessing import reduce_memory_usage, add_log_smoothing, cap_outliers, handle_missing_values, mark_anomalies
from .features import add_differencing, add_rolling_mean, add_rolling_std, add_rolling_max, add_rolling_min, add_positive_cumulative, add_negative_cumulative, add_grouped_rank, add_grouped_zscore
from .model_training import train_model

def check_time_sorting(df, time_col):
    """Check if time is sorted ascending."""
    if not df[time_col].is_monotonic_increasing:
        df = df.sort_values(time_col).reset_index(drop=True)
    return df

def predict_time_series(train_df, test_df, primary_key, time_col, feature_cols, group_cols, horizon_col, target_col, preprocessing_options, feature_additions, model_choice, top_n_features=None, nn_params=None):
    """Main function to predict."""
    # Preprocessing
    if preprocessing_options.get('reduce_memory'):
        train_df = reduce_memory_usage(train_df)
        test_df = reduce_memory_usage(test_df)
    
    if 'log_smooth' in preprocessing_options:
        train_df = add_log_smoothing(train_df, preprocessing_options['log_smooth'])
        test_df = add_log_smoothing(test_df, preprocessing_options['log_smooth'])
    
    if 'cap_outliers' in preprocessing_options:
        opts = preprocessing_options['cap_outliers']
        train_df = cap_outliers(train_df, opts['cols'], opts.get('std', 3))
        test_df = cap_outliers(test_df, opts['cols'], opts.get('std', 3))
    
    if preprocessing_options.get('handle_missing'):
        train_df = handle_missing_values(train_df, feature_cols, group_cols)
        test_df = handle_missing_values(test_df, feature_cols, group_cols)
    
    if preprocessing_options.get('mark_anomalies'):
        train_df = mark_anomalies(train_df, feature_cols)
        test_df = mark_anomalies(test_df, feature_cols)
    
    # Sort time
    train_df = check_time_sorting(train_df, time_col)
    test_df = check_time_sorting(test_df, time_col)
    
    # Add features
    for feat in feature_additions:
        if feat['type'] == 'differencing':
            train_df = add_differencing(train_df, feat['col'], feat.get('order', 1))
            test_df = add_differencing(test_df, feat['col'], feat.get('order', 1))
        elif feat['type'] == 'rolling_mean':
            train_df = add_rolling_mean(train_df, feat['col'], feat['n'])
            test_df = add_rolling_mean(test_df, feat['col'], feat['n'])
        elif feat['type'] == 'grouped_rank':
            train_df = add_grouped_rank(train_df, feat['col'], group_cols, time_col)
            test_df = add_grouped_rank(test_df, feat['col'], group_cols, time_col)
        elif feat['type'] == 'grouped_zscore':
            train_df = add_grouped_zscore(train_df, feat['col'], group_cols, time_col)
            test_df = add_grouped_zscore(test_df, feat['col'], group_cols, time_col)
        elif feat['type'] == 'grouped_correlation_weighted':
            train_df = add_grouped_correlation_weighted_feature(train_df, feat['col'], group_cols, time_col)
            test_df = add_grouped_correlation_weighted_feature(test_df, feat['col'], group_cols, time_col)
        # Add others similarly
    
    # Update feature_cols with new features
    all_features = [col for col in train_df.columns if col not in [primary_key, time_col, target_col, horizon_col] + group_cols]
    
    # Group by horizon and train
    horizons = train_df[horizon_col].unique()
    models = {}
    selected_features = {}
    for h in horizons:
        if model_choice in ['xgboost', 'lightgbm', 'catboost']:
            model_result = train_model(train_df, target_col, all_features, h, model_choice, top_n_features=top_n_features)
            if isinstance(model_result, tuple):
                models[h] = model_result[0]
                selected_features[h] = model_result[1]
            else:
                models[h] = model_result
                selected_features[h] = all_features
        elif model_choice in ['lstm', 'gru']:
            seq_length = nn_params.get('seq_length', 1) if nn_params else 1
            learning_rate = nn_params.get('learning_rate', 0.001) if nn_params else 0.001
            batch_size = nn_params.get('batch_size', 32) if nn_params else 32
            dropout_rate = nn_params.get('dropout_rate', 0.2) if nn_params else 0.2
            l1 = nn_params.get('l1', 0.0) if nn_params else 0.0
            l2 = nn_params.get('l2', 0.0) if nn_params else 0.0
            models[h] = train_model(train_df, target_col, all_features, h, model_choice, seq_length=seq_length, learning_rate=learning_rate, batch_size=batch_size, dropout_rate=dropout_rate, l1=l1, l2=l2)
            selected_features[h] = all_features
    
    # Predict
    predictions = []
    for h in horizons:
        test_h = test_df[test_df[horizon_col] == h]
        if not test_h.empty:
            X_test = test_h[selected_features[h]]
            if model_choice in ['xgboost', 'lightgbm', 'catboost']:
                pred = models[h].predict(X_test)
            elif model_choice in ['lstm', 'gru']:
                model, scaler = models[h]
                X_scaled = scaler.transform(X_test)
                seq_length = nn_params.get('seq_length', 1) if nn_params else 1
                X_seq = X_scaled.reshape((X_scaled.shape[0], seq_length, X_scaled.shape[1] // seq_length if seq_length > 1 else X_scaled.shape[1]))
                pred = model.predict(X_seq).flatten()
            test_h = test_h.copy()
            test_h['prediction'] = pred
            predictions.append(test_h[[primary_key, 'prediction']])
    
    result = pd.concat(predictions, ignore_index=True)
    return result