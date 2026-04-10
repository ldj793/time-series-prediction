import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
import optuna
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense
from sklearn.preprocessing import StandardScaler

def train_model(df_train, target_col, feature_cols, horizon, model_type, tune=True, top_n_features=None):
    """Train model for a specific horizon."""
    df_h = df_train[df_train['horizon'] == horizon].copy()
    X = df_h[feature_cols]
    y = df_h[target_col]
    
    if model_type in ['xgboost', 'lightgbm', 'catboost']:
        return train_tree_model(X, y, model_type, tune, top_n_features)
    elif model_type in ['lstm', 'gru']:
        return train_nn_model(X, y, model_type, tune)
    else:
        raise ValueError("Unsupported model type")

def train_tree_model(X, y, model_type, tune, top_n_features=None):
    """Train tree-based model with tuning."""
    if model_type == 'xgboost':
        model = xgb.XGBRegressor()
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5],
            'learning_rate': [0.01, 0.1],
            'min_child_weight': [1, 5],
            'reg_lambda': [0, 1],
            'reg_alpha': [0, 1]
        }
    elif model_type == 'lightgbm':
        model = lgb.LGBMRegressor()
        param_grid = {
            'n_estimators': [100, 200],
            'num_leaves': [31, 50],
            'learning_rate': [0.01, 0.1],
            'min_data_in_leaf': [20, 50],
            'lambda_l1': [0, 1],
            'lambda_l2': [0, 1]
        }
    elif model_type == 'catboost':
        model = cb.CatBoostRegressor(verbose=0)
        param_grid = {
            'iterations': [100, 200],
            'depth': [3, 5],
            'learning_rate': [0.01, 0.1],
            'min_data_in_leaf': [1, 5],
            'l2_leaf_reg': [0, 1],
            'reg_lambda': [0, 1]  # Assuming reg_lambda for l1
        }
    
    if tune:
        # Grid search
        grid = GridSearchCV(model, param_grid, cv=3)
        grid.fit(X, y)
        best_model = grid.best_estimator_
        
        # Bayesian optimization with optuna
        def objective(trial):
            params = {}
            if model_type == 'xgboost':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 5)
                }
            elif model_type == 'lightgbm':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 100),
                    'lambda_l1': trial.suggest_float('lambda_l1', 0, 5),
                    'lambda_l2': trial.suggest_float('lambda_l2', 0, 5)
                }
            elif model_type == 'catboost':
                params = {
                    'iterations': trial.suggest_int('iterations', 50, 300),
                    'depth': trial.suggest_int('depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 10),
                    'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0, 5),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 5)
                }
            model.set_params(**params)
            model.fit(X, y)
            return -model.score(X, y)  # minimize negative score
        
        study = optuna.create_study()
        study.optimize(objective, n_trials=10)
        best_params = study.best_params
        best_model.set_params(**best_params)
        best_model.fit(X, y)
    else:
        best_model = model.fit(X, y)
    
    # Feature selection
    if top_n_features and hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
        indices = np.argsort(importances)[-top_n_features:]
        selected_features = X.columns[indices]
        X_selected = X[selected_features]
        best_model.fit(X_selected, y)
        return best_model, selected_features
    else:
        return best_model

def train_nn_model(X, y, model_type, tune, seq_length=1, learning_rate=0.001, batch_size=32, dropout_rate=0.2, l1=0.0, l2=0.0):
    """Train neural network model."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_seq = X_scaled.reshape((X_scaled.shape[0], seq_length, X_scaled.shape[1] // seq_length if seq_length > 1 else X_scaled.shape[1]))  # Adjust for seq_length
    
    from tensorflow.keras.layers import Dropout
    from tensorflow.keras.regularizers import l1_l2
    from tensorflow.keras.optimizers import Adam
    
    model = Sequential()
    if model_type == 'lstm':
        model.add(LSTM(50, input_shape=(X_seq.shape[1], X_seq.shape[2]), kernel_regularizer=l1_l2(l1=l1, l2=l2)))
    elif model_type == 'gru':
        model.add(GRU(50, input_shape=(X_seq.shape[1], X_seq.shape[2]), kernel_regularizer=l1_l2(l1=l1, l2=l2)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1))
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse')
    
    model.fit(X_seq, y, epochs=10, batch_size=batch_size, verbose=0)
    return model, scaler