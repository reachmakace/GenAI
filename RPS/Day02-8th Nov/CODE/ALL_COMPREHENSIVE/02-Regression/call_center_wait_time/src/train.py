"""Train models for call center wait-time estimation

Usage:
    python src/train.py --data data/sample.csv

If `--data` is omitted, the script will call the generator to create a synthetic dataset.
"""
import argparse
from pathlib import Path
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor

from evaluate import regression_metrics

PROJECT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT / 'data'
MODELS_DIR = PROJECT / 'models'
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def load_data(path=None):
    if path:
        df = pd.read_csv(path)
    else:
        # generate synthetic
        from data.generate_synthetic import generate
        df = generate(10000)
    return df


def preprocess(df):
    X = df.drop(columns=['wait_time'])
    y = df['wait_time'].values

    # numeric and categorical features
    numeric = ['hour', 'day_of_week', 'queue_length', 'agents_on_shift']
    categorical = ['call_type']

    pre = ColumnTransformer([
        ('num', StandardScaler(), numeric),
        ('cat', OneHotEncoder(sparse=False, handle_unknown='ignore'), categorical)
    ])

    X_trans = pre.fit_transform(X)
    feature_names = numeric + list(pre.named_transformers_['cat'].get_feature_names_out(categorical))
    return X_trans, y, pre, feature_names


def train_and_eval(df):
    X, y, pre, feature_names = preprocess(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Linear regression baseline
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    metrics_lr = regression_metrics(y_test, y_pred_lr)

    # Gradient Boosting Regressor
    gbr = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=4, random_state=42)
    gbr.fit(X_train, y_train)
    y_pred_gbr = gbr.predict(X_test)
    metrics_gbr = regression_metrics(y_test, y_pred_gbr)

    # Save models and preprocessor
    joblib.dump({'model': lr, 'pre': pre, 'features': feature_names}, MODELS_DIR / 'linear_regression.joblib')
    joblib.dump({'model': gbr, 'pre': pre, 'features': feature_names}, MODELS_DIR / 'gbr.joblib')

    return metrics_lr, metrics_gbr


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=None, help='Path to CSV data file')
    args = parser.parse_args()

    data_path = args.data
    if data_path:
        data_path = PROJECT / data_path if not Path(data_path).is_absolute() else Path(data_path)

    df = load_data(data_path)
    print(f'Loaded data with {len(df)} rows')
    m_lr, m_gbr = train_and_eval(df)

    print('\nLinear Regression metrics:')
    for k, v in m_lr.items():
        print(f'  {k}: {v:.4f}')

    print('\nGradient Boosting Regression metrics:')
    for k, v in m_gbr.items():
        print(f'  {k}: {v:.4f}')

    print('\nSaved models to', MODELS_DIR)
