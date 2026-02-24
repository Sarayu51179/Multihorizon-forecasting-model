import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


DATA_PATH = "data/final/master_dataset.csv"


def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    print("\nEvaluation Metrics:")
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"MAPE : {mape:.4f}%")

    return mae, rmse, mape


def main():

    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)

    TARGET = "Target_1D"

    # -----------------------------
    # Remove infinite values
    # -----------------------------
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    # -----------------------------
    # Separate features & target
    # -----------------------------
    X = df.drop(columns=["Target_1D", "Target_7D", "Target_30D"])
    y = df[TARGET]

    # -----------------------------
    # Chronological Split (70/30)
    # -----------------------------
    split_index = int(len(df) * 0.7)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    print(f"\nTrain size: {len(X_train)}")
    print(f"Test size : {len(X_test)}")

    # -----------------------------
    # Model
    # -----------------------------
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )

    print("\nTraining RandomForest...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    evaluate_model(y_test, y_pred)

    # -----------------------------
    # Feature Importance
    # -----------------------------
    importance = pd.Series(
        model.feature_importances_,
        index=X.columns
    ).sort_values(ascending=False)

    print("\nTop 10 Feature Importances:")
    print(importance.head(10))


if __name__ == "__main__":
    main()
