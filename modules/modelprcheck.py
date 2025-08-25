from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.outliers_influence import variance_inflation_factor

def check_regression_assumptions(X, y_test, y_pred):
    """
    Check key Linear Regression assumptions:
    1. Linearity
    2. Normality of Residuals
    3. Homoscedasticity
    4. Independence of Errors
    5. Multicollinearity
    """

    residuals = y_test - y_pred

    # 1. Linearity: Actual vs Predicted
    plt.figure(figsize=(6,6))
    sns.scatterplot(x=y_test, y=y_pred)
    plt.plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()], 'r--')
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Actual vs Predicted (Linearity Check)")
    plt.show()

    # 2. Normality of Residuals
    plt.figure(figsize=(6,4))
    sns.histplot(residuals, bins=30, kde=True)
    plt.title("Residuals Distribution (Normality Check)")
    plt.show()

    sm.qqplot(residuals, line='s')
    plt.title("Q-Q Plot of Residuals")
    plt.show()

    # 3. Homoscedasticity: Predicted vs Residuals
    plt.figure(figsize=(6,4))
    sns.scatterplot(x=y_pred, y=residuals)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel("Predicted")
    plt.ylabel("Residuals")
    plt.title("Predicted vs Residuals (Homoscedasticity Check)")
    plt.show()

    # 4. Independence of Errors: Durbin-Watson test
    dw = durbin_watson(residuals)
    print(f"Durbin-Watson Statistic (Independence of Errors): {dw:.3f} (Ideal ~2)")

    # 5. Multicollinearity: VIF
    if isinstance(X, pd.DataFrame):
        X_with_const = sm.add_constant(X)
        vif_data = pd.DataFrame()
        vif_data["feature"] = X_with_const.columns
        vif_data["VIF"] = [variance_inflation_factor(X_with_const.values, i)
                           for i in range(X_with_const.shape[1])]
        print("\nVariance Inflation Factor (VIF):")
        print(vif_data)
    else:
        print("\nVIF check skipped (X is not a DataFrame)")


def evaluate_model(y_true, y_pred):
    """
    Prints common regression evaluation metrics.
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)

    print("Model Evaluation Results")
    print("="*30)
    print(f"MAE   : {mae:.2f}")
    print(f"MSE   : {mse:.2f}")
    print(f"RMSE  : {rmse:.2f}")
    print(f"MAPE  : {mape:.2f}%")
    print(f"R2    : {r2*100:.2f}%")
    
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "MAPE": mape, "R2": r2}