import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from sklearn.model_selection import train_test_split
import pandas as pd

def line_plot(data, title, xlabel, ylabel):
    plt.figure(figsize=(10, 6))
    ax = data.plot(kind='line', color='blue', marker='o')
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    model_path = os.path.join("models", "forecasting_model.pkl")
    model_data = joblib.load(model_path)
    model = model_data['model']
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), "data", "processed", "train_features.parquet")
    df = pd.read_parquet(data_path)
    feature_names = model.feature_name_
    # Make sure 'date' is included in the features
    if 'date' not in feature_names:
        feature_names = ['date'] + [f for f in feature_names if f != 'date']
    
    X = df[feature_names].copy()
    y = df['sale_amount']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Keep date column for plotting
    X_test_dates = X_test[['date']].copy()
    X_test = X_test[feature_names].drop(columns=['date'])
    X_train = X_train[feature_names].drop(columns=['date'])
    predictions = model.predict(X_test)
    predicted_sales_df = pd.DataFrame({
        'date': X_test_dates.values.flatten(),
        'predicted_sale_amount': predictions
    })
    predicted_sales_df['date'] = pd.to_datetime(predicted_sales_df['date'])
    predicted_sales_df = predicted_sales_df.set_index('date').sort_index()
    predicted_sales_by_week = predicted_sales_df['predicted_sale_amount'].resample('W').mean()
    line_plot(predicted_sales_by_week, 'Predicted Sales Trend by Week', 'Week', 'Average Predicted Sale Amount')
    output_dir = os.path.join("outputs")
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "sales_forecast.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Plot saved to {os.path.abspath(plot_path)}")