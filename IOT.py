import pandas as pd
import numpy as np
import io
import os
import tkinter as tk
from tkinter import ttk, messagebox
from cryptography.fernet import Fernet
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ENCRYPTED_FILE = "Gym_encrypted.bin"
PLAIN_FILE = "Gym.csv.csv" 
df = None 

def decrypt_dataset(key):
    try:
        cipher = Fernet(key.encode())
        with open(ENCRYPTED_FILE, "rb") as f:
            encrypted = f.read()
        decrypted_data = cipher.decrypt(encrypted)
        csv_text = decrypted_data.decode("utf-8")
        df = pd.read_csv(io.StringIO(csv_text))
        return df
    except Exception as e:
        messagebox.showerror("Decryption Failed", f"❌ Error: {str(e)}")
        return None

def on_submit():
    key = key_entry.get()
    if not key:
        messagebox.showwarning("Missing Key", "Please enter your encryption key.")
        return
    root.destroy()
    global df
    df = decrypt_dataset(key)
    if df is None:
        pass


root = tk.Tk()
root.title("🔐 Enter Encryption Key")
root.geometry("400x180")
root.configure(bg="#f2f2f2")
root.resizable(False, False)

style = ttk.Style()
style.theme_use("clam")
style.configure("TLabel", background="#f2f2f2", font=("Helvetica", 12))
style.configure("TEntry", font=("Helvetica", 12))
style.configure("TButton", font=("Helvetica", 12), padding=6)

frame = ttk.Frame(root, padding=20)
frame.pack(expand=True)

ttk.Label(frame, text="🔑 Enter your encryption key:").pack(pady=(0, 10))
key_entry = ttk.Entry(frame, width=40, show="*")
key_entry.pack(pady=(0, 10))
ttk.Button(frame, text="Unlock Dataset", command=on_submit).pack(pady=(10, 0))

root.mainloop()


if df is None:
    exit()

print("🔓 Dataset decrypted and loaded into pandas")
print("🔹 Dataset Shape:", df.shape)
print("🔹 Columns:", df.columns.tolist())
print(df.head())

label_encoders = {}
for col in ['Gender', 'Workout_Type']:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str)) 
        label_encoders[col] = le

X = df.drop(columns=['Calories_Burned'])
y = df['Calories_Burned']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


lin_reg = LinearRegression()
lin_reg.fit(X_train_scaled, y_train)
y_pred_lin = lin_reg.predict(X_test_scaled)

knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
y_pred_knn = knn.predict(X_test_scaled)


def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"\n📊 {model_name} Performance:")
    print(f"  Mean Absolute Error (MAE): {mae:.2f}")
    print(f"  Mean Squared Error (MSE): {mse:.2f}")
    print(f"  R² Score: {r2:.4f}")
    return pd.Series({'MAE': mae, 'MSE': mse, 'R2': r2})

results = pd.DataFrame({
    'Linear Regression': evaluate_model(y_test, y_pred_lin, 'Linear Regression'),
    'KNN Regressor': evaluate_model(y_test, y_pred_knn, 'KNN Regressor')
})

def plot_model_comparison(results_df):
    """Creates a bar chart to compare R2 scores."""
    plot_df = results_df.T[['R2']].sort_values(by='R2', ascending=False).reset_index()
    plot_df.columns = ['Model', 'R2 Score']
    
    plt.figure(figsize=(8, 5))
    sns.barplot(x='Model', y='R2 Score', data=plot_df, palette='viridis')
    plt.title('R² Score Comparison: Linear Regression vs. KNN')
    plt.ylim(0, 1.05)
    plt.ylabel('R² Score (Closer to 1 is Better)')
    plt.xlabel('Regression Model')
    for index, row in plot_df.iterrows():
        plt.text(row.name, row['R2 Score'] + 0.02, f"{row['R2 Score']:.4f}", color='black', ha="center")
    plt.savefig("model_r2_comparison.png")
    plt.close()
    print("📁 Saved R² comparison plot as 'model_r2_comparison.png'")


def plot_actual_vs_predicted(y_true, y_pred, model_name, filename):
    """Creates a scatter plot to show how close predictions are to actual values."""
    
    plt.figure(figsize=(7, 7))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.6)
    
    max_val = max(y_true.max(), y_pred.max())
    min_val = min(y_true.min(), y_pred.min())
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Perfect Prediction (y=x)')
    
    plt.title(f'Actual vs. Predicted Calories Burned ({model_name})')
    plt.xlabel('Actual Calories Burned')
    plt.ylabel('Predicted Calories Burned')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.savefig(filename)
    plt.close()
    print(f"📁 Saved Actual vs. Predicted plot for {model_name} as '{filename}'")

print("\n[Generating Model Visualizations...]")

# 1. Plot Linear Regression (The best model)
plot_actual_vs_predicted(y_test, y_pred_lin, 'Linear Regression', 'linear_regression_prediction.png')
# 2. Plot KNN Regressor (The weaker model)
plot_actual_vs_predicted(y_test, y_pred_knn, 'KNN Regressor', 'knn_regression_prediction.png')
# 3. Plot the final comparison
plot_model_comparison(results)


print("\n✅ Model Comparison:")
print(results)

results.to_csv("model_comparison_results.csv", index=False)
print("\n📁 Results saved as 'model_comparison_results.csv'")