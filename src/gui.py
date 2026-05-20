import sys
import os
sys.path.append(os.path.dirname(__file__))

import tkinter as tk
from tkinter import ttk, messagebox
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import kagglehub

# Load models
dt = joblib.load("models/decision_tree.pkl")
kmeans = joblib.load("models/knn_cluster.pkl")
lr = joblib.load("models/linear_regression.pkl")
scaler = joblib.load("models/scaler.pkl")
le = joblib.load("models/label_encoder.pkl")

# Load dataset path
path = kagglehub.dataset_download("atharvaingle/crop-recommendation-dataset")
csv_path = os.path.join(path, "Crop_recommendation.csv")

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

ZONE_DESC = {
    0: "Low Nutrient Zone",
    1: "High Moisture Zone",
    2: "Balanced Zone",
    3: "Dry Acidic Zone"
}

def predict():
    try:
        vals = [float(entries[f].get()) for f in FEATURES]
        x = np.array(vals).reshape(1, -1)
        x_scaled = scaler.transform(x)

        crop_idx = dt.predict(x_scaled)[0]
        crop = le.inverse_transform([crop_idx])[0]

        cluster = kmeans.predict(x_scaled)[0]
        zone = ZONE_DESC.get(cluster, "Unknown Zone")

        yield_pred = lr.predict(x_scaled)[0]

        result_var.set(
            f"Recommended Crop: {crop}     |     "
            f"Soil Zone: {cluster} - {zone}     |     "
            f"Yield Estimate: {yield_pred:.2f} mm"
        )
    except Exception as e:
        messagebox.showerror("Error", str(e))

def show_plots():
    df = pd.read_csv(csv_path)
    X = scaler.transform(df[FEATURES])
    clusters = kmeans.predict(X)
    y_reg = df["rainfall"].values
    y_pred = lr.predict(X)
    residuals = y_reg - y_pred

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Plot 1 - Feature Importance
    axes[0].barh(FEATURES, dt.feature_importances_, color="steelblue")
    axes[0].set_title("Feature Importance (Decision Tree)")
    axes[0].set_xlabel("Importance")

    # Plot 2 - Cluster Scatter
    scatter = axes[1].scatter(X[:, 0], X[:, 1], c=clusters, cmap="tab10", s=5)
    axes[1].set_title("Soil Clusters (N vs P)")
    axes[1].set_xlabel("N")
    axes[1].set_ylabel("P")
    plt.colorbar(scatter, ax=axes[1])

    # Plot 3 - Residual Plot
    axes[2].scatter(y_pred, residuals, s=5, alpha=0.5, color="coral")
    axes[2].axhline(0, color="red", linewidth=1)
    axes[2].set_title("Residual Plot (Linear Regression)")
    axes[2].set_xlabel("Predicted")
    axes[2].set_ylabel("Residual")

    plt.tight_layout()

    for widget in plot_frame.winfo_children():
        widget.destroy()

    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# --- Build GUI ---
root = tk.Tk()
root.title("Smart Agriculture Decision Support System")
root.geometry("1000x750")

# Input Frame
frame_inputs = ttk.LabelFrame(root, text="Enter Soil & Climate Parameters")
frame_inputs.pack(fill="x", padx=10, pady=10)

entries = {}
for i, f in enumerate(FEATURES):
    ttk.Label(frame_inputs, text=f).grid(row=0, column=i, padx=8, pady=5)
    e = ttk.Entry(frame_inputs, width=10)
    e.grid(row=1, column=i, padx=8, pady=5)
    entries[f] = e

# Buttons
btn_frame = ttk.Frame(root)
btn_frame.pack(pady=5)
ttk.Button(btn_frame, text="Run Analysis", command=predict).grid(row=0, column=0, padx=10)
ttk.Button(btn_frame, text="Show Plots", command=show_plots).grid(row=0, column=1, padx=10)

# Result Label
result_var = tk.StringVar()
ttk.Label(root, textvariable=result_var, font=("Arial", 10, "bold"), wraplength=950).pack(pady=5)

# Plot Frame
plot_frame = ttk.Frame(root)
plot_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

root.mainloop()