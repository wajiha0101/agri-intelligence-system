import sys
import os
sys.path.append(os.path.dirname(__file__))

import tkinter as tk
from tkinter import ttk, messagebox
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import kagglehub

# Load models
dt     = joblib.load("models/decision_tree.pkl")
kmeans = joblib.load("models/knn_cluster.pkl")
lr     = joblib.load("models/linear_regression.pkl")
scaler = joblib.load("models/scaler.pkl")
le     = joblib.load("models/label_encoder.pkl")

path     = kagglehub.dataset_download("atharvaingle/crop-recommendation-dataset")
csv_path = os.path.join(path, "Crop_recommendation.csv")

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
LABELS   = [
    "N  (Nitrogen)",
    "P  (Phosphorus)",
    "K  (Potassium)",
    "Temperature (°C)",
    "Humidity (%)",
    "pH  (Acidity)",
    "Rainfall (mm)",
]

ZONE_DESC = {
    0: "Low Nutrient Zone",
    1: "High Moisture Zone",
    2: "Balanced Zone",
    3: "Dry Acidic Zone",
}

# ── Colours ──────────────────────────────────────────────────────────────────
BG        = "#F7F9FC"
CARD      = "#FFFFFF"
PRIMARY   = "#3A7D44"   # forest green
ACCENT    = "#56A96A"
TEXT_DARK = "#1E2D2F"
TEXT_MID  = "#4A5568"
TEXT_LITE = "#718096"
BORDER    = "#E2E8F0"
SUCCESS   = "#2F855A"
WARN      = "#C05621"

# ── Root ─────────────────────────────────────────────────────────────────────
root = tk.Tk()
root.title("Smart Agriculture Decision Support System")
root.geometry("1060x780")
root.configure(bg=BG)
root.resizable(True, True)

# ── Style ─────────────────────────────────────────────────────────────────────
style = ttk.Style(root)
style.theme_use("clam")
style.configure("TFrame",       background=BG)
style.configure("Card.TFrame",  background=CARD, relief="flat")
style.configure("TLabel",       background=BG,   foreground=TEXT_DARK, font=("Segoe UI", 10))
style.configure("Card.TLabel",  background=CARD, foreground=TEXT_DARK, font=("Segoe UI", 10))
style.configure("Head.TLabel",  background=BG,   foreground=TEXT_DARK, font=("Segoe UI", 14, "bold"))
style.configure("Sub.TLabel",   background=BG,   foreground=TEXT_MID,  font=("Segoe UI", 9))
style.configure("Tag.TLabel",   background=BG,   foreground=TEXT_LITE, font=("Segoe UI", 8))
style.configure("Result.TLabel",background=BG,   foreground=SUCCESS,   font=("Segoe UI", 11, "bold"))

style.configure("Primary.TButton",
                background=PRIMARY, foreground="white",
                font=("Segoe UI", 10, "bold"),
                padding=(18, 8), relief="flat", borderwidth=0)
style.map("Primary.TButton",
          background=[("active", ACCENT), ("pressed", SUCCESS)])

style.configure("Ghost.TButton",
                background=CARD, foreground=PRIMARY,
                font=("Segoe UI", 10),
                padding=(18, 8), relief="flat", borderwidth=1)
style.map("Ghost.TButton",
          background=[("active", "#EDF7F0")])

style.configure("TEntry",
                fieldbackground=CARD, foreground=TEXT_DARK,
                font=("Segoe UI", 10), padding=6, relief="flat")

# ── Helper: rounded card ──────────────────────────────────────────────────────
def card(parent, **kw):
    f = tk.Frame(parent, bg=CARD, bd=0,
                 highlightbackground=BORDER, highlightthickness=1, **kw)
    return f

# ═══════════════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════════════
header = tk.Frame(root, bg=PRIMARY, pady=14)
header.pack(fill="x")

tk.Label(header, text="🌾  Smart Agriculture Decision Support System",
         bg=PRIMARY, fg="white",
         font=("Segoe UI", 15, "bold")).pack(side="left", padx=20)
tk.Label(header, text="AI-Powered Crop Intelligence",
         bg=PRIMARY, fg="#C6F6D5",
         font=("Segoe UI", 9)).pack(side="right", padx=20)

# ═══════════════════════════════════════════════════════════════════════════════
# INPUT CARD
# ═══════════════════════════════════════════════════════════════════════════════
input_card = card(root, pady=16, padx=20)
input_card.pack(fill="x", padx=18, pady=(14, 0))

tk.Label(input_card, text="Soil & Climate Parameters",
         bg=CARD, fg=TEXT_DARK,
         font=("Segoe UI", 11, "bold")).grid(row=0, column=0, columnspan=7,
                                              sticky="w", pady=(0, 10))

entries = {}
for i, (feat, label) in enumerate(zip(FEATURES, LABELS)):
    col_frame = tk.Frame(input_card, bg=CARD)
    col_frame.grid(row=1, column=i, padx=6)

    tk.Label(col_frame, text=label, bg=CARD, fg=TEXT_MID,
             font=("Segoe UI", 8, "bold"), wraplength=110,
             justify="center").pack(anchor="w")

    e = tk.Entry(col_frame, width=11, relief="solid", bd=1,
                 font=("Segoe UI", 10), fg=TEXT_DARK, bg="#F0FFF4",
                 highlightcolor=PRIMARY, highlightthickness=1,
                 insertbackground=PRIMARY)
    e.pack(pady=(3, 0), ipady=5)
    entries[feat] = e

# ═══════════════════════════════════════════════════════════════════════════════
# BUTTONS
# ═══════════════════════════════════════════════════════════════════════════════
btn_frame = tk.Frame(root, bg=BG)
btn_frame.pack(pady=12)

def make_btn(parent, text, cmd, primary=True):
    bg_c = PRIMARY if primary else CARD
    fg_c = "white"  if primary else PRIMARY
    b = tk.Button(parent, text=text, command=cmd,
                  bg=bg_c, fg=fg_c, activebackground=ACCENT,
                  activeforeground="white",
                  font=("Segoe UI", 10, "bold"),
                  relief="flat", bd=0, padx=22, pady=9,
                  cursor="hand2")
    b.pack(side="left", padx=6)
    return b

make_btn(btn_frame, "▶  Run Analysis", lambda: predict())
make_btn(btn_frame, "📊  Show Plots",  lambda: show_plots(), primary=False)
make_btn(btn_frame, "✕  Clear",        lambda: clear_inputs(), primary=False)

# ═══════════════════════════════════════════════════════════════════════════════
# RESULT CARDS
# ═══════════════════════════════════════════════════════════════════════════════
result_row = tk.Frame(root, bg=BG)
result_row.pack(fill="x", padx=18, pady=(0, 10))

def result_card(parent, icon, title, var, col):
    f = card(parent, padx=14, pady=12)
    f.grid(row=0, column=col, sticky="nsew", padx=5)
    parent.columnconfigure(col, weight=1)
    tk.Label(f, text=icon, bg=CARD, font=("Segoe UI", 20)).pack(anchor="w")
    tk.Label(f, text=title, bg=CARD, fg=TEXT_LITE,
             font=("Segoe UI", 8, "bold")).pack(anchor="w")
    tk.Label(f, textvariable=var, bg=CARD, fg=SUCCESS,
             font=("Segoe UI", 12, "bold"), wraplength=260).pack(anchor="w", pady=(4, 0))
    return f

crop_var  = tk.StringVar(value="-")
zone_var  = tk.StringVar(value="-")
yield_var = tk.StringVar(value="-")

result_card(result_row, "🌱", "RECOMMENDED CROP",  crop_var,  0)
result_card(result_row, "🗺️",  "SOIL ZONE",          zone_var,  1)
result_card(result_row, "📈", "YIELD ESTIMATE",     yield_var, 2)

# ═══════════════════════════════════════════════════════════════════════════════
# PLOT AREA
# ═══════════════════════════════════════════════════════════════════════════════
plot_outer = card(root, padx=10, pady=10)
plot_outer.pack(fill="both", expand=True, padx=18, pady=(0, 14))

plot_placeholder = tk.Label(plot_outer,
    text="📊  Click  'Show Plots'  to view Feature Importance · Soil Clusters · Residual Analysis",
    bg=CARD, fg=TEXT_LITE, font=("Segoe UI", 10))
plot_placeholder.pack(expand=True)

# ═══════════════════════════════════════════════════════════════════════════════
# LOGIC
# ═══════════════════════════════════════════════════════════════════════════════
def predict():
    try:
        vals    = [float(entries[f].get()) for f in FEATURES]
        x       = np.array(vals).reshape(1, -1)
        x_sc    = scaler.transform(x)

        crop    = le.inverse_transform(dt.predict(x_sc))[0].capitalize()
        cluster = kmeans.predict(x_sc)[0]
        zone    = ZONE_DESC.get(cluster, "Unknown Zone")
        yld     = lr.predict(x_sc)[0]

        crop_var .set(crop)
        zone_var .set(f"Zone {cluster}  -  {zone}")
        yield_var.set(f"{yld:.2f} mm")
    except ValueError:
        messagebox.showwarning("Missing Input", "Please fill in all parameter fields.")
    except Exception as e:
        messagebox.showerror("Error", str(e))

def clear_inputs():
    for e in entries.values():
        e.delete(0, tk.END)
    crop_var .set("-")
    zone_var .set("-")
    yield_var.set("-")

def show_plots():
    df      = pd.read_csv(csv_path)
    X       = scaler.transform(df[FEATURES])
    clusters= kmeans.predict(X)
    y_reg   = df["rainfall"].values
    y_pred  = lr.predict(X)
    residuals = y_reg - y_pred

    plt.rcParams.update({
        "font.family": "Segoe UI",
        "axes.facecolor": "#F7F9FC",
        "figure.facecolor": CARD,
        "axes.edgecolor": BORDER,
        "axes.labelcolor": TEXT_MID,
        "xtick.color": TEXT_LITE,
        "ytick.color": TEXT_LITE,
        "grid.color": BORDER,
        "grid.linestyle": "--",
        "grid.alpha": 0.7,
    })

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    fig.patch.set_facecolor(CARD)

    # Plot 1 - Feature Importance
    colors = [PRIMARY if v == max(dt.feature_importances_) else ACCENT
              for v in dt.feature_importances_]
    axes[0].barh(LABELS, dt.feature_importances_, color=colors, edgecolor="none", height=0.6)
    axes[0].set_title("Feature Importance", fontsize=11, fontweight="bold", color=TEXT_DARK, pad=10)
    axes[0].set_xlabel("Importance Score", fontsize=9)
    axes[0].invert_yaxis()
    axes[0].grid(axis="x")

    # Plot 2 - Cluster Scatter
    sc = axes[1].scatter(X[:, 0], X[:, 1], c=clusters,
                         cmap="Set2", s=8, alpha=0.7, edgecolors="none")
    axes[1].set_title("Soil Clusters  (N vs P)", fontsize=11, fontweight="bold", color=TEXT_DARK, pad=10)
    axes[1].set_xlabel("Nitrogen (N)", fontsize=9)
    axes[1].set_ylabel("Phosphorus (P)", fontsize=9)
    plt.colorbar(sc, ax=axes[1], label="Cluster")

    # Plot 3 - Residual
    axes[2].scatter(y_pred, residuals, s=6, alpha=0.5, color=ACCENT, edgecolors="none")
    axes[2].axhline(0, color=PRIMARY, linewidth=1.5, linestyle="--")
    axes[2].set_title("Residual Plot  (Linear Regression)", fontsize=11,
                      fontweight="bold", color=TEXT_DARK, pad=10)
    axes[2].set_xlabel("Predicted Value", fontsize=9)
    axes[2].set_ylabel("Residual", fontsize=9)
    axes[2].grid(True)

    plt.tight_layout(pad=2)

    # Clear placeholder and embed
    for w in plot_outer.winfo_children():
        w.destroy()

    canvas = FigureCanvasTkAgg(fig, master=plot_outer)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)

# ═══════════════════════════════════════════════════════════════════════════════
root.mainloop()