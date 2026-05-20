import sys
import os
sys.path.append(os.path.dirname(__file__))

import numpy as np
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             silhouette_score, mean_squared_error,
                             mean_absolute_error, r2_score)
from preprocessing import load_and_preprocess

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

def train_all():
    X_train, X_test, y_cls_train, y_cls_test, y_reg_train, y_reg_test, X_scaled, df = load_and_preprocess()

    # --- Decision Tree ---
    dt = DecisionTreeClassifier(max_depth=10, random_state=42)
    dt.fit(X_train, y_cls_train)
    y_pred_cls = dt.predict(X_test)
    print("=== Decision Tree ===")
    print("Accuracy :", accuracy_score(y_cls_test, y_pred_cls))
    print("Precision:", precision_score(y_cls_test, y_pred_cls, average="weighted"))
    print("Recall   :", recall_score(y_cls_test, y_pred_cls, average="weighted"))
    joblib.dump(dt, "models/decision_tree.pkl")

    # --- KMeans Clustering ---
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, cluster_labels)
    print("\n=== KMeans Clustering ===")
    print("Silhouette Score:", score)
    joblib.dump(kmeans, "models/knn_cluster.pkl")

    # --- Linear Regression ---
    lr = LinearRegression()
    lr.fit(X_train, y_reg_train)
    y_pred_reg = lr.predict(X_test)
    print("\n=== Linear Regression ===")
    print("RMSE:", np.sqrt(mean_squared_error(y_reg_test, y_pred_reg)))
    print("MAE :", mean_absolute_error(y_reg_test, y_pred_reg))
    print("R2  :", r2_score(y_reg_test, y_pred_reg))
    joblib.dump(lr, "models/linear_regression.pkl")

    print("\nAll models saved to models/")

if __name__ == "__main__":
    train_all()