import kagglehub
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib

def load_and_preprocess():
    path = kagglehub.dataset_download("atharvaingle/crop-recommendation-dataset")
    csv_path = os.path.join(path, "Crop_recommendation.csv")

    df = pd.read_csv(csv_path)
    df.dropna(inplace=True)

    le = LabelEncoder()
    df["label_encoded"] = le.fit_transform(df["label"])
    joblib.dump(le, "models/label_encoder.pkl")

    features = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    X = df[features]
    y_cls = df["label_encoded"]
    y_reg = df["rainfall"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, "models/scaler.pkl")

    X_train, X_test, y_cls_train, y_cls_test = train_test_split(
        X_scaled, y_cls, test_size=0.2, random_state=42
    )
    _, _, y_reg_train, y_reg_test = train_test_split(
        X_scaled, y_reg, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_cls_train, y_cls_test, y_reg_train, y_reg_test, X_scaled, df