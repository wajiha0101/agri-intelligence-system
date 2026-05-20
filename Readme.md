# Smart Agriculture Decision Support System

An AI-powered decision support system for precision agriculture, integrating three
machine learning models into a unified desktop application.

## System Architecture
agri-intelligence/
├── data/               # Dataset metadata
├── src/
│   ├── preprocessing.py   # Data loading and feature engineering
│   ├── models.py          # Model training and serialization
│   ├── gui.py             # Tkinter GUI application
│   └── utils.py           # Utility functions
├── models/             # Serialized model artifacts (.pkl)
├── results/            # Evaluation plots and screenshots
├── requirements.txt
├── LICENSE
└── README.md
## Models Used

| Model | Task | Metric |
|---|---|---|
| Decision Tree | Crop recommendation | Accuracy, Precision, Recall |
| KMeans Clustering | Soil zone segmentation | Silhouette Score |
| Linear Regression | Yield estimation | RMSE, MAE, R2 |

## Installation

```bash
pip install -r requirements.txt
```

## How to Run

Step 1 - Train and save all models:
```bash
python src/models.py
```

Step 2 - Launch the GUI application:
```bash
python src/gui.py
```

## Usage

Enter the soil and climate parameters in the input fields:
- N, P, K (soil nutrient levels)
- Temperature, Humidity, pH, Rainfall

Click **Run Analysis** to get:
- Recommended crop type
- Soil zone classification with agronomic guidance
- Estimated yield value

Click **Show Plots** to view:
- Feature importance chart (Decision Tree)
- Soil cluster scatter plot (KMeans)
- Residual analysis plot (Linear Regression)

## Dataset

Crop Recommendation Dataset by Atharva Ingle - sourced from Kaggle via kagglehub.

Features: N, P, K, temperature, humidity, ph, rainfall
Target: crop label (22 crop types)

## Performance Summary

| Model | Metric | Value |
|---|---|---|
| Decision Tree | Accuracy | ~0.98 |
| KMeans | Silhouette Score | ~0.27 |
| Linear Regression | R2 Score | ~0.12 |

## Future Work

1. **IoT Sensor Integration** - Replace manual input with real-time sensor data
streams from field-deployed IoT devices for continuous automated monitoring.

2. **Ensemble Deep Learning** - Replace the Decision Tree with a Random Forest
or XGBoost ensemble, and explore LSTM-based yield prediction using time-series
weather data for improved accuracy.

## License

MIT License