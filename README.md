# ⚽ FIFA 23 Players Project

## 📝 Introduction
Football, often referred to as the "beautiful game," thrives on talent, strategy, and data-driven decisions. This project leverages **data science** and **machine learning** to analyze FIFA 23 player data, providing insights into player attributes, market values, and performance dynamics.

We use two datasets of 4,000 players each:
- 📋 **FIFA Players' Names**: Full Name, Overall, Potential, Value (in Euro), Age
- 📊 **FIFA Players' Details**: Height (cm), Weight (kg), Pace Total, Shooting Total, Dribbling Total

These datasets are merged for comprehensive analysis.

---

## 🛠 Libraries and Frameworks
- 🐼 **Pandas**: Data manipulation
- 📈 **Matplotlib & Seaborn**: Data visualization
- 🧠 **Scikit-learn**: Metrics, train-test split
- 🚀 **XGBoost**: Machine learning regression model

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
