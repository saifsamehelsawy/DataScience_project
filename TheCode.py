# Libraries
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score 

# Read FIFA 23 Players' Names
df_1 = pd.read_csv("FIFA Players' Names.csv")
df_1 = df_1[:4000]
df_1

# Read FIFA 23 Players' Details
df_2 = pd.read_csv("FIFA Players' Details.csv")
df_2 = df_2[:4000]
df_2

# Merge FIFA 23 Data
merged_data = pd.concat([df_1, df_2])
merged_data.to_csv('FIFA_merged.csv', index=False)

# Check the merge
print(len(merged_data))
print(merged_data.columns)

merged_data = pd.read_csv("FIFA_merged.csv")
merged_data = merged_data[:4000]
merged_data

# EDA
merged_data.info()
merged_data.describe()

# Check for Cleaning
clean_data = pd.read_csv("FIFA_merged.csv")
clean_data = clean_data[:4000]
clean_data

# Handle missing values and duplicates
clean_data.fillna(0, inplace=True)
clean_data.dropna(inplace=True)
clean_data.drop_duplicates(inplace=True)
clean_data.isnull().sum()

# Correlation Matrix
df_num = clean_data[["Overall", "Potential", "Dribbling Total", "Shooting Total", "Pace Total", "Age", "Value(in Euro)"]]
corr = df_num.corr()
plt.figure(figsize=(10,10))
sn.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.show()

# Box Plot
df = pd.read_csv("FIFA_merged.csv")
plt.figure(figsize=(10, 6))
df.boxplot(column=['Overall'])
plt.title('Box Plot: Overall')
plt.ylabel('Overall Rating')
plt.grid(True)
plt.show()

# Scatter Plot
plt.figure(figsize=(10, 6))
plt.scatter(df['Overall'], df['Potential'], alpha=0.5)
plt.title('Scatter Plot: Overall vs. Potential')
plt.xlabel('Overall')
plt.ylabel('Potential')
plt.grid(True)
plt.show()

# Histogram
columns = ['Shooting Total', 'Pace Total']
for column in columns:
    plt.figure()
    df[column].hist(bins=20)
    plt.title(f'Histogram of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.grid(False)
    plt.show()

# Top 10 Players by Overall & Value(in Euro)
highest_paid_players = df.nlargest(10, 'Value(in Euro)')
plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
plt.barh(highest_paid_players['Full Name'], highest_paid_players['Overall'], color='skyblue')
plt.xlabel('Overall')
plt.ylabel('Player Name')
plt.title('Highest Overall Players by Value')
plt.subplot(1, 2, 2)
plt.barh(highest_paid_players['Full Name'], highest_paid_players['Value(in Euro)'], color='skyblue')
plt.xlabel('Value(in Euro) Rating')
plt.ylabel('Player Name')
plt.title('Highest Paid Players by Value Rating')
plt.tight_layout()
plt.show()

# Pie Chart for 10 Players
selected_players = df.head(10)
print(selected_players[['Full Name', 'Value(in Euro)']])
plt.figure(figsize=(10, 10))
plt.pie(selected_players['Value(in Euro)'], labels=selected_players['Full Name'], autopct='%1.1f%%', startangle=140)
plt.axis('equal')  
plt.title('Value(in Euro) Distribution of Selected Players', fontsize=20)
plt.show()

# Curve Plot of Pace Total For Top 10 Players
data = pd.read_csv('FIFA_merged.csv')
top_10_players = data.sort_values(by='Pace Total', ascending=False).head(10)
print(top_10_players[['Full Name', 'Pace Total']])
plt.figure(figsize=(12, 8))
plt.plot(top_10_players['Full Name'], top_10_players['Pace Total'], marker='o', linestyle='-', color='b')
plt.title('Top 10 Players by Pace Total')
plt.xlabel('Full Name')
plt.ylabel('Pace Total')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

# Feature selection, Machine Learning algorithm and Evaluation
df = pd.read_csv("FIFA_merged.csv")
features = ['Potential', 'Value(in Euro)', 'Age', 'Height(in cm)', 'Weight(in kg)', 'Pace Total', 'Shooting Total', 'Dribbling Total']
target = 'Overall'
X = df[features]  # Features
y = df[target]    # Target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# XGBoost model
xgb_model = XGBRegressor()
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test) 

# Evaluation
r2_xgb = r2_score(y_test, y_pred_xgb)
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
print("XGBoost R^2 =", r2_xgb)
print("XGBoost MSE =", mse_xgb)

# Residual plot
plt.figure(figsize=(6, 4))
plt.scatter(y_test, y_pred_xgb, color='purple', alpha=0.6, label='Predicted vs Actual')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', color='red', lw=2, label='Ideal Line')
plt.title('XGBoost: Prediction vs Actual')
plt.xlabel('Actual Overall Rating')
plt.ylabel('Predicted Overall Rating')
plt.legend()
plt.show()
