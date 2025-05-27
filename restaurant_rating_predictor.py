import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score



df = pd.read_csv('Dataset .csv')
df.head()
df.columns = df.columns.str.strip()

print("Dataset Shape:", df.shape)
print("\nDataset Info:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum())

print("\nStatistical Summary:")
print(df.describe())


columns_to_drop = ['Restaurant ID', 'Restaurant Name', 'Address',
                   'Locality', 'Locality Verbose', 'Rating color', 'Rating text']
df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True)

if 'Cuisines' in df.columns:
    df['Cuisines'] = df['Cuisines'].fillna('Unknown')

label_encoders = {}
categorical_columns = ['City', 'Cuisines', 'Currency',
                       'Has Table booking', 'Has Online delivery',
                       'Is delivering now', 'Switch to order menu']
for col in categorical_columns:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

scaler = StandardScaler()
numerical_columns = ['Longitude', 'Latitude', 'Average Cost for two', 'Price range', 'Votes']
for col in numerical_columns:
    if col in df.columns:
        df[col] = scaler.fit_transform(df[[col]])

if 'Aggregate rating' in df.columns:
    X = df.drop('Aggregate rating', axis=1)
    y = df['Aggregate rating']
else:
    raise KeyError("Column 'Aggregate rating' not found in dataset.")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)


plt.figure(figsize=(8, 6))
sns.histplot(df['Aggregate rating'], bins=20, kde=True, color='#1f77b4', edgecolor='#333333')
plt.title('Distribution of Aggregate Rating', fontsize=16, fontweight='bold', color='#444444')
plt.xlabel('Aggregate Rating', fontsize=14, color='#666666')
plt.ylabel('Frequency', fontsize=14, color='#666666')
plt.grid(True, linestyle='--', alpha=0.5, color='#888888')
plt.gca().set_facecolor('#f7f7f7')
plt.show()


plt.figure(figsize=(12, 10))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='viridis', linewidths=1,
            annot_kws={'size': 12, 'weight': 'bold', 'color': 'white'})
plt.title('Correlation Heatmap', fontsize=16, fontweight='bold', color='#222222')
plt.xticks(fontsize=12, rotation=45, ha='right', color='#555555')
plt.yticks(fontsize=12, rotation=0, color='#555555')
plt.gca().set_facecolor('#f0f0f0')
plt.show()


models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(f"\n{name} Performance:")
    print("Mean Absolute Error:", mean_absolute_error(y_test, predictions))
    print("Mean Squared Error:", mean_squared_error(y_test, predictions))
    print("Root Mean Squared Error:", np.sqrt(mean_squared_error(y_test, predictions)))
    print("R-squared Score:", r2_score(y_test, predictions))


rf_model = models['Random Forest']
importances = rf_model.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df.sort_values(by='Importance', ascending=False, inplace=True)

plt.figure(figsize=(10, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Feature Importances from Random Forest')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()