import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
df = pd.read_csv("diabetes_synthetic_realistic.csv")
df.head()
df.info()
df.describe()
df.isnull().sum()


#convert the target variable from yes/no to 1/0

print(df["Diabetes_Status"].unique())

# Convert target to binary
df["Diabetes_Status"] = df["Diabetes_Status"].map({"Yes":1, "No":0})

# Verify
print(df["Diabetes_Status"].value_counts())
print(df["Diabetes_Status"].dtype)

# Handling missing alcohol consumption values by keeping it unknown
# Check how many missing values
print("Missing Alcohol_Intake:", df["Alcohol_Intake"].isnull().sum())

# Replace missing with 'Unknown'
df["Alcohol_Intake"] = df["Alcohol_Intake"].fillna("Unknown")

# Verify replacement
print(df["Alcohol_Intake"].value_counts())
print("Remaining missing:", df["Alcohol_Intake"].isnull().sum())

leakage_cols = [
    "HBA1C",
    "Fasting_Blood_Sugar",
    "Postprandial_Blood_Sugar",
    "Glucose_Tolerance_Test_Result"
]

# Separate features and target
X = df.drop("Diabetes_Status", axis=1)
y = df["Diabetes_Status"]

# --- MEANINGFUL FEATURE ENGINEERING ---
# 1. BMI Categories (Standard medical ranges)
# Underweight: <18.5, Normal: 18.5-24.9, Overweight: 25-29.9, Obese: >=30
X["BMI_Category"] = pd.cut(
    X["BMI"], 
    bins=[0, 18.5, 24.9, 29.9, 100], 
    labels=["Underweight", "Normal", "Overweight", "Obese"]
)

# 2. Age Groups
# Young: <30, Adult: 30-50, Senior: >50
X["Age_Group"] = pd.cut(
    X["Age"], 
    bins=[0, 30, 50, 120], 
    labels=["Young", "Adult", "Senior"]
)

# 3. High Risk Score (Interaction Term)
# Simple additive score: +1 for High BMI (>25), +1 for Senior (>50), +1 for Smoker, +1 for Family History
# Note: We need to handle categorical values for Smoking and Family History first if we want to include them here.
# For now, let's just do numerical interactions.
X["BMI_Age_Interaction"] = X["BMI"] * X["Age"]

# 4. Risk Factor Count (Aggregate "Lifestyle Load")
# Count how many of these bad habits/conditions are present
X["Risk_Factor_Count"] = (
    (X["Hypertension"] == "Yes").astype(int) +
    (X["Family_History"] == "Yes").astype(int) +
    (X["Smoking_Status"] == "Smoker").astype(int) +
    (X["Stress_Level"] == "High").astype(int) +
    (X["Cholesterol_Level"] == "High").astype(int) +
    (X["Physical_Activity"] == "Low").astype(int) +
    (X["Sleep_Quality"] == "Poor").astype(int) +
    (X["Age"] > 50).astype(int) + 
    (X["BMI"] > 30).astype(int)
)

# 5. Metabolic Syndrome (Synergistic Effect)
# Obesity + Hypertension + High Cholesterol
X["Metabolic_Syndrome_Score"] = (
    (X["BMI"] > 30).astype(int) + 
    (X["Hypertension"] == "Yes").astype(int) + 
    (X["Cholesterol_Level"] == "High").astype(int)
)

# 6. High Risk Lifestyle
# Smoking + Alcohol + Sedentary
X["High_Risk_Lifestyle"] = (
    (X["Smoking_Status"] == "Smoker").astype(int) + 
    (X["Alcohol_Intake"] == "High").astype(int) + 
    (X["Physical_Activity"] == "Low").astype(int)
)

print("Added advanced features: Metabolic_Syndrome_Score, High_Risk_Lifestyle")

# Check Correlation of numerical columns with Target
# We temporarily join y back to check correlation
temp_df = X.select_dtypes(include=["number"]).copy()
temp_df["Diabetes_Status"] = y
print("\nCorrelation with Target:")
print(temp_df.corr()["Diabetes_Status"].sort_values(ascending=False))
del temp_df


# Leakage columns are not present in the new synthetic dataset, so we don't need to drop them.
# X = X.drop(leakage_cols, axis=1)

print("Remaining features:", X.shape[1])
print(X.columns.value_counts())

# Check shapes
print("Features shape:", X.shape)
print("Target shape:", y.shape)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,      # VERY IMPORTANT for imbalance
    random_state=42
)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)
print("\nTrain distribution:\n", y_train.value_counts(normalize=True))
print("\nTest distribution:\n", y_test.value_counts(normalize=True))
# Identify categorical and numerical columns

categorical_cols = X_train.select_dtypes(include="object").columns.tolist()
numerical_cols = X_train.select_dtypes(exclude="object").columns.tolist()

print("Categorical features:", len(categorical_cols))
print(categorical_cols)

print("\nNumerical features:", len(numerical_cols))
print(numerical_cols)
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

# Identify columns again - AFTER feature engineering
categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
numerical_cols = X_train.select_dtypes(exclude=["object", "category"]).columns.tolist()

# Convert categorical columns to string type to ensure OneHotEncoder works
X_train[categorical_cols] = X_train[categorical_cols].astype(str)
X_test[categorical_cols] = X_test[categorical_cols].astype(str)

print("Updated Categorical features:", len(categorical_cols))
print(categorical_cols)
print("Updated Numerical features:", len(numerical_cols))
print(numerical_cols)

# Encoder & Scaler
encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
scaler = StandardScaler()

preprocessor = ColumnTransformer(
    transformers=[
        ("num", scaler, numerical_cols),
        ("cat", encoder, categorical_cols)
    ]
)

# Fit only on training
X_train_encoded = preprocessor.fit_transform(X_train)
X_test_encoded = preprocessor.transform(X_test)

print("Encoded train shape:", X_train_encoded.shape)
print("Encoded test shape:", X_test_encoded.shape)
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier

# Define the parameter grid
# Define the parameter grid - OPTIMIZED FOR 90% ACCURACY
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 4],
    'subsample': [0.8, 0.9],         # Stochastic Gradient Boosting (Regularization)
    'min_samples_leaf': [1, 2, 4]    # Regularization to prevent overfitting
}

gb = GradientBoostingClassifier(random_state=42)

print("\nStarting Hyperparameter Tuning (GridSearch)... this may take a minute.")
grid_search = GridSearchCV(
    estimator=gb,
    param_grid=param_grid,
    cv=3,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train_encoded, y_train)

print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best CV Accuracy: {grid_search.best_score_:.4f}")

# Use best model
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_encoded)
y_prob = best_model.predict_proba(X_test_encoded)[:, 1]
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix

print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# --- VISUALIZATIONS ---

# 1. Confusion Matrix
plt.figure(figsize=(6, 4))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('confusion_matrix.png')
print("Saved confusion_matrix.png")

# 2. ROC Curve
from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.savefig('roc_curve.png')
print("Saved roc_curve.png")

# 3. Feature Importance
# Get feature names from preprocessor
try:
    feature_names = preprocessor.get_feature_names_out()
    # Clean up names (remove "num__", "cat__")
    feature_names = [f.split("__")[-1] for f in feature_names]
    
    importances = best_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Plot Top 10 Features
    plt.figure(figsize=(10, 6))
    top_n = 10
    sns.barplot(x=importances[indices[:top_n]], y=[feature_names[i] for i in indices[:top_n]], palette="viridis")
    plt.title('Top 10 Feature Importances')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    print("Saved feature_importance.png")
except Exception as e:
    print(f"Could not plot feature importance: {e}")

# 4. Probability Distribution Plot
plt.figure(figsize=(8, 6))
sns.histplot(y_prob[y_test == 0], color='skyblue', label='No Diabetes', kde=True, stat="density", bins=20)
sns.histplot(y_prob[y_test == 1], color='red', label='Diabetes', kde=True, stat="density", bins=20, alpha=0.6)
plt.title('Prediction Probability Distribution')
plt.xlabel('Predicted Probability of Diabetes')
plt.ylabel('Density')
plt.legend()
plt.savefig('prob_dist.png')
print("Saved prob_dist.png")


# 5. Precision-Recall Curve
from sklearn.metrics import precision_recall_curve, average_precision_score

precision, recall, _ = precision_recall_curve(y_test, y_prob)
avg_precision = average_precision_score(y_test, y_prob)

plt.figure(figsize=(6, 4))
plt.plot(recall, precision, color='purple', lw=2, label=f'PR curve (AP = {avg_precision:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
plt.grid(True, alpha=0.3)
plt.savefig('precision_recall_curve.png')
print("Saved precision_recall_curve.png")

# --- SAVE MODEL ---
import joblib

# Save the best model and the preprocessor
joblib.dump(best_model, 'diabetes_model.pkl')
joblib.dump(preprocessor, 'preprocessor.pkl')
print("\nModel saved to 'diabetes_model.pkl'")
print("Preprocessor saved to 'preprocessor.pkl'")

