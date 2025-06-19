import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

try:
    # Updated to load the uploaded Churns (Bank).csv file
    df = pd.read_csv('Churns (Bank).csv')
    print("Data loaded successfully from 'Churns (Bank).csv'.")
    print(df.head())
    print(df.info())
except FileNotFoundError:
    print("Error: 'Churns (Bank).csv' not found. Generating a dummy dataset for demonstration.")
    data = {
        'customer_id': range(1, 1001),
        'age': np.random.randint(18, 70, 1000),
        'gender': np.random.choice(['Male', 'Female', 'Other'], 1000, p=[0.48, 0.48, 0.04]),
        'subscription_duration_months': np.random.randint(1, 60, 1000),
        'monthly_bill': np.random.uniform(20, 150, 1000),
        'data_usage_gb': np.random.uniform(5, 100, 1000),
        'customer_service_calls': np.random.randint(0, 10, 1000),
        'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'], 1000, p=[0.5, 0.3, 0.2]),
        'payment_method': np.random.choice(['Credit card', 'Bank transfer', 'Electronic check', 'Mailed check'], 1000),
        'has_online_security': np.random.choice(['Yes', 'No'], 1000),
        'total_spent': np.random.uniform(50, 5000, 1000),
        'churn': np.random.choice([0, 1], 1000, p=[0.75, 0.25])
    }
    df = pd.DataFrame(data)
    print("\nDummy Data created for demonstration:")
    print(df.head())
    print(df.info())

# Updated TARGET to 'Exited' assuming it's the churn column in Churns (Bank).csv
TARGET = 'Exited' # Common target column name for bank churn datasets
FEATURES = [col for col in df.columns if col not in ['customer_id', 'RowNumber', 'Surname', TARGET]] # Exclude additional non-feature columns

X = df[FEATURES]
y = df[TARGET]

numerical_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(include='object').columns.tolist()

print(f"\nIdentified Numerical Features: {numerical_features}")
print(f"Identified Categorical Features: {categorical_features}")

numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"\nTraining set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")
print(f"Churn rate in training set: {y_train.mean():.2f}")
print(f"Churn rate in testing set: {y_test.mean():.2f}")

models = {
    'Logistic Regression': LogisticRegression(random_state=42, solver='liblinear', class_weight='balanced', max_iter=1000),
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100, class_weight='balanced'),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42, n_estimators=100, learning_rate=0.1)
}

results = {}

for name, model in models.items():
    print(f"\n--- Training {name} ---")

    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', model)])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

    print(f"\nClassification Report for {name}:\n")
    print(classification_report(y_test, y_pred))

    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"ROC AUC Score for {name}: {roc_auc:.4f}")

    results[name] = {
        'model': pipeline,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'roc_auc': roc_auc
    }

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Churn (0)', 'Churn (1)'],
                yticklabels=['No Churn (0)', 'Churn (1)'])
    plt.title(f'Confusion Matrix for {name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier (AUC = 0.50)')
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Recall)')
    plt.title(f'ROC Curve for {name}')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

best_model_name = max(results, key=lambda k: results[k]['roc_auc'])
print(f"\n--- Best Model based on ROC AUC: {best_model_name} ---")
print(f"ROC AUC: {results[best_model_name]['roc_auc']:.4f}")

if best_model_name in ['Random Forest', 'Gradient Boosting']:
    best_tree_model = results[best_model_name]['model'].named_steps['classifier']

    transformed_features_names = []
    dummy_row_data = {col: [0 if col in numerical_features else 'dummy_cat' for col in FEATURES]}
    dummy_df = pd.DataFrame(dummy_row_data)

    onehot_encoder = results[best_model_name]['model'].named_steps['preprocessor'].named_transformers_['cat']
    ohe_feature_names = onehot_encoder.get_feature_names_out(categorical_features)
    all_feature_names = numerical_features + list(ohe_feature_names)

    importances = best_tree_model.feature_importances_
    feature_importances_df = pd.DataFrame({'feature': all_feature_names, 'importance': importances})
    feature_importances_df = feature_importances_df.sort_values(by='importance', ascending=False)
    print(f"\nFeature Importances ({best_model_name}):\n", feature_importances_df.head(10))

    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importances_df.head(10))
    plt.title(f'Top 10 Feature Importances ({best_model_name})')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.show()

elif best_model_name == 'Logistic Regression':
    best_lr_model = results['Logistic Regression']['model'].named_steps['classifier']

    onehot_encoder = results[best_model_name]['model'].named_steps['preprocessor'].named_transformers_['cat']
    ohe_feature_names = onehot_encoder.get_feature_names_out(categorical_features)
    all_feature_names = numerical_features + list(ohe_feature_names)

    coefficients = best_lr_model.coef_[0]
    coef_df = pd.DataFrame({'feature': all_feature_names, 'coefficient': coefficients})
    coef_df['abs_coefficient'] = np.abs(coef_df['coefficient'])
    coef_df = coef_df.sort_values(by='abs_coefficient', ascending=False)
    print("\nFeature Coefficients (Logistic Regression):\n", coef_df.head(10))

    plt.figure(figsize=(10, 6))
    sns.barplot(x='coefficient', y='feature', data=coef_df.head(10))
    plt.title('Top 10 Feature Coefficients (Logistic Regression)')
    plt.xlabel('Coefficient Value')
    plt.ylabel('Feature')
    plt.show()
