# train_model.py - Fixed Version
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib
import warnings
warnings.filterwarnings('ignore')

print("🚀 Starting model training...")

# Load and prepare data
def load_and_prepare_data():
    """Load and clean the Adult Census dataset"""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    
    # Column names for the Adult dataset
    columns = ['age', 'workclass', 'fnlwgt', 'education', 'educational_num',
               'marital_status', 'occupation', 'relationship', 'race', 'sex',
               'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income']
    
    # Load data
    df = pd.read_csv(url, names=columns, na_values=' ?', skipinitialspace=True)
    
    print(f"📊 Original dataset shape: {df.shape}")
    
    # Clean data
    df = df.dropna()
    print(f"📊 After removing missing values: {df.shape}")
    
    # Convert target variable
    df['income'] = df['income'].apply(lambda x: 1 if x == '>50K' else 0)
    
    # Check class distribution
    print(f"📊 Income distribution:")
    print(f"   ≤50K: {(df['income'] == 0).sum()} ({(df['income'] == 0).mean()*100:.1f}%)")
    print(f"   >50K: {(df['income'] == 1).sum()} ({(df['income'] == 1).mean()*100:.1f}%)")
    
    return df

# Load data
df = load_and_prepare_data()

# Define features and target
X = df.drop('income', axis=1)
y = df['income']

# Define categorical and numerical columns
categorical_columns = ['workclass', 'education', 'marital_status', 'occupation',
                      'relationship', 'race', 'sex', 'native_country']
numerical_columns = ['age', 'fnlwgt', 'educational_num', 'capital_gain', 
                    'capital_loss', 'hours_per_week']

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_columns),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_columns)
    ],
    remainder='drop'
)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"📊 Training set: {X_train.shape}")
print(f"📊 Test set: {X_test.shape}")

# Fit preprocessor and transform data
print("🔄 Preprocessing data...")
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Train the model
print("🤖 Training Random Forest model...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_processed, y_train)

# Evaluate the model
print("📈 Evaluating model...")
y_pred = model.predict(X_test_processed)
accuracy = accuracy_score(y_test, y_pred)

print(f"✅ Model Accuracy: {accuracy:.4f}")
print(f"\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=['≤50K', '>50K']))

# Test with specific high-income example
def test_high_income_example(model, preprocessor):
    """Test model with expected high-income profile"""
    test_data = {
        'age': 45,
        'workclass': 'Private',
        'fnlwgt': 192776,
        'education': 'Bachelors',
        'educational_num': 13,
        'marital_status': 'Married-civ-spouse',
        'occupation': 'Exec-managerial',
        'relationship': 'Husband',
        'race': 'White',
        'sex': 'Male',
        'capital_gain': 5000,
        'capital_loss': 0,
        'hours_per_week': 60,
        'native_country': 'United-States'
    }
    
    test_df = pd.DataFrame([test_data])
    test_processed = preprocessor.transform(test_df)
    prediction = model.predict(test_processed)[0]
    probabilities = model.predict_proba(test_processed)[0]
    
    print(f"\n🧪 Testing High-Income Example:")
    print(f"   Prediction: {'>50K' if prediction == 1 else '≤50K'}")
    print(f"   Confidence: {max(probabilities)*100:.1f}%")
    print(f"   Probabilities: ≤50K={probabilities[0]:.3f}, >50K={probabilities[1]:.3f}")
    
    if prediction == 1:
        print("   ✅ TEST PASSED!")
    else:
        print("   ❌ TEST FAILED!")
    
    return prediction == 1

# Run validation test
test_passed = test_high_income_example(model, preprocessor)

# Save the model and preprocessor
print("\n💾 Saving model and preprocessor...")
joblib.dump(model, 'salary_model.pkl')
joblib.dump(preprocessor, 'salary_preprocessor.pkl')

# Save feature information for the app
feature_info = {
    'categorical_columns': categorical_columns,
    'numerical_columns': numerical_columns,
    'all_categories': {}
}

# Get all unique categories for each categorical column
for col in categorical_columns:
    feature_info['all_categories'][col] = sorted(df[col].unique().tolist())

joblib.dump(feature_info, 'feature_info.pkl')

print("✅ Model training completed and files saved!")
print("📁 Files created: salary_model.pkl, salary_preprocessor.pkl, feature_info.pkl")
print(f"\n🎯 Model ready! Validation test: {'PASSED' if test_passed else 'FAILED'}")