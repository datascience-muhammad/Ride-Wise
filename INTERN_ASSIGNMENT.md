# RideWise Churn Prediction - Intern Assignment

## Goal
Build ML pipeline → Train model → Deploy REST API → Test it

**Deadline**: Tomorrow (November 19th, 2024)  
**Repository**: https://github.com/datascience-muhammad/Ride-Wise

---

## Before You Start

**Email me NOW to get repository access:**

```
Name: [Your Full Name]
GitHub Username: [your-github-username]
Email: [your-email]
```

---

## Task 1: Build & Train Model

### Setup

```bash
git clone https://github.com/datascience-muhammad/Ride-Wise.git
cd Ride-Wise
git checkout -b intern/your-name

python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows
pip install -r requirements.txt

git push -u origin intern/your-name
```

---

### Training Pipeline

Create: `notebooks/my_training.ipynb`

**Complete Code:**
```python
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from imblearn.over_sampling import SMOTE

# Load
df = pd.read_csv("../data/riders.csv")
data = df.copy()

# Preprocess
data['city'] = data['city'].str.strip().str.title()
data['loyalty_status'] = data['loyalty_status'].str.strip().str.title()
data['referred_by'] = data['referred_by'].fillna('Direct').str.strip().str.title()
data['was_referred'] = (data['referred_by'] != 'Direct').astype(int)

# Encode
city_encoder = LabelEncoder()
data['city_encoded'] = city_encoder.fit_transform(data['city'])
loyalty_encoder = LabelEncoder()
data['loyalty_encoded'] = loyalty_encoder.fit_transform(data['loyalty_status'])

# Scale
scaler = StandardScaler()
data[['age', 'avg_rating_given']] = scaler.fit_transform(data[['age', 'avg_rating_given']])

# Train
feature_cols = ['age', 'avg_rating_given', 'city_encoded', 'loyalty_encoded', 'was_referred']
X = data[feature_cols]
y = (data['churn_prob'] > 0.5).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

model = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced', solver='liblinear')
model.fit(X_train_balanced, y_train_balanced)

# Evaluate
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print(f"AUC: {roc_auc_score(y_test, y_proba):.4f}")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")

# Save
joblib.dump(model, '../models/churn_model.pkl')

artifacts = {
    'encoders': {
        'city': city_encoder,
        'loyalty': loyalty_encoder,
        'scaler': scaler,
        'valid_cities': list(city_encoder.classes_),
        'valid_loyalty_statuses': list(loyalty_encoder.classes_)
    },
    'feature_names': feature_cols,
    'metrics': {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_proba)
    },
    'model_version': '1.0.0'
}
joblib.dump(artifacts, '../models/churn_artifacts.pkl')

print("✓ Model saved successfully!")
```

**Commit:**
```bash
git add notebooks/my_training.ipynb models/
git commit -m "feat: Complete training pipeline"
git push
```

---

## Task 2: Test API

### Start API

```bash
cd api
uvicorn my_app:app --reload
```

### Browser Tests

Visit and screenshot:
- http://localhost:8000
- http://localhost:8000/docs
- http://localhost:8000/health

---

### Postman Tests

**Test 1: Lagos Customer**
```
POST http://localhost:8000/predict
```
```json
{
  "age": 28,
  "avg_rating_given": 4.2,
  "city": "Lagos",
  "loyalty_status": "Bronze",
  "referred_by": "R00123"
}
```

**Test 2: Nairobi Customer**
```json
{
  "age": 45,
  "avg_rating_given": 3.8,
  "city": "Nairobi",
  "loyalty_status": "Silver",
  "referred_by": "Direct"
}
```

**Test 3: Invalid City**
```json
{
  "age": 30,
  "avg_rating_given": 4.0,
  "city": "London",
  "loyalty_status": "Bronze"
}
```

Screenshot all results.

---

## Submit (Deadline: Tomorrow 11:59 PM)

### Checklist
- [ ] Training notebook works
- [ ] Model saved (AUC > 0.65)
- [ ] API runs
- [ ] 3 Postman tests done
- [ ] Screenshots captured

### Pull Request

1. Go to: https://github.com/datascience-muhammad/Ride-Wise
2. New Pull Request
3. `intern/your-name` → `main`
4. Title: `[Your Name] - RideWise Complete`
5. Add screenshots
6. Submit

---

## Quick Help

**Stuck?**
- Slack: #ridewise-help
- Email: [instructor.email]

**Commands:**
```bash
# Activate venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows

# Start API
cd api
uvicorn my_app:app --reload

# Git
git add .
git commit -m "message"
git push
```

---

**DEADLINE: Tomorrow 11:59 PM - Don't delay! 🚀**
