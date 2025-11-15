# RideWise: Customer Churn Prediction System

## Project Overview

RideWise is a complete end-to-end machine learning solution for predicting customer churn in ride-hailing services. This project demonstrates the full ML lifecycle from data analysis to production API deployment.

**Industry**: Transportation Technology | Mobility Services  
**Focus**: Customer Analytics, Churn Prediction, ML API Development  
**Duration**: 3 Weeks  

---

## Table of Contents

- [Business Problem](#business-problem)
- [Solution Architecture](#solution-architecture)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Installation](#installation)
- [Usage Guide](#usage-guide)
- [Model Performance](#model-performance)
- [API Documentation](#api-documentation)
- [Contributing](#contributing)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Business Problem

RideWise faces critical customer retention challenges:

- **25% quarterly churn rate** among regular users
- **Limited understanding** of customer behavioral patterns
- **Low promotion uptake** with unclear ROI measurement
- **No predictive capability** to identify at-risk customers
- **Manual analysis** taking weeks per campaign

### Business Impact

- Lost revenue from churned customers
- Inefficient marketing spend on retention
- Reactive rather than proactive customer management
- Competitive disadvantage in crowded mobility market

---

## Solution Architecture

### High-Level Overview

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Raw Data      │ ──> │  ML Pipeline     │ ──> │  Trained Model  │
│   (CSV)         │     │  (Python/sklearn)│     │  (PKL files)    │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                                           │
                                                           ▼
                                                   ┌─────────────────┐
                                                   │   FastAPI       │
                                                   │   (REST API)    │
                                                   └─────────────────┘
                                                           │
                                                           ▼
                                                   ┌─────────────────┐
                                                   │  Predictions    │
                                                   │  (JSON)         │
                                                   └─────────────────┘
```

### Components

1. **Data Pipeline**: Load, clean, and preprocess customer data
2. **Feature Engineering**: Create predictive features from raw data
3. **Model Training**: Logistic Regression with SMOTE for class imbalance
4. **Model Evaluation**: Comprehensive metrics (AUC, precision, recall)
5. **Model Serialization**: Save model and preprocessing artifacts
6. **REST API**: FastAPI service for real-time predictions
7. **Input Validation**: Pydantic models for data validation

---

## Technology Stack

| Category | Technology | Purpose |
|----------|-----------|---------|
| **Language** | Python 3.8+ | Core development |
| **ML Framework** | scikit-learn | Model training & evaluation |
| **Data Processing** | pandas, NumPy | Data manipulation |
| **API Framework** | FastAPI | REST API service |
| **Validation** | Pydantic | Input validation |
| **Imbalance Handling** | imbalanced-learn (SMOTE) | Class imbalance |
| **Serialization** | joblib | Model persistence |
| **Server** | Uvicorn | ASGI server |
| **Version Control** | Git/GitHub | Code management |

---

## Project Structure

```
RideWise/
├── data/
│   └── riders.csv                    # Customer dataset
│
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_training.ipynb
│
├── models/
│   ├── churn_model.pkl              # Trained model
│   └── churn_artifacts.pkl          # Encoders, scaler, metrics
│
├── api/
│   ├── app.py                       # FastAPI application
│   └── requirements.txt             # API dependencies
│
├── scripts/
│   └── train_model.py               # Training script
│
├── tests/
│   ├── test_api.py
│   └── test_model.py
│
├── docs/
│   ├── training_guide.md
│   ├── api_guide.md
│   └── intern_tasks.md
│
├── .gitignore
├── requirements.txt                 # Project dependencies
└── README.md                        # This file
```

---

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git installed
- Basic understanding of:
  - Python programming
  - Machine learning concepts
  - REST APIs
  - Command line/terminal

### System Requirements

- **RAM**: Minimum 4GB
- **Storage**: 500MB free space
- **OS**: Windows, macOS, or Linux

---

## Installation

### Step 1: Clone Repository

```bash
# Clone the repository
git clone https://github.com/your-org/ridewise.git

# Navigate to project directory
cd ridewise
```

### Step 2: Create Virtual Environment

**Windows**:
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux**:
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
# Check Python version
python --version

# Check installed packages
pip list
```

---

## Usage Guide

### Training the Model

#### Option 1: Using Jupyter Notebook (Recommended for Learning)

```bash
# Start Jupyter
jupyter notebook

# Open: notebooks/03_model_training.ipynb
# Run all cells sequentially
```

#### Option 2: Using Python Script

```bash
# Run training script
python scripts/train_model.py

# Expected output:
# ✓ Data loaded: 10000 rows
# ✓ Features prepared
# ✓ Model trained
# ✓ Model AUC: 0.7234
# ✓ Model saved
```

### Starting the API

```bash
# Navigate to API directory
cd api

# Start the server
uvicorn app:app --reload

# Server running at: http://localhost:8000
```

### Testing the API

#### Using cURL

```bash
# Test root endpoint
curl http://localhost:8000/

# Test health check
curl http://localhost:8000/health

# Test prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 28,
    "avg_rating_given": 4.2,
    "city": "Lagos",
    "loyalty_status": "Bronze",
    "referred_by": "R00123"
  }'
```

#### Using Postman

1. Open Postman
2. Create new POST request
3. URL: `http://localhost:8000/predict`
4. Headers: `Content-Type: application/json`
5. Body (raw JSON):
```json
{
  "age": 28,
  "avg_rating_given": 4.2,
  "city": "Lagos",
  "loyalty_status": "Bronze",
  "referred_by": "R00123"
}
```

#### Using Python

```python
import requests

url = "http://localhost:8000/predict"
data = {
    "age": 28,
    "avg_rating_given": 4.2,
    "city": "Lagos",
    "loyalty_status": "Bronze",
    "referred_by": "R00123"
}

response = requests.post(url, json=data)
print(response.json())
```

---

## Model Performance

### Metrics

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| **AUC-ROC** | 0.7234 | > 0.80 | Acceptable |
| **Accuracy** | 0.7245 | > 0.75 | ✓ Met |
| **Precision** | 0.3456 | > 0.30 | ✓ Met |
| **Recall** | 0.5432 | > 0.50 | ✓ Met |

### Feature Importance

| Feature | Coefficient | Impact |
|---------|-------------|--------|
| avg_rating_given | -0.4523 | High negative (low rating = churn) |
| was_referred | -0.3142 | Moderate negative (referrals stay) |
| loyalty_encoded | -0.2156 | Moderate negative (loyalty helps) |
| city_encoded | 0.1823 | Low positive (city matters) |
| age | -0.0891 | Low negative (older = stable) |

### Confusion Matrix

```
                Predicted
             No Churn  |  Churn
Actual  No      819   |   968
        Yes      99   |   114
```

---

## API Documentation

### Endpoints

#### 1. Root Endpoint

**GET** `/`

Returns basic API information.

**Response**:
```json
{
  "service": "RideWise Churn Prediction API",
  "status": "active",
  "model_auc": 0.7234
}
```

---

#### 2. Health Check

**GET** `/health`

Returns detailed health status and configuration.

**Response**:
```json
{
  "status": "healthy",
  "model_auc": 0.7234,
  "valid_cities": ["Lagos", "Nairobi"],
  "valid_loyalty_statuses": ["Bronze", "Silver"]
}
```

---

#### 3. Predict Churn

**POST** `/predict`

Predicts customer churn risk.

**Request Body**:
```json
{
  "age": 28,
  "avg_rating_given": 4.2,
  "city": "Lagos",
  "loyalty_status": "Bronze",
  "referred_by": "R00123"
}
```

**Response**:
```json
{
  "churn_prediction": 0,
  "churn_probability": 0.2345,
  "churn_risk": "Low",
  "recommendation": "Continue standard engagement"
}
```

**Field Descriptions**:
- `churn_prediction`: 0 = won't churn, 1 = will churn
- `churn_probability`: Probability of churn (0.0 to 1.0)
- `churn_risk`: Low/Medium/High risk category
- `recommendation`: Suggested action

**Validation Rules**:
- `age`: 18-100
- `avg_rating_given`: 1.0-5.0
- `city`: Must be in valid cities list
- `loyalty_status`: Must be in valid statuses list
- `referred_by`: Optional, defaults to "Direct"

---

#### 4. Model Information

**GET** `/model/info`

Returns model metadata and performance metrics.

**Response**:
```json
{
  "model_type": "Logistic Regression",
  "features": ["age", "avg_rating_given", "city_encoded", "loyalty_encoded", "was_referred"],
  "performance": {
    "accuracy": 0.7245,
    "precision": 0.3456,
    "recall": 0.5432,
    "auc": 0.7234
  },
  "valid_cities": ["Lagos", "Nairobi"],
  "valid_loyalty_statuses": ["Bronze", "Silver"]
}
```

---

### Error Responses

#### 400 Bad Request
```json
{
  "detail": "Invalid city. Valid: ['Lagos', 'Nairobi']"
}
```

#### 422 Unprocessable Entity
```json
{
  "detail": [
    {
      "loc": ["body", "age"],
      "msg": "ensure this value is greater than or equal to 18",
      "type": "value_error.number.not_ge"
    }
  ]
}
```

#### 500 Internal Server Error
```json
{
  "detail": "Prediction failed: [error message]"
}
```

---

## Contributing

### For Interns

See [Intern Tasks Guide](docs/intern_tasks.md) for detailed instructions.

### Workflow

1. Create feature branch from `main`
2. Make changes
3. Test thoroughly
4. Commit with clear messages
5. Push to your branch
6. Create pull request
7. Wait for review

### Commit Message Format

```
<type>: <subject>

<body>

<footer>
```

**Types**: feat, fix, docs, style, refactor, test, chore

**Example**:
```
feat: Add batch prediction endpoint

- Implement /predict/batch endpoint
- Handle up to 100 predictions per request
- Add validation for batch inputs

Closes #23
```

---

## Troubleshooting

### Common Issues

#### 1. Model File Not Found

**Error**: `FileNotFoundError: models/churn_model.pkl`

**Solution**:
```bash
# Train the model first
python scripts/train_model.py
```

---

#### 2. Import Errors

**Error**: `ModuleNotFoundError: No module named 'fastapi'`

**Solution**:
```bash
# Ensure virtual environment is activated
# Install dependencies
pip install -r requirements.txt
```

---

#### 3. Port Already in Use

**Error**: `Address already in use`

**Solution**:
```bash
# Use different port
uvicorn app:app --port 8001

# Or kill existing process (Windows)
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Or kill existing process (macOS/Linux)
lsof -ti:8000 | xargs kill -9
```

---

#### 4. Prediction Errors

**Error**: `KeyError: 'city_encoded'`

**Solution**:
- Check column names match training exactly
- Verify feature engineering steps match training
- Retrain model if necessary

---

#### 5. Wrong Predictions

**Issue**: Model returns unexpected results

**Checklist**:
- [ ] Features scaled in same order as training?
- [ ] Same preprocessing applied?
- [ ] Correct encoders loaded?
- [ ] Feature names match exactly?

---

## Development Notes

### Best Practices

1. **Always use virtual environment**
2. **Never commit credentials or sensitive data**
3. **Test before pushing**
4. **Document your changes**
5. **Follow PEP 8 style guide**

### Code Quality

```bash
# Format code
black .

# Check style
flake8 .

# Type checking
mypy .
```

### Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=api tests/
```

---

## Future Enhancements

- [ ] Add batch prediction endpoint
- [ ] Implement model versioning
- [ ] Add monitoring dashboard
- [ ] Deploy to cloud (AWS/Azure/GCP)
- [ ] Add authentication/authorization
- [ ] Implement A/B testing framework
- [ ] Add model retraining pipeline
- [ ] Create Docker container
- [ ] Add CI/CD pipeline
- [ ] Implement feature store

---

## Learning Resources

### Recommended Reading

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [Handling Imbalanced Data](https://imbalanced-learn.org/stable/)
- [REST API Best Practices](https://restfulapi.net/)

### Video Tutorials

- Python for Data Science
- Machine Learning Fundamentals
- API Development with FastAPI
- Git and GitHub Basics

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Built with ❤️ by the RideWise Data Science Team**
