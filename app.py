# step 1: import libraries
from fastapi import FastAPI
from pydantic import BaseModel, Field, validator
import joblib
import numpy as np
import pandas as pd
import logging


# step 2: logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# step 3 : load models and artifacts
model = joblib.load(
    r"C:\Users\user\Desktop\NEW PROJECT\mntdataridewise_project\models\churn_model.pkl"
)
artifacts = joblib.load(
    r"C:\Users\user\Desktop\NEW PROJECT\mntdataridewise_project\models\churn_artifacts.pkl"
)

# step 4 : Extract components of the encoder and the artifacts
city_encoder = artifacts["encoders"]["city"]
loyalty_encoder = artifacts["encoders"]["loyalty"]
scaler = artifacts["encoders"]["scaler"]  # FIX: Now scaler exists
feature_names = artifacts["feature_names"]
model_metrics = artifacts["metrics"]

VALID_CITIES = artifacts["encoders"]["valid_cities"]
VALID_LOYALTY_STATUSES = artifacts["encoders"]["valid_loyalty_statuses"]

# step 5 : initialize FastAPI app - starting the ignition
app = FastAPI(title="RideWise Churn Prediction API", version="1.0")


# step 6 : create a Validation model for our input data
class RiderData(BaseModel):
    age: float = Field(..., ge=18, le=100)
    avg_rating_given: float = Field(..., ge=1.0, le=5.0)
    city: str
    loyalty_status: str
    referred_by: str = "Direct"

    # example input data
    class Config:
        schema_extra = {
            "example": {
                "age": 29,
                "avg_rating_given": 4.5,
                "city": "Lagos",
                "loyalty_status": "Gold",
                "referred_by": "R00123",
            }
        }

    # step 7 : include validators to our input data
    @validator("city", "loyalty_status", "referred_by", pre=True)
    def normalize_text(cls, v):
        return v.strip().title() if isinstance(v, str) else v

    @validator("city")
    def validate_city(cls, v):
        if v not in VALID_CITIES:
            raise ValueError(f"Invalid city. Valid: {VALID_CITIES}")
        return v

    @validator("loyalty_status")
    def validate_loyalty(cls, v):
        if v not in VALID_LOYALTY_STATUSES:
            raise ValueError(f"Invalid loyalty. Valid: {VALID_LOYALTY_STATUSES}")
        return v


# step 8 :  endpoint 1 - default endpoint
@app.get("/")
def root():
    return {"service": "RideWise Churn Prediction API"}


# step 9 : endpoint 2 - health check endpoint
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "valid_cities": VALID_CITIES,
        "valid_loyalty_statuses": VALID_LOYALTY_STATUSES,
    }


# step 10:  endpoint 3 or address 3 - prediction endpoint


@app.post("/predict")
def predict(data: RiderData):
    # put a try/except block
    try:
        # step 10a : convert input data to dataframe
        df = pd.DataFrame([data.dict()])

        # step 10b : feature engineering
        df["was_referred"] = (df["referred_by"] != "Direct").astype(
            int
        )  # "R00123" -> 1, "Direct" -> 0
        df["city_encoded"] = city_encoder.transform(
            df["city"]
        )  # "Lagos" -> 0, "Nairobi" -> 1, "Cairo" -> 2
        df["loyalty_encoded"] = loyalty_encoder.transform(
            df["loyalty_status"]
        )  # "Gold" -> 0, "Silver" -> 1, "Bronze" -> 2

        # step 10c : feature scaling
        df[["age", "avg_rating_given"]] = scaler.transform(
            df[["age", "avg_rating_given"]]
        )

        # step 10d : select features for prediction
        features = df[feature_names]

        # step 10e : make prediction
        prediction = int(model.predict(features)[0])
        probability = float(model.predict_proba(features)[0][1])  # Probability of churn

        # step 10f : Risk Category

        if probability < 0.3:
            risk = "Low"
        elif probability < 0.7:
            risk = "Medium"
        else:
            risk = "High"

        # step 10g : recommendations based on risk
        recommendations = {
            "Low": "Maintain current engagement strategies.",
            "Medium": "Consider targeted promotions and engagement.",
            "High": "Immediate retention efforts required, such as special offers or personalized outreach.",
        }

        # step 10h : return response
        return {
            "churn_prediction": prediction,
            "churn_probability": round(probability, 4),
            "risk_category": risk,
            "recommendations": recommendations[risk],
        }

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return {"error": "An error occurred during prediction.", "details": str(e)}
