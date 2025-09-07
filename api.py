"""
Diabetic Readmission Prediction API
===================================

FastAPI application for predicting diabetic patient readmission within 30 days.
Uses a trained XGBoost model saved with pickle.

Author: AI Assistant
Date: 2025
"""

import os
import sys
import pickle
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path

import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import uvicorn

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add src path for imports if needed
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Initialize FastAPI app
app = FastAPI(
    title="Diabetic Readmission Prediction API",
    description="API for predicting diabetic patient readmission within 30 days using XGBoost model",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variable to store the loaded model
model_data = None
feature_names = []

class PatientData(BaseModel):
    """
    Patient data model for prediction input.
    All features required by the diabetic readmission model.
    """
    # Demographic and basic info
    age_numeric: float = Field(..., ge=0, le=120, description="Patient age in years")
    time_in_hospital: int = Field(..., ge=1, le=14, description="Days spent in hospital")
    
    # Medical procedures and tests
    num_lab_procedures: int = Field(..., ge=0, description="Number of lab procedures performed")
    num_procedures: int = Field(..., ge=0, description="Number of procedures performed") 
    num_medications: int = Field(..., ge=0, description="Number of distinct medications")
    number_outpatient: int = Field(..., ge=0, description="Number of outpatient visits")
    number_emergency: int = Field(..., ge=0, description="Number of emergency visits")
    number_inpatient: int = Field(..., ge=0, description="Number of inpatient visits")
    number_diagnoses: int = Field(..., ge=0, description="Number of diagnoses")
    
    # Computed features
    total_healthcare_encounters: int = Field(..., ge=0, description="Total healthcare encounters")
    emergency_ratio: float = Field(..., ge=0, le=1, description="Ratio of emergency visits")
    inpatient_ratio: float = Field(..., ge=0, le=1, description="Ratio of inpatient visits")
    medical_complexity_score: float = Field(..., ge=0, description="Medical complexity score")
    short_stay: int = Field(..., ge=0, le=1, description="1 if short stay, 0 otherwise")
    long_stay: int = Field(..., ge=0, le=1, description="1 if long stay, 0 otherwise")
    total_medications_changed: int = Field(..., ge=0, description="Total medications changed")
    diabetes_managed: int = Field(..., ge=0, le=1, description="1 if diabetes managed, 0 otherwise")
    
    # Encoded categorical features
    race_encoded: int = Field(..., ge=0, description="Encoded race")
    gender_encoded: int = Field(..., ge=0, description="Encoded gender")
    payer_code_encoded: int = Field(..., ge=0, description="Encoded payer code")
    medical_specialty_encoded: int = Field(..., ge=0, description="Encoded medical specialty")
    max_glu_serum_encoded: int = Field(..., ge=0, description="Encoded max glucose serum")
    A1Cresult_encoded: int = Field(..., ge=0, description="Encoded A1C result")
    change_encoded: int = Field(..., ge=0, description="Encoded change")
    diabetesMed_encoded: int = Field(..., ge=0, description="Encoded diabetes medication")
    diag_1_category_encoded: int = Field(..., ge=0, description="Encoded primary diagnosis category")
    diag_2_category_encoded: int = Field(..., ge=0, description="Encoded secondary diagnosis category")
    diag_3_category_encoded: int = Field(..., ge=0, description="Encoded tertiary diagnosis category")
    
    # Medication binary features
    metformin_binary: int = Field(..., ge=0, le=1, description="1 if metformin prescribed")
    repaglinide_binary: int = Field(..., ge=0, le=1, description="1 if repaglinide prescribed")
    nateglinide_binary: int = Field(..., ge=0, le=1, description="1 if nateglinide prescribed")
    chlorpropamide_binary: int = Field(..., ge=0, le=1, description="1 if chlorpropamide prescribed")
    glimepiride_binary: int = Field(..., ge=0, le=1, description="1 if glimepiride prescribed")
    acetohexamide_binary: int = Field(..., ge=0, le=1, description="1 if acetohexamide prescribed")
    glipizide_binary: int = Field(..., ge=0, le=1, description="1 if glipizide prescribed")
    glyburide_binary: int = Field(..., ge=0, le=1, description="1 if glyburide prescribed")
    tolbutamide_binary: int = Field(..., ge=0, le=1, description="1 if tolbutamide prescribed")
    pioglitazone_binary: int = Field(..., ge=0, le=1, description="1 if pioglitazone prescribed")
    rosiglitazone_binary: int = Field(..., ge=0, le=1, description="1 if rosiglitazone prescribed")
    acarbose_binary: int = Field(..., ge=0, le=1, description="1 if acarbose prescribed")
    miglitol_binary: int = Field(..., ge=0, le=1, description="1 if miglitol prescribed")
    troglitazone_binary: int = Field(..., ge=0, le=1, description="1 if troglitazone prescribed")
    tolazamide_binary: int = Field(..., ge=0, le=1, description="1 if tolazamide prescribed")
    examide_binary: int = Field(..., ge=0, le=1, description="1 if examide prescribed")
    citoglipton_binary: int = Field(..., ge=0, le=1, description="1 if citoglipton prescribed")
    insulin_binary: int = Field(..., ge=0, le=1, description="1 if insulin prescribed")
    
    # Combination medication binary features
    glyburide_metformin_binary: int = Field(..., ge=0, le=1, description="1 if glyburide-metformin prescribed", alias="glyburide-metformin_binary")
    glipizide_metformin_binary: int = Field(..., ge=0, le=1, description="1 if glipizide-metformin prescribed", alias="glipizide-metformin_binary")
    glimepiride_pioglitazone_binary: int = Field(..., ge=0, le=1, description="1 if glimepiride-pioglitazone prescribed", alias="glimepiride-pioglitazone_binary")
    metformin_rosiglitazone_binary: int = Field(..., ge=0, le=1, description="1 if metformin-rosiglitazone prescribed", alias="metformin-rosiglitazone_binary")
    metformin_pioglitazone_binary: int = Field(..., ge=0, le=1, description="1 if metformin-pioglitazone prescribed", alias="metformin-pioglitazone_binary")

    class Config:
        allow_population_by_field_name = True
        schema_extra = {
            "example": {
                "age_numeric": 65.0,
                "time_in_hospital": 3,
                "num_lab_procedures": 45,
                "num_procedures": 2,
                "num_medications": 15,
                "number_outpatient": 0,
                "number_emergency": 0,
                "number_inpatient": 0,
                "number_diagnoses": 7,
                "total_healthcare_encounters": 0,
                "emergency_ratio": 0.0,
                "inpatient_ratio": 0.0,
                "medical_complexity_score": 3.5,
                "short_stay": 0,
                "long_stay": 0,
                "total_medications_changed": 2,
                "diabetes_managed": 1,
                "race_encoded": 1,
                "gender_encoded": 0,
                "payer_code_encoded": 2,
                "medical_specialty_encoded": 5,
                "max_glu_serum_encoded": 1,
                "A1Cresult_encoded": 2,
                "change_encoded": 1,
                "diabetesMed_encoded": 1,
                "diag_1_category_encoded": 3,
                "diag_2_category_encoded": 8,
                "diag_3_category_encoded": 5,
                "metformin_binary": 1,
                "repaglinide_binary": 0,
                "nateglinide_binary": 0,
                "chlorpropamide_binary": 0,
                "glimepiride_binary": 0,
                "acetohexamide_binary": 0,
                "glipizide_binary": 0,
                "glyburide_binary": 0,
                "tolbutamide_binary": 0,
                "pioglitazone_binary": 0,
                "rosiglitazone_binary": 0,
                "acarbose_binary": 0,
                "miglitol_binary": 0,
                "troglitazone_binary": 0,
                "tolazamide_binary": 0,
                "examide_binary": 0,
                "citoglipton_binary": 0,
                "insulin_binary": 1,
                "glyburide-metformin_binary": 0,
                "glipizide-metformin_binary": 0,
                "glimepiride-pioglitazone_binary": 0,
                "metformin-rosiglitazone_binary": 0,
                "metformin-pioglitazone_binary": 0
            }
        }

class PredictionResponse(BaseModel):
    """Response model for predictions"""
    patient_id: Optional[str] = None
    prediction: int = Field(..., description="0 = No readmission, 1 = Readmission within 30 days")
    probability: float = Field(..., ge=0, le=1, description="Probability of readmission within 30 days")
    risk_level: str = Field(..., description="Risk level: Low, Medium, or High")
    model_version: str = Field(..., description="Model version used for prediction")

class HealthResponse(BaseModel):
    """Health check response model"""
    status: str
    message: str
    model_loaded: bool

def load_model() -> bool:
    """Attempt to load the trained model. Returns True if loaded else False."""
    global model_data, feature_names
    env_path = os.getenv("MODEL_PATH")
    candidates = [
        Path(env_path) if env_path else None,
        Path("diabetic_model.pkl"),
        Path("models/diabetic_model.pkl"),
    ]
    candidates = [p for p in candidates if p is not None]

    chosen = None
    for p in candidates:
        if p.exists():
            chosen = p
            break

    if not chosen:
        logger.warning("Model file not found. Checked: %s", ", ".join(str(p) for p in candidates))
        model_data = None
        feature_names = []
        return False

    try:
        with open(chosen, 'rb') as f:
            model_data = pickle.load(f)
        feature_names = model_data['feature_names']
        logger.info(f"Model loaded successfully from {chosen}")
        logger.info(f"Model type: {model_data['model_type']}")
        logger.info(f"Number of features: {len(feature_names)}")
        return True
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        model_data = None
        feature_names = []
        return False

def classify_risk(probability: float) -> str:
    """Classify risk level based on probability"""
    if probability < 0.3:
        return "Low"
    elif probability < 0.7:
        return "Medium"
    else:
        return "High"

@app.on_event("startup")
async def startup_event():
    """Try to load model on startup, but keep API alive even if missing."""
    logger.info("Starting up Diabetic Readmission Prediction API...")
    loaded = load_model()
    if loaded:
        logger.info("API startup completed successfully!")
    else:
        logger.warning("API started without model. Place the pickle and call /reload_model.")

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "Diabetic Readmission Prediction API",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if model_data is not None else "unhealthy",
        message="API is running" if model_data is not None else "Model not loaded",
        model_loaded=model_data is not None
    )

@app.get("/test")
async def test_endpoint():
    """Simple test endpoint"""
    return {"message": "API is working", "timestamp": pd.Timestamp.now().isoformat()}

@app.post("/reload_model", response_model=HealthResponse)
async def reload_model_endpoint():
    """Reload model on demand."""
    loaded = load_model()
    return HealthResponse(
        status="healthy" if loaded else "unhealthy",
        message="Model loaded" if loaded else "Model not loaded",
        model_loaded=loaded,
    )

@app.get("/features", response_model=Dict[str, List[str]])
async def get_features():
    """Get list of required features"""
    if model_data is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    return {"features": feature_names}

@app.post("/predict", response_model=PredictionResponse)
async def predict_readmission(patient_data: PatientData, patient_id: Optional[str] = None):
    """
    Predict diabetic patient readmission within 30 days.
    
    Args:
        patient_data: Patient clinical data
        patient_id: Optional patient identifier
        
    Returns:
        Prediction results with probability and risk level
    """
    if model_data is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    try:
        # Convert patient data to DataFrame
        input_dict = patient_data.dict(by_alias=True)
        
        # Ensure all required features are present and in correct order
        feature_values = []
        for feature in feature_names:
            if feature in input_dict:
                feature_values.append(input_dict[feature])
            else:
                # Handle missing features with default values
                logger.warning(f"Feature {feature} not found in input, using default value 0")
                feature_values.append(0.0)
        
        # Create DataFrame
        X = pd.DataFrame([feature_values], columns=feature_names)
        
        # Make prediction
        calibrated_model = model_data['calibrated_model']
        scaler = model_data['scaler']
        
        # Scale features
        X_scaled = pd.DataFrame(
            scaler.transform(X),
            columns=X.columns
        )
        
        # Get prediction and probability
        prediction = calibrated_model.predict(X_scaled)[0]
        probability = calibrated_model.predict_proba(X_scaled)[0, 1]
        
        # Classify risk level
        risk_level = classify_risk(probability)
        
        logger.info(f"Prediction made - Patient: {patient_id}, Prediction: {prediction}, Probability: {probability:.4f}")
        
        return PredictionResponse(
            patient_id=patient_id,
            prediction=int(prediction),
            probability=float(probability),
            risk_level=risk_level,
            model_version=f"{model_data['model_type']}_v1.0"
        )
        
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error making prediction: {str(e)}"
        )

@app.post("/batch_predict")
async def batch_predict(patients_data: List[PatientData]):
    """
    Predict readmission for multiple patients.
    
    Args:
        patients_data: List of patient clinical data
        
    Returns:
        List of prediction results
    """
    if model_data is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    try:
        predictions = []
        
        for i, patient_data in enumerate(patients_data):
            patient_id = f"batch_patient_{i}"
            prediction = await predict_readmission(patient_data, patient_id)
            predictions.append(prediction)
        
        return {"predictions": predictions}
        
    except Exception as e:
        logger.error(f"Error in batch prediction: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error in batch prediction: {str(e)}"
        )

if __name__ == "__main__":
    # Run the API server
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
