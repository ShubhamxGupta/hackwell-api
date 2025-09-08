import os
import pickle
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


app = FastAPI(title="Diabetic Readmission Prediction API", version="1.0.0")


class PatientData(BaseModel):
  # This mirrors the features used by the trained model
  age_numeric: float
  time_in_hospital: int
  num_lab_procedures: int
  num_procedures: int
  num_medications: int
  number_outpatient: int
  number_emergency: int
  number_inpatient: int
  number_diagnoses: int
  total_healthcare_encounters: int
  emergency_ratio: float
  inpatient_ratio: float
  medical_complexity_score: float
  short_stay: int
  long_stay: int
  total_medications_changed: int
  diabetes_managed: int
  race_encoded: int
  gender_encoded: int
  payer_code_encoded: int
  medical_specialty_encoded: int
  max_glu_serum_encoded: int
  A1Cresult_encoded: int
  change_encoded: int
  diabetesMed_encoded: int
  diag_1_category_encoded: int
  diag_2_category_encoded: int
  diag_3_category_encoded: int
  metformin_binary: int
  repaglinide_binary: int
  nateglinide_binary: int
  chlorpropamide_binary: int
  glimepiride_binary: int
  acetohexamide_binary: int
  glipizide_binary: int
  glyburide_binary: int
  tolbutamide_binary: int
  pioglitazone_binary: int
  rosiglitazone_binary: int
  acarbose_binary: int
  miglitol_binary: int
  troglitazone_binary: int
  tolazamide_binary: int
  examide_binary: int
  citoglipton_binary: int
  insulin_binary: int
  glyburide_metformin_binary: int
  glipizide_metformin_binary: int
  glimepiride_pioglitazone_binary: int
  metformin_rosiglitazone_binary: int
  metformin_pioglitazone_binary: int


model_data = None
feature_names: List[str] = []


def classify_risk(prob: float) -> str:
  if prob < 0.3:
    return "Low"
  if prob < 0.7:
    return "Medium"
  return "High"


def load_model() -> bool:
  global model_data, feature_names
  env = os.getenv("MODEL_PATH")
  candidates = [
    Path(env) if env else None,
    Path(__file__).resolve().parents[1] / "diabetic_model.pkl",
    Path(__file__).resolve().parents[1] / "models" / "diabetic_model.pkl",
  ]
  candidates = [p for p in candidates if p is not None]
  chosen = next((p for p in candidates if p.exists()), None)
  if not chosen:
    model_data = None
    feature_names = []
    return False
  with open(chosen, "rb") as f:
    model_data = pickle.load(f)
  feature_names = model_data["feature_names"]
  return True


@app.get("/health")
def health():
  loaded = model_data is not None or load_model()
  return {"status": "healthy" if loaded else "unhealthy", "model_loaded": bool(loaded)}


@app.get("/test")
def test():
  return {"message": "API is working"}


@app.get("/features")
def features():
  if not (model_data or load_model()):
    raise HTTPException(status_code=503, detail="Model not loaded")
  return {"features": feature_names}


@app.post("/predict")
def predict(p: PatientData):
  if not (model_data or load_model()):
    raise HTTPException(status_code=503, detail="Model not loaded")
  data = p.dict()
  vals = [data.get(feat, 0) for feat in feature_names]
  X = pd.DataFrame([vals], columns=feature_names)
  scaler = model_data["scaler"]
  clf = model_data["calibrated_model"]
  Xs = pd.DataFrame(scaler.transform(X), columns=X.columns)
  pred = int(clf.predict(Xs)[0])
  prob = float(clf.predict_proba(Xs)[0, 1])
  return {
    "prediction": pred,
    "probability": prob,
    "risk_level": classify_risk(prob),
    "model_version": f"{model_data.get('model_type','model')}_v1.0"
  }


if __name__ == "__main__":
  import uvicorn
  uvicorn.run("python.main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")))


