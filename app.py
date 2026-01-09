from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import joblib
import numpy as np
import uvicorn

app = FastAPI(title="TCS Customer Churn API")

# HTML templates
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# Load model and scaler
model = joblib.load("models/churn_model.pkl")
scaler = joblib.load("models/scaler.pkl")


class Customer(BaseModel):
    tenure: float
    MonthlyCharges: float
    TotalCharges: float
    tenure_group: float = 2
    charges_ratio: float = 0.1
    gender: float = 0.0
    Partner: float = 0.0
    Dependents: float = 0.0
    PhoneService: float = 0.0
    InternetService: float = 0.0
    Contract: float = 0.0


@app.post("/predict")
def predict_churn(c: Customer):
    x = np.array([[c.tenure, c.MonthlyCharges, c.TotalCharges,
                   c.tenure_group, c.charges_ratio,
                   c.gender, c.Partner, c.Dependents,
                   c.PhoneService, c.InternetService, c.Contract]])
    x_scaled = scaler.transform(x)
    prob = model.predict_proba(x_scaled)[0, 1]
    risk = "High" if prob > 0.7 else "Medium" if prob > 0.4 else "Low"
    return {"churn_probability": float(prob), "risk_level": risk}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)