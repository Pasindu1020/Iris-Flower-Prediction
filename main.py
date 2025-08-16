from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import numpy as np
import joblib, json

app = FastAPI(title="Iris Classifier API with UI")

MODEL_PATH = "model.pkl"
META_PATH = "model_meta.json"

templates = Jinja2Templates(directory="templates")

model = None
meta = None

@app.on_event("startup")
def load_artifacts():
    global model, meta
    model = joblib.load(MODEL_PATH)
    with open(META_PATH, "r") as f:
        meta = json.load(f)

def _predict(features: np.ndarray):
    proba = model.predict_proba(features)[0]
    pred_idx = int(np.argmax(proba))
    classes = meta["target_names"]
    prediction = classes[pred_idx]
    confidence = float(np.max(proba))
    return prediction, confidence

@app.get("/ui", response_class=HTMLResponse)
def iris_form(request: Request):
    return templates.TemplateResponse("iris_form.html", {"request": request})

@app.post("/ui/predict", response_class=HTMLResponse)
def iris_form_predict(
    request: Request,
    sepal_length: float = Form(...),
    sepal_width: float = Form(...),
    petal_length: float = Form(...),
    petal_width: float = Form(...),
):
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    proba = model.predict_proba(features)[0]
    pred_idx = int(np.argmax(proba))
    classes = meta["target_names"]
    prediction = classes[pred_idx]
    confidence = float(np.max(proba))
    probabilities = {classes[i]: float(proba[i]) for i in range(len(classes))}

    return templates.TemplateResponse(
        "iris_result.html",
        {
            "request": request,
            "prediction": prediction,
            "confidence": f"{confidence:.2f}",
            "probabilities": probabilities   
        }
    )

