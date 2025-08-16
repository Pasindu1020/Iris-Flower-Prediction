
# Iris Classifier - FastAPI Inference Service

This project trains a simple ML model on the Iris dataset and serves it via a FastAPI API.

## 1) Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2) Train the model

**Option A (recommended): use scikit-learn's built-in Iris dataset**
```bash
python train_model.py --data-source sklearn --model-path model.pkl --meta-path model_meta.json
```

**Option B: load from Kaggle (uciml/iris)**  
(requires: `pip install kagglehub[pandas-datasets]`)
```bash
python train_model.py --data-source kaggle --kaggle-file Iris.csv --model-path model.pkl --meta-path model_meta.json
```

## 3) Run the API

```bash
uvicorn main:app --reload
# Visit http://127.0.0.1:8000/docs
```

## 4) Example Requests

**Single prediction**
```bash
curl -X POST "http://127.0.0.1:8000/predict"       -H "Content-Type: application/json"       -d '{
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
  }'
```

**Batch prediction (bonus)**
```bash
curl -X POST "http://127.0.0.1:8000/predict-batch"       -H "Content-Type: application/json"       -d '{
    "items": [
      {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},
    {"sepal_length": 6.7, "sepal_width": 3.0, "petal_length": 5.0, "petal_width": 1.7}
    ]
  }'
```

**Model info**
```bash
curl "http://127.0.0.1:8000/model-info"
```

## Notes

- The API validates inputs using Pydantic.
- `model.pkl` and `model_meta.json` are loaded at startup.
- The response includes predicted class, confidence, and per-class probabilities.
- Keep dependencies minimal; KaggleHub is optional.
