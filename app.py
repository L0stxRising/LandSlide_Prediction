from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import onnxruntime as rt
import numpy as np
import os

# === Initialize app ===

app = FastAPI(title="Landslide Prediction API")

# === Enable CORS ===

app.add_middleware(
CORSMiddleware,
allow_origins=["*"],
allow_credentials=True,
allow_methods=["*"],
allow_headers=["*"],
)

# === Serve static files ===

if not os.path.exists("static"):
    os.makedirs("static")

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def serve_home():
"""Serve the main website page"""
index_path = os.path.join("static", "index.html")
if os.path.exists(index_path):
return FileResponse(index_path)
return {"error": "index.html not found"}

# === ONNX Model ===

sess = rt.InferenceSession("model.onnx")
encList = ["Low", "Moderate", "High", "Very High"]

# === Input data schema ===

class InputData(BaseModel):
numbers: list[float]  # Now expects 7 values: temp, humidity, precipitation, soil moisture, elevation, slope, vegetation

@app.post("/predict")
def predict(data: InputData):
try:
if len(data.numbers) != 7:
return {"error": "Input must contain exactly 7 values: temp, humidity, precipitation, soil_moisture, elevation, slope, vegetation"}

```
    # Model inputs (first 5 values only)
    model_input = np.array(data.numbers[:5], dtype=np.float32)
    input_name = sess.get_inputs()[0].name
    output = sess.run(None, {input_name: model_input})
    prediction = output[0]
    pred_idx = int(np.argmax(prediction))
    predclass = encList[pred_idx]

    # === Post-processing adjustments ===
    slope = data.numbers[5]
    vegetation = data.numbers[6]

    # Adjust based on slope
    if slope > 30 and predclass != "Very High":
        idx = encList.index(predclass)
        predclass = encList[min(idx + 1, len(encList) - 1)]
    elif slope < 15 and predclass != "Low":
        idx = encList.index(predclass)
        predclass = encList[max(idx - 1, 0)]

    # Adjust based on vegetation
    if vegetation < 30 and predclass != "Very High":
        idx = encList.index(predclass)
        predclass = encList[min(idx + 1, len(encList) - 1)]
    elif vegetation > 70 and predclass != "Low":
        idx = encList.index(predclass)
        predclass = encList[max(idx - 1, 0)]

    return {"prediction": predclass}

except Exception as e:
    return {"error": str(e)}
```

