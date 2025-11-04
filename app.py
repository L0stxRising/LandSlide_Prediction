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
# If your index.html and assets are inside a folder named "static"
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

class InputData(BaseModel):
    numbers: list[float]

@app.post("/predict")
def predict(data: InputData):
    try:
        input_name = sess.get_inputs()[0].name
        inputs = {input_name: [data.numbers]}
        output = sess.run(None, inputs)
        prediction = output[0][0]
        pred_index = int(np.argmax(prediction))
        pred_class = encList[pred_index]
        return {
            "prediction": pred_class,
            "raw_output": prediction.tolist(),
            "index": pred_index
        }
    except Exception as e:
        return {"error": str(e)}
