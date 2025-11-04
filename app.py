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
        # model expects 1D, so send np.array directly
        input_data = np.array(data.numbers, dtype=np.float32)

        # run inference
        output = sess.run(None, {input_name: input_data})
        prediction = output[0]

        pred = int(np.argmax(prediction))
        predclass = encList[pred]

        return {"prediction": predclass}
    except Exception as e:
        return {"error": str(e)}

