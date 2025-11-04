from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import onnxruntime as rt
import numpy as np

# === Initialize FastAPI app ===
app = FastAPI(title="Landslide Prediction API")

# === Enable CORS (so frontend can fetch data) ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all origins (you can restrict later)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Load ONNX model ===
sess = rt.InferenceSession("model.onnx")

# === Input Schema ===
class InputData(BaseModel):
    numbers: list[float]

@app.get("/")
def root():
    return {"message": "Landslide Prediction API is running!"}

encList = ["Low", "Moderate", "High", "Very High"]

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
