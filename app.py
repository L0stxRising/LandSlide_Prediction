from fastapi import FastAPI
from pydantic import BaseModel
import onnxruntime as rt
import numpy as np  # <-- add this

# Initialize FastAPI app
app = FastAPI(title="Number Prediction API")

# Load ONNX model
sess = rt.InferenceSession("model.onnx")

# Input schema
class InputData(BaseModel):
    numbers: list[float]

@app.get("/")
def root():
    return {"message": "ONNX Prediction API is running!"}
encList=["Low","Moderate","High","Very High"]
@app.post("/predict")
@app.post("/predict")
def predict(data: InputData):
    try:
        input_name = sess.get_inputs()[0].name
        # Send as 1D array
        inputs = {input_name: np.array(data.numbers, dtype=np.float32)}
        
        output = sess.run(None, inputs)
        preds = np.array(output[0])
        
        pred_idx = int(np.argmax(preds))
        pred_class = encList[pred_idx] if pred_idx < len(encList) else f"Class {pred_idx}"

        return {
            "predicted_class": pred_class,
            "index": pred_idx,
            "raw_output": preds.tolist()
        }

    except Exception as e:
        return {"error": str(e)}
