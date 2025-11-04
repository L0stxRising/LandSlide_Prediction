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
def predict(data: InputData):
    try:
        # Prepare input for ONNX
        input_name = sess.get_inputs()[0].name
        inputs = {input_name: [data.numbers]}
        
        # Run inference
        output = sess.run(None, inputs)
        prediction = output[0][0]
        pred=int(np.argmax(prediction))
        predclass=encList[pred]
        return {"prediction": (predclass)}
    except Exception as e:
        return {"error": str(e)}
