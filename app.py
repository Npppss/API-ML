from fastapi import FastAPI, Request
import pickle
import numpy as np
import uvicorn

app = FastAPI()

# Load model saat startup
with open("transformer_recommender.pkl", "rb") as f:
    model = pickle.load(f)

@app.get("/")
def root():
    return {"message": "Recommender API Ready"}

@app.post("/recommend")
async def recommend(data: dict):
    user_input = np.array(data["features"]).reshape(1, -1)
    prediction = model.predict(user_input)
    return {"recommendation": prediction.tolist()}
