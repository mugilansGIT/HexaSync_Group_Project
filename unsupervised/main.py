import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Music Listener Segmentation API")

# Load model artifacts (adjust path if saved inside model_output/)
try:
    model = joblib.load("model_output/model.pkl")
    scaler = joblib.load("model_output/scaler.pkl")
except Exception:
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")

# Define segment names matching Srimathi's output
SEGMENT_MAP = {
    0: "Regular Listener",
    1: "Power Listener",
    2: "Casual Listener",
}


class ListenerInput(BaseModel):
    listening_hours_per_week: float
    songs_per_day: float
    skip_rate: float
    playlist_count: float


@app.get("/")
def home():
    return {
        "status": "online",
        "message": "Music Listener Segmentation API is running",
    }


@app.post("/predict_cluster")
def predict_cluster(data: ListenerInput):
    # Prepare input array
    features = np.array(
        [
            [
                data.listening_hours_per_week,
                data.songs_per_day,
                data.skip_rate,
                data.playlist_count,
            ]
        ]
    )

    # Scale features & predict cluster ID
    scaled_features = scaler.transform(features)
    cluster_id = int(model.predict(scaled_features)[0])
    segment_name = SEGMENT_MAP.get(cluster_id, f"Cluster {cluster_id}")

    return {
        "cluster_id": cluster_id,
        "segment_name": segment_name,
    }