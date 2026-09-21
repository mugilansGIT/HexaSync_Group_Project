import requests
import streamlit as st

st.set_page_config(
    page_title="Music Listener Segmentation", page_icon="🎵", layout="centered"
)

st.title("🎵 Music Listener Persona Predictor")
st.write(
    "Input listener behavior metrics to identify their listener segment."
)

st.divider()

# Input controls
listening_hours = st.slider("Listening Hours per Week", 0, 50, 15)
songs_per_day = st.slider("Songs per Day", 0, 200, 60)
skip_rate = st.slider("Skip Rate (%)", 0, 100, 25)
playlist_count = st.slider("Playlist Count", 0, 50, 15)

if st.button("Predict Segment", type="primary"):
    payload = {
        "listening_hours_per_week": listening_hours,
        "songs_per_day": songs_per_day,
        "skip_rate": skip_rate,
        "playlist_count": playlist_count,
    }

    try:
        # Request prediction from API running on port 8001
        response = requests.post(
            "http://127.0.0.1:8001/predict_cluster", json=payload
        )

        if response.status_code == 200:
            result = response.json()
            segment = result["segment_name"]

            st.success(f"**Predicted Persona:** {segment}")

            if segment == "Power Listener":
                st.info(
                    "🔥 **High Engagement:** Consumes large volumes of music, keeps skip rates low, and curates extensive playlists."
                )
            elif segment == "Regular Listener":
                st.info(
                    "🎧 **Balanced User:** Displays moderate listening habits and steady daily playback."
                )
            else:
                st.info(
                    "⚡ **Casual / Skipper:** Low overall stream time and high track skip frequency."
                )
        else:
            st.error("Error from API backend. Please verify FastAPI status.")

    except Exception as e:
        st.error(
            f"Could not connect to FastAPI server on port 8001. Ensure `main.py` is running.\n\nDetails: {e}"
        )