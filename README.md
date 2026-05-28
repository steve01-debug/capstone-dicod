---
title: Capstone Dicod
emoji: 📦
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.35.0
app_file: main.py
pinned: false
---

---
layout: "model"
base_model: "None"
model_name: "PharmaSix-LSTM-Forecasting"

--- # PharmaSix v3 LSTM Forecasting API

This Hugging Face Space hosts a FastAPI application for pharmaceutical inventory forecasting using an LSTM model. It provides predictions for units sold, calculates safety stock, reorder points, and generates procurement recommendations based on defined service levels.

## API Endpoints

*   `/`: Returns a welcome message.
*   `/check-all`: Returns a list of reports for all medicines.
*   `/report/{medicine_id}`: Returns a report for a specific medicine ID.

## How to use (for developers)

The API is built using FastAPI. You can interact with it programmatically using standard HTTP requests.

Example using `curl`:

```bash
curl -X GET "[YOUR_SPACE_URL]/check-all" \
     -H "accept: application/json"
```

## Deployment Details

This Space is configured to run a FastAPI application. The `Dockerfile` handles the environment setup and runs `uvicorn` to serve the API.

**Note:**
*   The `data_feature_engineering.csv`, `pharmasix_v3_lstm.keras`, and `pharmasix_v3_scalers.gz` files must be present in the same directory as `model_api.py` for the API to function correctly.
*   The chatbot functionality (if enabled) relies on `GOOGLE_API_KEY` which should be configured as a secret in your Hugging Face Space settings if used.
