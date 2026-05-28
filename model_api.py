
import os
import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Layer
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from scipy import stats as scipy_stats
from typing import List, Dict, Union

# --- Custom Keras Layers (MUST be defined for model loading) ---
class TemporalAttention(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight',
                                 shape=(input_shape[-1], 1),
                                 initializer='glorot_uniform',
                                 trainable=True)
        self.b = self.add_weight(name='attention_bias',
                                 shape=(input_shape[1], 1),
                                 initializer='zeros',
                                 trainable=True)
        super().build(input_shape)

    def call(self, x):
        e = tf.keras.activations.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
        alpha = tf.keras.activations.softmax(e, axis=1)
        context_vector = tf.reduce_sum(x * alpha, axis=1)
        return context_vector

class PharmaSixModel(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.lstm1 = LSTM(128, return_sequences=True, name="lstm_extractor_1")
        self.drop1 = Dropout(0.15)
        self.lstm2 = LSTM(64, return_sequences=True, name="lstm_extractor_2")
        self.attention = TemporalAttention(name="pharmacy_attention")
        self.dense_hidden = Dense(32, activation='relu')
        self.out_layer = Dense(1, name="forecast_output")

    def call(self, inputs, training=False):
        x = self.lstm1(inputs)
        x = self.drop1(x, training=training)
        x = self.lstm2(x)
        x = self.attention(x)
        x = self.dense_hidden(x)
        return self.out_layer(x)

    # train_step is not needed for inference, but included for completeness if training were part of the API
    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}


app = FastAPI()

# --- Global Variables & Model Loading ---
# Define file paths
DATA_FILE_PATH = '/content/data_feature_engineering.csv'
MODEL_FILE = '/content/pharmasix_v3_lstm.keras'
SCALERS_FILE = '/content/pharmasix_v3_scalers.gz'

# Load dataset, model, and scalers once at startup
try:
    df_raw = pd.read_csv(DATA_FILE_PATH)
    df_raw['date'] = pd.to_datetime(df_raw['date'])
    print(f"[API Startup] Successfully loaded {DATA_FILE_PATH}")
except Exception as e:
    print(f"[API Startup Error] Failed to load data_feature_engineering.csv: {e}")
    df_raw = None

try:
    model = load_model(
        MODEL_FILE,
        custom_objects={'TemporalAttention': TemporalAttention, 'PharmaSixModel': PharmaSixModel}
    )
    print(f"[API Startup] Successfully loaded {MODEL_FILE}")
except Exception as e:
    print(f"[API Startup Error] Failed to load model {MODEL_FILE}: {e}")
    model = None

try:
    scalers = joblib.load(SCALERS_FILE)
    print(f"[API Startup] Successfully loaded {SCALERS_FILE}")
except Exception as e:
    print(f"[API Startup Error] Failed to load scalers {SCALERS_FILE}: {e}")
    scalers = None

# --- Feature and Sequence Definitions (as used in training) ---
features = [
    'units_sold',
    'stock_level',
    'lag_1',
    'lag_2',
    'rolling_mean_2',
    'growth_rate',
    'stock_coverage_days',
    'expiry_days_remaining',
    'near_expiry',
]
target_col = 'units_sold'
target_idx = features.index(target_col)
sequence_length = 30

# --- Supply Chain Constants (as used in notebook) ---
SERVICE_LEVELS = {
    'kritis' : 0.99,
    'reguler': 0.95,
    'rendah' : 0.90,
}
z_scores = {k: scipy_stats.norm.ppf(v) for k, v in SERVICE_LEVELS.items()}
LEAD_TIME_MEAN  = 3
LEAD_TIME_STD   = 1

# --- Helper Functions (from notebook) ---
def klasifikasi_service_level(row: dict) -> str:
    if row.get('near_expiry', 0) == 1:
        return 'rendah'
    if row.get('stock_coverage_days', 99) < 3:
        return 'kritis'
    if row.get('expiry_days_remaining', 999) < 60:
        return 'kritis'
    return 'reguler'

def hitung_safety_stock_lstm(forecast_demand: float, forecast_std: float, service_level_key: str = 'reguler') -> dict:
    z = z_scores.get(service_level_key, z_scores['reguler'])
    safety_stock = z * np.sqrt(
        LEAD_TIME_MEAN * (forecast_std ** 2) +
        (forecast_demand ** 2) * (LEAD_TIME_STD ** 2)
    )
    rop = (forecast_demand * LEAD_TIME_MEAN) + safety_stock

    return {
        'lstm_forecast_demand' : round(float(forecast_demand), 2),
        'lstm_forecast_std'    : round(float(forecast_std), 2),
        'safety_stock'         : round(float(safety_stock), 0),
        'reorder_point'        : round(float(rop), 0),
    }

# --- Prediction Function for API ---
def predict_and_calculate_report_for_medicine(medicine_id: int) -> Dict[str, Union[int, float, str]]:
    if df_raw is None or model is None or scalers is None:
        raise HTTPException(status_code=500, detail="API backend not initialized. Model or data not loaded.")

    df_med = df_raw[df_raw['medicine'] == medicine_id].sort_values('date').copy()

    if len(df_med) < sequence_length:
        raise HTTPException(status_code=404, detail=f"Not enough historical data ({len(df_med)} rows) for medicine {medicine_id} to form a sequence of length {sequence_length}.")

    if medicine_id not in scalers:
        raise HTTPException(status_code=404, detail=f"Scaler not found for medicine {medicine_id}.")

    scaler = scalers[medicine_id]

    # Get the last 'sequence_length' rows for prediction
    last_sequence_raw = df_med[features].tail(sequence_length).values
    last_sequence_scaled = scaler.transform(last_sequence_raw)

    # Reshape for LSTM input: (1, sequence_length, num_features)
    X_pred = np.array([last_sequence_scaled])

    # Make prediction
    predicted_scaled = model.predict(X_pred, verbose=0)[0][0]

    # Inverse transform prediction
    dummy_pred = np.zeros((1, len(features)))
    dummy_pred[:, target_idx] = predicted_scaled
    predicted_actual = scaler.inverse_transform(dummy_pred)[:, target_idx][0]
    predicted_actual = max(0.0, predicted_actual) # Ensure non-negative

    # For forecast_std, use standard deviation of units_sold in the last sequence
    # This is a proxy for demand variability. A more sophisticated model would provide this directly.
    forecast_std = np.std(df_med['units_sold'].tail(sequence_length))
    if np.isnan(forecast_std): # Handle cases with constant demand or very few data points
        forecast_std = 0.0

    # Get last known stock information
    last_row = df_med.iloc[-1]
    stok_sekarang = int(last_row['stock_level'])

    # Determine Service Level
    sl_key = klasifikasi_service_level(last_row.to_dict())

    # Calculate Safety Stock and Reorder Point
    supply_chain_data = hitung_safety_stock_lstm(predicted_actual, forecast_std, sl_key)

    rop = supply_chain_data['reorder_point']
    ss = supply_chain_data['safety_stock']

    # Determine Alert Status
    if stok_sekarang <= 0:          alert = 'STOCKOUT'
    elif stok_sekarang < ss:        alert = 'KRITIS'
    elif stok_sekarang < rop:       alert = 'PERLU ORDER'
    else:                           alert = 'AMAN'

    # Calculate Order Recommendation
    order_qty = 0
    if alert in ['STOCKOUT', 'KRITIS', 'PERLU ORDER']:
        order_qty = (predicted_actual * 30) + ss - stok_sekarang # Cover 30 days + Safety Stock
        order_qty = max(0, int(order_qty))

    report = {
        'medicine': medicine_id,
        'service_level': sl_key,
        'status_alert': alert,
        'stock_sekarang': stok_sekarang,
        'stock_coverage_days': round(float(np.nan_to_num(last_row['stock_coverage_days'])), 1),
        'expiry_days_remaining': int(np.nan_to_num(last_row['expiry_days_remaining'])),
        'near_expiry': int(np.nan_to_num(last_row['near_expiry'])),
        'lstm_forecast_demand': supply_chain_data['lstm_forecast_demand'],
        'lstm_forecast_std': supply_chain_data['lstm_forecast_std'],
        'safety_stock': supply_chain_data['safety_stock'],
        'reorder_point': supply_chain_data['reorder_point'],
        'order_rekomendasi': order_qty,
    }
    return report

# --- API Endpoints ---
@app.get("/check-all", response_model=List[Dict[str, Union[int, float, str]]])
async def get_all_medicine_reports():
    if df_raw is None:
        raise HTTPException(status_code=500, detail="Data not loaded, cannot generate report.")

    all_medicine_reports = []
    unique_medicines = sorted(df_raw['medicine'].unique())

    for med_id in unique_medicines:
        try:
            report = predict_and_calculate_report_for_medicine(int(med_id))
            all_medicine_reports.append(report)
        except HTTPException as e:
            print(f"Warning: Skipped medicine {med_id} due to error: {e.detail}")
        except Exception as e:
            print(f"Error processing medicine {med_id}: {e}")

    return all_medicine_reports

@app.get("/report/{medicine_id}", response_model=Dict[str, Union[int, float, str]])
async def get_medicine_report(medicine_id: int):
    try:
        report = predict_and_calculate_report_for_medicine(medicine_id)
        return report
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")

@app.get("/")
async def read_root():
    return {"message": "PharmaSix v3 Forecasting API. Use /check-all or /report/{medicine_id} for reports."}
