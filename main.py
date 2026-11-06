"""FastAPI service for synthetic environmental monitoring demonstrations."""
from __future__ import annotations

import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
DB_PATH = Path(os.getenv("DATABASE_PATH", str(ROOT / "data" / "synthetic_demo.sqlite3")))
VERSION = "synthetic-demo-v1"
MIGRATION_PATH = ROOT / "migrations" / "002_prediction_notifications.sql"
MODEL_NAMES = ["flood", "fire", "water_anomaly", "pm25"]
FIELD_LABELS = {"rainfall_mm_1h": "Rainfall", "temperature_c": "Temperature", "humidity_pct": "Humidity",
                "wind_speed_kmh": "Wind speed", "vegetation_dryness_index": "Vegetation dryness",
                "water_level_m": "Water level", "water_level_rise_m_per_h": "Water rise rate", "pm10_ug_m3": "PM10",
                "water_ph": "Water pH", "water_turbidity_ntu": "Turbidity", "dissolved_oxygen_mg_l": "Dissolved oxygen"}
UNITS = {"rainfall_mm_1h": "mm/h", "temperature_c": "°C", "humidity_pct": "%", "wind_speed_kmh": "km/h",
         "vegetation_dryness_index": "index", "water_level_m": "m", "water_level_rise_m_per_h": "m/h",
         "pm10_ug_m3": "µg/m³", "water_ph": "pH", "water_turbidity_ntu": "NTU", "dissolved_oxygen_mg_l": "mg/L"}
DATA_PATH = ROOT / "environmental_monitoring_synthetic_dataset.csv"
DATA: pd.DataFrame | None = None
MODELS: dict[str, dict[str, Any]] = {}
app = FastAPI(title="Environmental Monitoring Demo API", version=VERSION,
              description="Synthetic data demonstration; not validated for operational decisions.")
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as con:
        con.execute("""CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT, event_type TEXT NOT NULL,
            timestamp TEXT NOT NULL, zone_type TEXT NOT NULL,
            measurements_json TEXT NOT NULL, result_json TEXT NOT NULL,
            data_mode TEXT NOT NULL CHECK(data_mode IN ('synthetic_demo','live')),
            event_status TEXT NOT NULL DEFAULT 'unread'
        )""")
        columns = {row[1] for row in con.execute("PRAGMA table_info(events)")}
        if "event_status" not in columns:
            con.execute("ALTER TABLE events ADD COLUMN event_status TEXT NOT NULL DEFAULT 'unread'")
        con.executescript(MIGRATION_PATH.read_text(encoding="utf-8"))
        con.execute("INSERT OR IGNORE INTO notification_preferences(preference_id) VALUES(1)")
        # Email delivery has been rolled back. Keep historical delivery rows, but
        # disable any previously saved preference without deleting user data.
        con.execute("UPDATE notification_preferences SET email_enabled=0, synthetic_email_enabled=0 WHERE preference_id=1")
        con.execute("UPDATE email_delivery_logs SET status='failed',last_error='Email notifications were rolled back.',updated_at=? WHERE status IN ('pending','sending')", (datetime.now(timezone.utc).isoformat(),))


def persist(event_type: str, timestamp: str, zone_type: str, measurements: dict, result: dict):
    with sqlite3.connect(DB_PATH) as con:
        con.execute("INSERT INTO events(event_type,timestamp,zone_type,measurements_json,result_json,data_mode) VALUES(?,?,?,?,?,'synthetic_demo')",
                    (event_type, timestamp, zone_type, json.dumps(measurements), json.dumps(result)))


@app.on_event("startup")
def startup():
    global DATA
    init_db()
    if DATA_PATH.exists():
        DATA = pd.read_csv(DATA_PATH)
        DATA["timestamp"] = pd.to_datetime(DATA["timestamp"], utc=True, errors="coerce")
        DATA = DATA.dropna(subset=["timestamp"]).sort_values("timestamp")
    MODELS.clear()
    for name in MODEL_NAMES:
        path = MODEL_DIR / f"{name}.joblib"
        if path.exists():
            MODELS[name] = joblib.load(path)
    migrate_legacy_predictions()


class SensorReading(BaseModel):
    model_config = ConfigDict(extra="forbid")
    timestamp: datetime | None = None
    zone_type: str = Field(min_length=1, max_length=64)
    rainfall_mm_1h: float = Field(ge=0, le=500)
    temperature_c: float = Field(ge=-60, le=65)
    humidity_pct: float = Field(ge=0, le=100)
    wind_speed_kmh: float = Field(ge=0, le=400)
    vegetation_dryness_index: float = Field(ge=0, le=100)
    water_level_m: float = Field(ge=-10, le=30)
    water_level_rise_m_per_h: float = Field(ge=-20, le=20)
    pm10_ug_m3: float = Field(ge=0, le=5000)
    water_ph: float = Field(ge=0, le=14)
    water_turbidity_ntu: float = Field(ge=0, le=10000)
    dissolved_oxygen_mg_l: float = Field(ge=0, le=30)

    def feature_row(self):
        values = self.model_dump()
        return pd.DataFrame([{**values, "timestamp": None, "pm25_ug_m3": 0.0}])

    def stored_timestamp(self):
        return (self.timestamp or datetime.now(timezone.utc)).isoformat()


class ModelStatus(BaseModel):
    loaded: bool


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_version: str
    data_mode: Literal["synthetic_demo"]
    synthetic_data_notice: str
    models: dict[str, ModelStatus]


class RiskResult(BaseModel):
    category: Literal["Low", "Moderate", "High"]
    risk_score: float = Field(ge=0, le=1)


class PredictionResponse(BaseModel):
    event_id: str
    model_version: str
    prediction_timestamp: datetime
    data_mode: Literal["synthetic_demo"]
    notification_processing_status: str
    flood: RiskResult
    fire: RiskResult
    pm25_forecast_ug_m3: float
    anomaly: "AnomalyResult"
    supporting_measurements: dict[str, Any]


class AnomalyResult(BaseModel):
    is_anomaly: bool
    category: Literal["Anomaly", "Normal"]
    anomaly_score: float


class AnomalyResponse(BaseModel):
    is_anomaly: bool
    category: Literal["Anomaly", "Normal"]
    anomaly_score: float
    model_version: str
    prediction_timestamp: datetime
    data_mode: Literal["synthetic_demo"]
    features: dict[str, float]


class InAppPreferencesRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    in_app_enabled: bool = True
    hazard_types: list[Literal["flood", "fire", "water_quality"]] = Field(default_factory=lambda: ["flood", "fire", "water_quality"])
    severities: list[Literal["Low", "Moderate", "High"]] = Field(default_factory=lambda: ["High"])


def get_model(name: str):
    if name not in MODELS:
        raise HTTPException(status_code=503, detail=f"Model '{name}' is unavailable. Run `python train_models.py` and restart the API.")
    return MODELS[name]


def risk_category(probability: float) -> str:
    return "High" if probability >= 0.66 else "Moderate" if probability >= 0.33 else "Low"


def positive_probability(pipeline, features) -> float:
    classes = list(pipeline.classes_)
    if 1 not in classes:
        return 0.0
    return float(pipeline.predict_proba(features)[0][classes.index(1)])


def migrate_legacy_predictions():
    """Expose existing prediction rows to the shared stream without re-notifying them."""
    with sqlite3.connect(DB_PATH) as con:
        rows = con.execute("SELECT id,timestamp,zone_type,measurements_json,result_json FROM events WHERE event_type='prediction' AND data_mode='synthetic_demo'").fetchall()
        for row_id, timestamp, zone, measurements_json, result_json in rows:
            event_id = f"legacy-{row_id}"
            measurements, result = json.loads(measurements_json), json.loads(result_json)
            result.setdefault("event_id", event_id)
            result.setdefault("notification_processing_status", "legacy")
            con.execute("""INSERT OR IGNORE INTO prediction_events(event_id,hazard_type,source,zone_type,measurements_json,prediction_json,
                         model_names_json,model_version,prediction_timestamp,data_quality_status,notification_processing_status,created_at)
                         VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (event_id, "environmental", "synthetic_demo", zone, json.dumps(measurements), json.dumps(result),
                         json.dumps(["RandomForestClassifier:flood", "RandomForestClassifier:fire", "RandomForestRegressor:pm25"]),
                         result.get("model_version", VERSION), result.get("prediction_timestamp", timestamp),
                         "legacy_synthetic_record", "legacy", timestamp))


def notification_preferences(con: sqlite3.Connection) -> dict[str, Any]:
    row = con.execute("SELECT * FROM notification_preferences WHERE preference_id=1").fetchone()
    return {"in_app_enabled": bool(row["in_app_enabled"]), "hazard_types": json.loads(row["hazard_types_json"]),
            "severities": json.loads(row["severities_json"])}


def save_prediction_event(idempotency_key: str | None, measurements: dict[str, Any], result: dict[str, Any]):
    hazards = [("flood", result["flood"]["category"], result["flood"]["risk_score"]),
               ("fire", result["fire"]["category"], result["fire"]["risk_score"]),
               ("water_quality", "High" if result["anomaly"]["is_anomaly"] else "Low", result["anomaly"]["anomaly_score"])]
    event_id = str(uuid.uuid4())
    with sqlite3.connect(DB_PATH, timeout=10) as con:
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA foreign_keys=ON")
        con.execute("BEGIN IMMEDIATE")
        if idempotency_key:
            existing = con.execute("SELECT prediction_json FROM prediction_events WHERE idempotency_key=?", (idempotency_key,)).fetchone()
            if existing:
                return json.loads(existing["prediction_json"]), False, False
        prefs = notification_preferences(con)
        source = "synthetic_demo"
        now = result["prediction_timestamp"]
        con.execute("""INSERT INTO prediction_events(event_id,idempotency_key,hazard_type,source,zone_type,measurements_json,
                     prediction_json,model_names_json,model_version,prediction_timestamp,data_quality_status,notification_processing_status)
                     VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (event_id, idempotency_key, "environmental", source, measurements["zone_type"], json.dumps(measurements),
                     json.dumps({**result, "event_id": event_id}), json.dumps(["RandomForestClassifier:flood", "RandomForestClassifier:fire",
                     "IsolationForest:water_quality", "RandomForestRegressor:pm25"]), VERSION, now, "synthetic_demo_unvalidated", "processing"))
        for hazard, severity, score in hazards:
            qualifies = hazard in prefs["hazard_types"] and severity in prefs["severities"]
            if not qualifies:
                continue
            title = f"{hazard.replace('_', ' ').title()} risk · {severity}"
            message = f"{measurements['zone_type']} · {fmt_iso(now)} · Synthetic demo prediction"
            if prefs["in_app_enabled"]:
                con.execute("""INSERT OR IGNORE INTO in_app_notifications(event_id,hazard_type,severity,title,message,source,zone_type)
                             VALUES(?,?,?,?,?,?,?)""", (event_id, hazard, severity, title, message, source, measurements["zone_type"]))
        status = "processed"
        final_result = {**result, "event_id": event_id, "notification_processing_status": status}
        con.execute("UPDATE prediction_events SET prediction_json=?,notification_processing_status=? WHERE event_id=?",
                    (json.dumps(final_result), status, event_id))
        con.commit()
    return final_result, True, False


def fmt_iso(value: str) -> str:
    try:
        return datetime.fromisoformat(value).astimezone().strftime("%Y-%m-%d %H:%M %Z")
    except (TypeError, ValueError):
        return value


@app.get("/")
def dashboard():
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/api/health", response_model=HealthResponse)
def health():
    return {"status": "ok" if len(MODELS) == len(MODEL_NAMES) else "degraded",
            "model_version": VERSION, "data_mode": "synthetic_demo",
            "synthetic_data_notice": "Synthetic-data demonstration only; no live sensors are connected or validated.",
            "models": {name: {"loaded": name in MODELS} for name in MODEL_NAMES}}


@app.post("/api/predict", response_model=PredictionResponse)
def predict(reading: SensorReading, idempotency_key: str | None = Header(default=None, alias="Idempotency-Key")):
    if idempotency_key:
        with sqlite3.connect(DB_PATH) as con:
            row = con.execute("SELECT prediction_json FROM prediction_events WHERE idempotency_key=?", (idempotency_key,)).fetchone()
        if row:
            return json.loads(row[0])
    row = reading.feature_row()
    try:
        flood, fire, pm = (get_model(n) for n in ("flood", "fire", "pm25"))
        water = get_model("water_anomaly")
        flood_pipe, fire_pipe, pm_pipe = flood["pipeline"], fire["pipeline"], pm["pipeline"]
        flood_input = row[flood["features"]]
        fire_input = row[fire["features"]]
        pm_input = row[pm["features"]]
        flood_prob = positive_probability(flood_pipe, flood_input)
        fire_prob = positive_probability(fire_pipe, fire_input)
        pm_forecast = float(pm_pipe.predict(pm_input)[0])
        water_input = row[water["features"]]
        is_anomaly = bool(water["pipeline"].predict(water_input)[0] == -1)
        anomaly_score = float(-water["pipeline"].decision_function(water_input)[0])
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc
    predicted_at = datetime.now(timezone.utc).isoformat()
    measurements = reading.model_dump(mode="json")
    measurements["timestamp"] = reading.stored_timestamp()
    result = {"model_version": VERSION, "prediction_timestamp": predicted_at, "data_mode": "synthetic_demo",
              "flood": {"category": risk_category(flood_prob), "risk_score": flood_prob},
              "fire": {"category": risk_category(fire_prob), "risk_score": fire_prob},
              "anomaly": {"is_anomaly": is_anomaly, "category": "Anomaly" if is_anomaly else "Normal", "anomaly_score": anomaly_score},
              "pm25_forecast_ug_m3": pm_forecast, "supporting_measurements": measurements}
    try:
        result, created, _notifications_created = save_prediction_event(idempotency_key, measurements, result)
        if created:
            persist("prediction", reading.stored_timestamp(), reading.zone_type, measurements, result)
            for hazard, obj in (("flood", result["flood"]), ("fire", result["fire"])):
                if obj["category"] == "High":
                    persist("alert", reading.stored_timestamp(), reading.zone_type, measurements,
                            {"hazard": hazard, "category": obj["category"], "risk_score": obj["risk_score"]})
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction succeeded but database persistence failed: {exc}") from exc
    return result


@app.post("/api/anomaly", response_model=AnomalyResponse)
def detect_anomaly(reading: SensorReading):
    bundle = get_model("water_anomaly")
    try:
        prediction = int(bundle["pipeline"].predict(reading.feature_row()[bundle["features"]])[0] == -1)
        score = float(-bundle["pipeline"].decision_function(reading.feature_row()[bundle["features"]])[0])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Anomaly detection failed: {exc}") from exc
    result = {"is_anomaly": bool(prediction), "category": "Anomaly" if prediction else "Normal",
              "anomaly_score": score, "model_version": VERSION,
              "prediction_timestamp": datetime.now(timezone.utc).isoformat(), "data_mode": "synthetic_demo",
              "features": {key: getattr(reading, key) for key in ("water_ph", "water_turbidity_ntu", "dissolved_oxygen_mg_l")}}
    try:
        persist("anomaly", reading.stored_timestamp(), reading.zone_type,
                {**reading.model_dump(mode="json"), "timestamp": reading.stored_timestamp()}, result)
        if prediction:
            persist("alert", reading.stored_timestamp(), reading.zone_type, result["features"], result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Anomaly result generated but database persistence failed: {exc}") from exc
    return result


@app.get("/api/prediction-events")
def prediction_events(limit: int = 30):
    if not 1 <= limit <= 200:
        raise HTTPException(status_code=422, detail="limit must be between 1 and 200")
    with sqlite3.connect(DB_PATH) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute("SELECT * FROM prediction_events ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
    return [{"event_id": r["event_id"], "source": r["source"], "zone_type": r["zone_type"],
             "measurements": json.loads(r["measurements_json"]), "prediction": json.loads(r["prediction_json"]),
             "model_names": json.loads(r["model_names_json"]), "model_version": r["model_version"],
             "prediction_timestamp": r["prediction_timestamp"], "data_quality_status": r["data_quality_status"],
             "notification_processing_status": r["notification_processing_status"]} for r in rows]


@app.get("/api/notifications")
def list_notifications(limit: int = 20):
    if not 1 <= limit <= 100:
        raise HTTPException(status_code=422, detail="limit must be between 1 and 100")
    with sqlite3.connect(DB_PATH) as con:
        con.row_factory = sqlite3.Row
        unread = con.execute("SELECT COUNT(*) FROM in_app_notifications WHERE read_at IS NULL").fetchone()[0]
        rows = con.execute("SELECT * FROM in_app_notifications ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
    return {"unread_count": unread, "notifications": [dict(r) for r in rows]}


@app.post("/api/notifications/{notification_id}/read")
def mark_notification_read(notification_id: int):
    with sqlite3.connect(DB_PATH) as con:
        cur = con.execute("UPDATE in_app_notifications SET read_at=? WHERE notification_id=? AND read_at IS NULL",
                          (datetime.now(timezone.utc).isoformat(), notification_id))
        if not cur.rowcount:
            exists = con.execute("SELECT 1 FROM in_app_notifications WHERE notification_id=?", (notification_id,)).fetchone()
            if not exists:
                raise HTTPException(status_code=404, detail="Notification not found")
    return {"notification_id": notification_id, "read": True}


@app.post("/api/notifications/read-all")
def mark_all_notifications_read():
    with sqlite3.connect(DB_PATH) as con:
        cur = con.execute("UPDATE in_app_notifications SET read_at=? WHERE read_at IS NULL", (datetime.now(timezone.utc).isoformat(),))
    return {"updated": cur.rowcount}


@app.get("/api/notification-settings")
def get_notification_settings():
    with sqlite3.connect(DB_PATH) as con:
        con.row_factory = sqlite3.Row
        return notification_preferences(con)


@app.post("/api/notification-settings")
def update_notification_settings(settings: InAppPreferencesRequest):
    with sqlite3.connect(DB_PATH) as con:
        cur = con.execute("""UPDATE notification_preferences SET in_app_enabled=?,hazard_types_json=?,severities_json=?,updated_at=?
                             WHERE preference_id=1""",
                          (int(settings.in_app_enabled), json.dumps(settings.hazard_types), json.dumps(settings.severities),
                           datetime.now(timezone.utc).isoformat()))
        if cur.rowcount != 1:
            raise HTTPException(status_code=500, detail="Notification preferences are unavailable.")
    return {"saved": True, **settings.model_dump()}


@app.get("/api/history")
def history(limit: int = 50):
    if not 1 <= limit <= 200:
        raise HTTPException(status_code=422, detail="limit must be between 1 and 200")
    with sqlite3.connect(DB_PATH) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute("SELECT * FROM events WHERE data_mode='synthetic_demo' ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [{**dict(row), "measurements": json.loads(row["measurements_json"]),
             "result": json.loads(row["result_json"])} for row in rows]


@app.get("/api/overview")
def overview():
    if DATA is None:
        raise HTTPException(status_code=503, detail="Synthetic dataset is unavailable")
    with sqlite3.connect(DB_PATH) as con:
        event_counts = dict(con.execute("SELECT event_type, COUNT(*) FROM events WHERE data_mode='synthetic_demo' GROUP BY event_type").fetchall())
        unread_alerts = con.execute("SELECT COUNT(*) FROM events WHERE data_mode='synthetic_demo' AND event_type='alert' AND event_status='unread'").fetchone()[0]
    zones = {str(k): int(v) for k, v in DATA["zone_type"].value_counts().items()}
    labels = {"flood": int(DATA["flood_risk_label"].sum()),
              "fire": int(DATA["fire_risk_label"].sum()),
              "water_quality": int(DATA["water_quality_anomaly_label"].sum()),
              "air_quality": None}
    latest = DATA.iloc[-1]
    return {"data_mode": "synthetic_demo", "dataset_rows": len(DATA),
            "dataset_start": DATA["timestamp"].min().isoformat(), "dataset_end": DATA["timestamp"].max().isoformat(),
            "latest_reading": {"timestamp": latest["timestamp"].isoformat(), "zone_type": latest["zone_type"],
                               **{key: float(latest[key]) for key in ["temperature_c", "rainfall_mm_1h", "water_level_m", "pm25_ug_m3", "pm10_ug_m3", "water_ph", "water_turbidity_ntu", "dissolved_oxygen_mg_l"]}},
            "zones": zones, "synthetic_label_counts": labels,
            "sensors": {"registered": None, "online": None, "reason": "The dataset has no sensor identifiers or connectivity telemetry."},
            "events": {"stored_predictions": event_counts.get("prediction", 0), "stored_anomaly_checks": event_counts.get("anomaly", 0),
                       "stored_alerts": event_counts.get("alert", 0), "unread_alerts": unread_alerts},
            "models": {name: name in MODELS for name in MODEL_NAMES},
            "source_status": "available"}


@app.get("/api/trends")
def trends(metric: str = "temperature_c", period: str = "7d"):
    if DATA is None:
        raise HTTPException(status_code=503, detail="Synthetic dataset is unavailable")
    allowed = {"temperature_c", "rainfall_mm_1h", "water_level_m", "pm25_ug_m3", "pm10_ug_m3",
               "water_ph", "water_turbidity_ntu", "dissolved_oxygen_mg_l"}
    windows = {"24h": pd.Timedelta(hours=24), "7d": pd.Timedelta(days=7), "30d": pd.Timedelta(days=30)}
    if metric not in allowed:
        raise HTTPException(status_code=422, detail=f"metric must be one of {sorted(allowed)}")
    if period not in windows:
        raise HTTPException(status_code=422, detail="period must be 24h, 7d, or 30d")
    cutoff = DATA["timestamp"].max() - windows[period]
    subset = DATA.loc[DATA["timestamp"] >= cutoff, ["timestamp", metric]].copy()
    if not subset.empty:
        subset["hour"] = subset["timestamp"].dt.floor("h")
        subset = subset.groupby("hour", as_index=False)[metric].mean()
        if len(subset) > 100:
            subset = subset.iloc[:: max(1, len(subset) // 100)]
    return {"metric": metric, "period": period, "data_mode": "synthetic_demo",
            "window_end": DATA["timestamp"].max().isoformat(),
            "points": [{"timestamp": row["hour"].isoformat(), "value": float(row[metric])} for _, row in subset.iterrows()]}


@app.post("/api/alerts/{event_id}/{action}")
def update_alert(event_id: int, action: Literal["acknowledge", "resolve"]):
    status = "acknowledged" if action == "acknowledge" else "resolved"
    with sqlite3.connect(DB_PATH) as con:
        cur = con.execute("UPDATE events SET event_status=? WHERE id=? AND event_type='alert' AND data_mode='synthetic_demo'", (status, event_id))
        if cur.rowcount == 0:
            raise HTTPException(status_code=404, detail="Synthetic alert not found")
    return {"id": event_id, "status": status, "data_mode": "synthetic_demo"}
