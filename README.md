# Fieldwatch Environmental Monitoring Network

This workspace is a synthetic-data demonstration built from `environmental_monitoring_synthetic_dataset.csv`. It has no connected live sensors. Model evaluation on synthetic data can be unrealistically optimistic and is not evidence of real-world performance or suitable for operational decisions.

## Run in Replit

1. Import this folder into Replit and confirm the CSV is present at the project root.
2. Replit installs packages from `requirements.txt` (or run `pip install -r requirements.txt` in the Shell).
3. Train the models: `python train_models.py`. This validates the CSV and writes model pipelines and `training_report.json` to `models/`.
4. Start the application with `uvicorn main:app --host 0.0.0.0 --port 8000`, or use the configured Run button.
5. Open the web preview. The dashboard calls the FastAPI service on the same origin. API documentation is at `/docs`.

The SQLite database defaults to `data/synthetic_demo.sqlite3`. Set `DATABASE_PATH` in Replit Secrets only if you need a different server-side path. The browser never receives database credentials or secrets. Synthetic events are tagged `synthetic_demo`; future live events must be ingested into a separately authorized live-data workflow rather than labeled live by the demonstration API.

## API

- `GET /api/health`: service and model load status.
- `POST /api/predict`: flood and fire risk probabilities/categories and PM2.5 regression forecast; persists reading, prediction and high-risk alerts.
- `POST /api/anomaly`: Isolation Forest result using pH, turbidity and dissolved oxygen; persists reading/result and anomaly alerts. The synthetic anomaly label is not accepted by the API or used to fit the model.
- `GET /api/history?limit=50`: recent synthetic demonstration events.
- `GET /api/overview`: CSV-derived row counts, date range, zones, label counts, model status, and database event totals.
- `GET /api/trends?metric=temperature_c&period=7d`: hourly aggregates from the CSV for supported measured fields and 24h/7d/30d windows.
- `POST /api/alerts/{id}/acknowledge` and `/resolve`: update the status of an existing synthetic alert record.

The frontend is served directly by FastAPI from `static/` and uses same-origin API calls; it has no separate Node build step. Navigation includes the overview, model prediction studio, hazard views, source and history views, alerts, and a geographic coverage view. A geographic map and device online counts remain unavailable until coordinates and sensor/device telemetry are provided.

## Prediction synchronization and notifications

The Risk Predictor and Synthetic Lab use the same `POST /api/predict` model result. Each request creates one persistent `prediction_events` row; the Synthetic Lab uses an idempotency key and the Risk Predictor polls the shared event/notification feed every 8 seconds. Existing legacy prediction rows are copied into the new event stream without re-triggering notifications. In-app notifications persist in SQLite and support read/mark-all-read. Email settings and delivery have been rolled back; historical email log records remain in the database, and pending jobs are disabled at startup.

Notification preferences control in-app delivery by hazard type and severity. No email provider credentials or notification admin token are needed. SQLite migration `migrations/002_prediction_notifications.sql` adds prediction events, persistent in-app notifications, preferences, and a legacy email-log table retained for data preservation. The existing `events` table and model endpoints remain in place. This project does not currently include user authentication; protect the Replit deployment with an appropriate access layer before exposing the application publicly.

Requests include a zone plus validated sensor measurements and may include an ISO timestamp. Timestamp is retained in storage and zone is used as a categorical model feature. Neither target labels nor target-derived fields are model inputs. Training uses seed 42 and an 80/20 holdout; classification splits are stratified. See `models/training_report.json` for dataset validation and holdout metrics.
