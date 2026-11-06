PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS prediction_events (
  event_id TEXT PRIMARY KEY,
  idempotency_key TEXT UNIQUE,
  hazard_type TEXT NOT NULL DEFAULT 'environmental',
  source TEXT NOT NULL CHECK(source IN ('synthetic_demo','live_sensor','external_data')),
  zone_type TEXT NOT NULL,
  measurements_json TEXT NOT NULL,
  prediction_json TEXT NOT NULL,
  model_names_json TEXT NOT NULL,
  model_version TEXT NOT NULL,
  prediction_timestamp TEXT NOT NULL,
  data_quality_status TEXT NOT NULL,
  notification_processing_status TEXT NOT NULL DEFAULT 'pending',
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_prediction_events_created ON prediction_events(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_prediction_events_source ON prediction_events(source, created_at DESC);

CREATE TABLE IF NOT EXISTS in_app_notifications (
  notification_id INTEGER PRIMARY KEY AUTOINCREMENT,
  event_id TEXT NOT NULL REFERENCES prediction_events(event_id) ON DELETE CASCADE,
  hazard_type TEXT NOT NULL,
  severity TEXT NOT NULL,
  title TEXT NOT NULL,
  message TEXT NOT NULL,
  source TEXT NOT NULL,
  zone_type TEXT NOT NULL,
  read_at TEXT,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  UNIQUE(event_id, hazard_type)
);
CREATE INDEX IF NOT EXISTS idx_notifications_unread ON in_app_notifications(read_at, created_at DESC);

CREATE TABLE IF NOT EXISTS notification_preferences (
  preference_id INTEGER PRIMARY KEY CHECK(preference_id = 1),
  recipient_email TEXT NOT NULL DEFAULT 'divyadeva0908@gmail.com',
  in_app_enabled INTEGER NOT NULL DEFAULT 1,
  email_enabled INTEGER NOT NULL DEFAULT 0,
  hazard_types_json TEXT NOT NULL DEFAULT '["flood","fire","water_quality"]',
  severities_json TEXT NOT NULL DEFAULT '["High"]',
  synthetic_email_enabled INTEGER NOT NULL DEFAULT 0,
  updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS email_delivery_logs (
  delivery_id INTEGER PRIMARY KEY AUTOINCREMENT,
  event_id TEXT REFERENCES prediction_events(event_id) ON DELETE SET NULL,
  hazard_type TEXT NOT NULL,
  recipient_email TEXT NOT NULL,
  subject TEXT NOT NULL,
  status TEXT NOT NULL CHECK(status IN ('pending','sending','accepted','failed')),
  attempts INTEGER NOT NULL DEFAULT 0,
  next_attempt_at TEXT,
  provider_message_id TEXT,
  provider_response_json TEXT,
  last_error TEXT,
  is_test INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  UNIQUE(event_id, hazard_type, recipient_email)
);
CREATE INDEX IF NOT EXISTS idx_email_queue ON email_delivery_logs(status, next_attempt_at, created_at);
