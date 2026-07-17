CREATE TABLE IF NOT EXISTS videos (
  video_id STRING PRIMARY KEY,
  source_path STRING,
  r2_raw_path STRING NULL,
  r2_raw_deleted_at TIMESTAMPTZ NULL,
  duration_raw FLOAT8,
  fps_raw FLOAT8,
  width_raw INT8,
  height_raw INT8,
  checksum_raw STRING,
  processing_status STRING,
  model_version STRING,
  config_version STRING,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS shots (
  shot_id STRING PRIMARY KEY,
  video_id STRING NOT NULL REFERENCES videos(video_id) ON DELETE CASCADE,
  shot_index INT8,
  shot_start_frame INT8,
  shot_end_frame INT8,
  shot_start_time_raw FLOAT8,
  shot_end_time_raw FLOAT8,
  duration_raw FLOAT8,
  r2_proxy_path STRING NULL,
  proxy_status STRING NULL,
  proxy_checksum STRING NULL,
  processing_status STRING,
  model_version STRING,
  config_version STRING,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_shots_video_id ON shots(video_id);

CREATE TABLE IF NOT EXISTS keyframes (
  keyframe_id STRING PRIMARY KEY,
  video_id STRING NOT NULL REFERENCES videos(video_id) ON DELETE CASCADE,
  shot_id STRING NOT NULL REFERENCES shots(shot_id) ON DELETE CASCADE,
  frame_index_raw INT8,
  timestamp_raw FLOAT8,
  timestamp_in_shot FLOAT8,
  selection_reason STRING,
  beit3_distance_prev FLOAT8 NULL,
  beit3_distance_last_keyframe FLOAT8 NULL,
  zilliz_inserted BOOL NOT NULL DEFAULT false,
  object_counts JSONB NOT NULL DEFAULT '{}'::JSONB,
  processing_status STRING,
  model_version STRING,
  config_version STRING,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_keyframes_video_time ON keyframes(video_id, timestamp_raw);
CREATE INDEX IF NOT EXISTS idx_keyframes_shot_id ON keyframes(shot_id);

CREATE TABLE IF NOT EXISTS processing_jobs (
  job_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  job_type STRING,
  video_id STRING NULL,
  shot_id STRING NULL,
  status STRING,
  error_message STRING NULL,
  retry_count INT8 NOT NULL DEFAULT 0,
  model_version STRING,
  config_version STRING,
  started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  finished_at TIMESTAMPTZ NULL
);

CREATE INDEX IF NOT EXISTS idx_processing_jobs_video_shot ON processing_jobs(video_id, shot_id);

CREATE TABLE IF NOT EXISTS query_cache (
  query_hash STRING,
  query_text STRING,
  gemini_paraphrases JSONB NOT NULL DEFAULT '[]'::JSONB,
  gemini_object_constraints JSONB NOT NULL DEFAULT '[]'::JSONB,
  sd_prompt STRING NULL,
  sd_image_paths_or_hashes JSONB NOT NULL DEFAULT '[]'::JSONB,
  sd_seeds JSONB NOT NULL DEFAULT '[]'::JSONB,
  cache_version STRING,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (query_hash, cache_version)
);

CREATE TABLE IF NOT EXISTS eval_runs (
  eval_run_id STRING PRIMARY KEY,
  dataset_name STRING,
  dataset_path STRING,
  config_version STRING,
  model_version STRING,
  latency_mode STRING,
  branch_config JSONB NOT NULL DEFAULT '{}'::JSONB,
  metrics JSONB NOT NULL DEFAULT '{}'::JSONB,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS eval_query_results (
  eval_run_id STRING NOT NULL REFERENCES eval_runs(eval_run_id) ON DELETE CASCADE,
  query_id STRING NOT NULL,
  query_text STRING,
  gt_video_id STRING,
  gt_start FLOAT8,
  gt_end FLOAT8,
  top_retrieved_keyframes JSONB NOT NULL DEFAULT '[]'::JSONB,
  hit_at_1 BOOL,
  hit_at_5 BOOL,
  hit_at_10 BOOL,
  latency_ms FLOAT8,
  error_message STRING NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (eval_run_id, query_id)
);

