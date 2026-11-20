-- BigQuery GoogleSQL. Replace YOUR_PROJECT.YOUR_DATASET with your destination.
-- Change both clusterdata_2019_a references to use another cell.
DECLARE trace_start TIMESTAMP DEFAULT TIMESTAMP('2019-05-01 07:00:00+00');
DECLARE trace_offset_us INT64 DEFAULT 600000000;
DECLARE window_us INT64 DEFAULT 300000000;
DECLARE train_windows INT64 DEFAULT 14 * 24 * 12;
DECLARE min_training_windows INT64 DEFAULT 12 * 24 * 12;
DECLARE trace_windows INT64 DEFAULT 31 * 24 * 12;
DECLARE serving_jobs INT64 DEFAULT 600;
DECLARE batch_jobs INT64 DEFAULT 400;

CREATE TABLE `YOUR_PROJECT.YOUR_DATASET.autopilot_jobs` AS
WITH initial_requests AS (
  SELECT
    e.collection_id,
    STRUCT(
      e.time AS event_time,
      e.resource_request.cpus AS cpu_request,
      e.resource_request.memory AS memory_request,
      e.priority AS priority,
      e.scheduling_class AS scheduling_class
    ) AS request
  FROM `google.com:google-cluster-data`.clusterdata_2019_a.instance_events AS e
  WHERE e.collection_type = 0
    -- A later first request cannot precede enough training windows to qualify.
    AND e.time <= trace_offset_us + (train_windows - min_training_windows + 1) * window_us
    AND e.resource_request.cpus BETWEEN 0 AND 1
    AND e.resource_request.memory BETWEEN 0 AND 1
    AND e.scheduling_class IS NOT NULL
  QUALIFY ROW_NUMBER() OVER (
    PARTITION BY e.collection_id
    ORDER BY e.time, e.instance_index, e.type, e.resource_request.cpus,
      e.resource_request.memory, e.scheduling_class, e.priority
  ) = 1
),
reporting_tasks AS (
  SELECT
    u.collection_id,
    DIV(u.end_time - trace_offset_us - 1, window_us) AS window_index,
    u.instance_index,
    u.end_time,
    i.request,
    u.end_time - u.start_time AS duration_us,
    u.average_usage.cpus AS cpu_avg,
    u.average_usage.memory AS memory_avg,
    GREATEST(
      u.average_usage.cpus,
      IF(u.maximum_usage.cpus BETWEEN 0 AND 1, u.maximum_usage.cpus, 0),
      IFNULL(ARRAY_REVERSE(u.cpu_usage_distribution)[SAFE_OFFSET(0)], 0)
    ) AS cpu_peak,
    GREATEST(
      u.average_usage.memory,
      IF(u.maximum_usage.memory BETWEEN 0 AND 1, u.maximum_usage.memory, 0)
    ) AS memory_peak
  FROM `google.com:google-cluster-data`.clusterdata_2019_a.instance_usage AS u
  JOIN initial_requests AS i USING (collection_id)
  WHERE u.collection_type = 0
    AND u.end_time > trace_offset_us
    AND u.end_time <= trace_offset_us + trace_windows * window_us
    AND u.end_time - u.start_time >= 240000000
    AND u.average_usage.cpus BETWEEN 0 AND 1
    AND u.average_usage.memory BETWEEN 0 AND 1
),
task_windows AS (
  SELECT
    collection_id,
    window_index,
    instance_index,
    MIN(end_time) AS first_usage_end,
    ANY_VALUE(request) AS request,
    SUM(cpu_avg * duration_us) / SUM(duration_us) AS cpu_avg,
    SUM(memory_avg * duration_us) / SUM(duration_us) AS memory_avg,
    MAX(LEAST(cpu_peak, 1)) AS cpu_peak,
    MAX(memory_peak) AS memory_peak
  FROM reporting_tasks
  GROUP BY collection_id, window_index, instance_index
),
job_windows AS (
  SELECT
    collection_id,
    window_index,
    MIN(first_usage_end) AS first_usage_end,
    ANY_VALUE(request) AS request,
    AVG(cpu_avg) AS cpu_avg,
    AVG(memory_avg) AS memory_avg,
    AVG(cpu_peak) AS cpu_peak,
    MAX(memory_peak) AS memory_peak,
    COUNT(*) AS task_count
  FROM task_windows
  GROUP BY collection_id, window_index
),
qualified_jobs AS (
  -- Compute coverage and the export panel from the same usage scan.
  SELECT
    collection_id,
    ANY_VALUE(request) AS request,
    ARRAY_AGG(STRUCT(window_index, cpu_avg, memory_avg, cpu_peak, memory_peak, task_count)) AS windows
  FROM job_windows AS w
  GROUP BY w.collection_id
  HAVING COUNTIF(w.window_index < train_windows) >= min_training_windows
    AND MIN(w.request.event_time)
      <= MIN(IF(w.window_index < train_windows, w.first_usage_end, NULL))
),
ranked AS (
  SELECT *,
    ROW_NUMBER() OVER (
      PARTITION BY request.scheduling_class >= 2
      ORDER BY FARM_FINGERPRINT(CAST(collection_id AS STRING)), collection_id
    ) AS sample_rank
  FROM qualified_jobs
),
sample AS (
  SELECT *
  FROM ranked
  WHERE sample_rank <= IF(request.scheduling_class >= 2, serving_jobs, batch_jobs)
)
SELECT
  CAST(s.collection_id AS STRING) AS job_id,
  TIMESTAMP_MICROS(UNIX_MICROS(trace_start) + u.window_index * window_us) AS timestamp,
  u.cpu_avg,
  u.memory_avg,
  u.cpu_peak,
  u.memory_peak,
  u.task_count,
  s.request.cpu_request,
  s.request.memory_request,
  s.request.priority,
  s.request.scheduling_class,
  s.sample_rank
FROM sample AS s
CROSS JOIN UNNEST(s.windows) AS u;
