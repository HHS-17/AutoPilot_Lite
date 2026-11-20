# Autopilot-lite

Autopilot-lite replays five-minute Borg observations and issues one CPU limit and one memory limit per job per hour. Each limit represents a per-task allocation for the job's replicas.

| CLI model | Recommendation |
| --- | --- |
| `autopilot` | Chooses limits from exponentially decayed histograms. An ensemble of half-lives and additive safety margins is scored online on overruns, underruns, and limit changes. Model switching has its own penalty. |
| `chronos` | Runs Chronos-Bolt Base zero-shot. Forecasts each resource independently and takes the largest p90 across the next 12 windows, then applies a tuned margin. |
| `toto` | Runs Toto Open Base 1.0 zero-shot on the CPU and memory series jointly. Samples forecast paths and takes the largest marginal p98 across the next 12 windows, then applies a tuned margin. |
| `lightgbm` | Fits one global p98 quantile regressor per resource on the training period. The target is the following hour's peak. Features include lags, changes, rolling statistics through 72 hours, calendar features, and initial job metadata. |
| `baseline` | Holds each job at the initial CPU and memory requests. |

## Run on WSL

Run these commands in **Bash inside Ubuntu on WSL 2**. The project-local `.venv-wsl` keeps Linux packages separate from the Windows `.venv`.

### 1. Open the project and create the Linux virtual environment

```bash
cd '/mnt/c/Users/philh/Downloads/CS/Cloud Computing'

sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg libgomp1
```

On Ubuntu 24.04, install Python 3.12 and create the environment.

```bash
sudo apt-get install -y python3.12 python3.12-venv
python3.12 -m venv .venv-wsl
```

If your distro's package repository lacks Python 3.12, use these commands to create the environment. [uv installation and Python management](https://docs.astral.sh/uv/getting-started/installation/)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
"$HOME/.local/bin/uv" venv --python 3.12 --seed .venv-wsl
```

Activate the environment and check that Python reports version 3.12. Run the `cd` and `source` commands again when you open a fresh WSL terminal.

```bash
source .venv-wsl/bin/activate
python --version
```

### 2. Install all Python dependencies

This command installs all four models' Python dependencies, including PyTorch.

```bash
python -m pip install -r requirements.txt
```

On x86-64 WSL, the default PyTorch wheel includes CUDA dependencies, so the download can be large. The evaluator's `--device auto` uses CUDA when PyTorch can access a compatible NVIDIA GPU and otherwise runs on CPU. [PyTorch package](https://pypi.org/project/torch/2.7.0/#files) · [WSL GPU setup](https://learn.microsoft.com/en-us/windows/ai/directml/gpu-cuda-in-wsl)

### 3. Run the unit tests and the synthetic demos

The unit tests use fixture forecasts for the two neural adapters. The first demo uses two synthetic jobs with one day per split and runs on CPU.

```bash
python -m unittest -v

python evaluate.py --demo --models baseline autopilot lightgbm \
  --device cpu --output results/demo
head -n 8 results/demo/summary.csv
```

The second demo runs Chronos-Bolt and Toto on one synthetic job with fewer Toto samples for a quick check. The first run downloads both checkpoints from Hugging Face. Use `--local-models-only` after both checkpoints are cached if you want to run offline.

```bash
python evaluate.py --demo --max-jobs 1 --models chronos toto \
  --device auto --context 512 --batch-size 1 \
  --toto-samples 64 --toto-samples-per-batch 8 --output results/neural-demo
head -n 6 results/neural-demo/summary.csv
```

### 4. Set up access to BigQuery and Cloud Storage

For the real experiment, use a Google Cloud project with billing and permission to create BigQuery tables and export to Cloud Storage. Sign in with the account that has access to the Borg trace. If the source dataset is restricted for your account, follow the access instructions in Google's trace documentation. [Google's Borg trace access instructions](https://github.com/google/cluster-data/blob/master/ClusterData2019.md)

Install the Google Cloud CLI if `gcloud` and `bq` are missing.

```bash
curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
  | sudo gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg
printf '%s\n' 'deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main' \
  | sudo tee /etc/apt/sources.list.d/google-cloud-sdk.list >/dev/null
sudo apt-get update
sudo apt-get install -y google-cloud-cli
```

Open the sign-in URL in your Windows browser and paste the authorization code back into WSL. Fill in your project ID and a globally unique bucket name. Keep the same terminal for the export steps. [Google Cloud CLI installation](https://docs.cloud.google.com/sdk/docs/install-sdk) · [Browser sign-in](https://docs.cloud.google.com/sdk/gcloud/reference/auth/login)

```bash
gcloud auth login --no-launch-browser

AUTOPILOT_PROJECT='your-google-cloud-project-id'
AUTOPILOT_DATASET='autopilot_lite'
AUTOPILOT_BUCKET='your-globally-unique-bucket-name'
AUTOPILOT_SOURCE='google.com:google-cluster-data:clusterdata_2019_a'

# Run this if the APIs still need to be enabled in your project.
gcloud services enable bigquery.googleapis.com storage.googleapis.com \
  --project="$AUTOPILOT_PROJECT"

AUTOPILOT_LOCATION="$(bq --project_id="$AUTOPILOT_PROJECT" show \
  --format=prettyjson "$AUTOPILOT_SOURCE" \
  | python -c 'import json, sys; print(json.load(sys.stdin)["location"])')"
printf 'Borg dataset location: %s\n' "$AUTOPILOT_LOCATION"
```

Create the dataset and bucket once in the source dataset's location. If you already have them in that location, skip their creation commands.

```bash
bq --location="$AUTOPILOT_LOCATION" mk --dataset "$AUTOPILOT_PROJECT:$AUTOPILOT_DATASET"
gcloud storage buckets create "gs://$AUTOPILOT_BUCKET" \
  --project="$AUTOPILOT_PROJECT" --location="$AUTOPILOT_LOCATION" \
  --uniform-bucket-level-access
```

### 5. Build and download the Borg job panel

[export_borg.sql](export_borg.sql) uses cell A and samples up to 600 serving-like and 400 batch-like jobs using training-period coverage and scheduling class. Make a local copy with your destination. The second copy skips table creation so the dry run can estimate the data scan. Review the estimate before submitting the billable query. A smaller sample limits the exported rows and replay work. BigQuery can still scan the same source columns. [BigQuery query cost controls](https://docs.cloud.google.com/bigquery/docs/best-practices-costs)

```bash
mkdir -p data
sed "s/YOUR_PROJECT\.YOUR_DATASET/$AUTOPILOT_PROJECT.$AUTOPILOT_DATASET/" \
  export_borg.sql > data/export_borg.local.sql
sed '/^CREATE TABLE .* AS[[:space:]]*$/d' data/export_borg.local.sql > data/export_borg.estimate.sql

bq --project_id="$AUTOPILOT_PROJECT" --location="$AUTOPILOT_LOCATION" query \
  --use_legacy_sql=false --dry_run < data/export_borg.estimate.sql
```

Enter the maximum bytes you want to allow for the query as a whole number, based on the dry-run estimate. The query creates `autopilot_jobs`. If that table already contains the export you want, go straight to the Parquet export.

```bash
read -r -p 'Maximum bytes to bill for the query: ' AUTOPILOT_MAX_BYTES
bq --project_id="$AUTOPILOT_PROJECT" --location="$AUTOPILOT_LOCATION" query \
  --use_legacy_sql=false --maximum_bytes_billed="$AUTOPILOT_MAX_BYTES" \
  < data/export_borg.local.sql

bq --project_id="$AUTOPILOT_PROJECT" show "$AUTOPILOT_PROJECT:$AUTOPILOT_DATASET.autopilot_jobs"
```

Export to Parquet and download all shards to a fresh local data directory. For a second export, choose a new local directory and use it in the replay command. [BigQuery export formats and locations](https://docs.cloud.google.com/bigquery/docs/exporting-data)

```bash
AUTOPILOT_EXPORT_PREFIX="autopilot/$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p data/borg_jobs
bq --project_id="$AUTOPILOT_PROJECT" --location="$AUTOPILOT_LOCATION" extract \
  --destination_format=PARQUET --compression=SNAPPY \
  "$AUTOPILOT_PROJECT:$AUTOPILOT_DATASET.autopilot_jobs" \
  "gs://$AUTOPILOT_BUCKET/$AUTOPILOT_EXPORT_PREFIX/jobs-*.parquet"
gcloud storage cp "gs://$AUTOPILOT_BUCKET/$AUTOPILOT_EXPORT_PREFIX/jobs-*.parquet" data/borg_jobs/
ls -lh data/borg_jobs/
```

### 6. Run the real-data replay

If you already have a Borg export, pass its CSV, Parquet file, or Parquet directory to `--data`.

Try all four models and the baseline on ten jobs first. Both commands use a 512-window history and a small inference batch. Toto is substantially slower on CPU.

```bash
python evaluate.py --data data/borg_jobs --max-jobs 10 --device auto \
  --context 512 --batch-size 1 --toto-samples-per-batch 8 \
  --output results/borg-10
```

Run the same comparison on every exported job.

```bash
python evaluate.py --data data/borg_jobs --device auto \
  --context 512 --batch-size 1 --toto-samples-per-batch 8 \
  --output results/borg
```

Every run needs a new or empty output directory. To rerun, use a new path such as `results/borg-rerun`. If Toto runs out of GPU memory, lower `--toto-samples-per-batch` to 4. The total sample count must be divisible by the samples per batch.

### 7. Inspect the results

```bash
head -n 11 results/borg/summary.csv
ls -lh results/borg/
explorer.exe "$(wslpath -w "$PWD/results/borg")"
```

For the ten-job run, replace `results/borg` with `results/borg-10`.

## Input format

The evaluator accepts a directory of Parquet shards or a single file containing the columns below. Use normalized Borg units and one row per job and five-minute window. Leave missing windows absent from the file.

```text
job_id,timestamp,cpu_avg,memory_avg,cpu_peak,memory_peak,cpu_request,memory_request,task_count,priority,scheduling_class,sample_rank
```

`timestamp` is the UTC window start. The trace's documented origin is May 1, 2019 at 00:00 PDT, or 07:00 UTC. `task_count` defaults to 1. Priority, scheduling class, and sample rank are optional. Request and metadata values are held fixed per job. The export uses mean usage across reporting tasks for slack, mean task peak for the CPU overrun proxy, and maximum task peak for the memory overrun proxy. It uses the CPU distribution's top percentile alongside the recorded maximum and falls back to mean usage when a peak is unavailable.

## Replay and results

Trace days 1–14 train LightGBM, days 15–21 tune the Autopilot cost configurations and forecast margins, and days 22–31 are held out for evaluation. Tuning selects the configuration with the lowest relative slack within the 5% CPU and 2% memory overrun budgets. If none meet the budget, it selects the lowest overrun rate, then the lowest slack. Use `--help` to see the split and tuning options.

Every method uses the same limit filter. Increases take effect immediately. A decrease requires a proposed drop greater than 15% and two consecutive completed hours of low observed usage. Forecast contexts fill missing windows forward from past values. Missing windows stay out of training targets and scoring, and an incomplete hour interrupts the filter's low-usage streak.

Open `summary.csv` first.

| Output | Contents |
| --- | --- |
| `summary.csv` | Test-period aggregate slack, overrun rate, limit changes, and mean and p95 compute time by method and resource. |
| `per_job.csv` and `per_job_day.csv` | Job-level scores plus daily p95-based relative slack and mean signed net slack. |
| `hourly_*.csv` | Proposed limits, applied limits, realized peaks, and coverage flags for each held-out hour. |
| `tuning.csv` and `run.json` | Candidate scores, selected settings, split dates, options, and elapsed runtime. |
| `slack_cdf.png` and `tradeoff.png` | Relative-slack distribution and slack versus overrun plots. |
| `slack_by_method.png` | Relative and absolute slack for CPU and memory by method. |
| `overruns_and_limit_changes.png` | OOM and throttling proxy rates and CPU and memory limit-change counts. |
| `compute_cost.png` | Mean and p95 prediction time plus elapsed method runtime when `run.json` is available. |
| `lightgbm_importance.csv` | Feature gains from the fitted CPU and memory models. |

To regenerate the plots from the completed Borg run, use the saved results directory.

```bash
python evaluate.py --plot-from results/borg-final
```

Aggregate relative slack is the fraction of reserved capacity left unused. Absolute slack integrates positive headroom over reporting task-hours in normalized resource units. Signed net slack also counts deficits. The job-day relative metric uses the p95 of five-minute job-level average usage as a public-trace approximation to the paper's p95 across task readings. Prediction times are amortized per job-hour and cover both resources. They exclude model loading, training, tuning, and CSV output.

Overrun rates count scored five-minute job windows whose peak proxy exceeds the applied limit. They estimate OOM or throttling exposure from the recorded workload. The simulated limits do not feed back into the trace. Autopilot's cost and model-selection logic follows equations 4–9 using 256-bin histograms of the available five-minute job proxies. The paper's production recommender uses finer-grained per-task distributions. [Google Autopilot paper](https://doi.org/10.1145/3342195.3387524)
