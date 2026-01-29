# INF2009 – **Profiling for Edge Computing**

## 1. Overview

In the previous labs, you learned how to perform:

*   Image & Video analytics on the edge (OpenCV)
*   Sound analytics (streaming, FFT, feature extraction)
*   Deep Learning inference (FP vs quantized)
*   IoT communication using MQTT

This lab brings everything together: **you will measure, compare, and schedule** your own processes using proper profiling tools. By the end, you will know how your pipelines behave on **1‑core vs multi‑core**, how to quantify latency/CPU/memory, and how to **schedule tasks based on deadlines**.

***

## 2. Learning Objectives

By the end of this lab, you will be able to:

1.  Profile a script/pipeline using:
    *   Wall‑clock time vs CPU time
    *   CPU % / memory usage
    *   Latency distribution (mean, p50, p95, p99, max)
    *   System counters (perf cycles, cache misses, context switches)

2.  Compare **single‑core** vs **multi‑core** execution and reason about scaling.

3.  Create a **Profiling Card** for each pipeline (image/video, audio, DL, MQTT).

4.  Use OS‑level controls (taskset, priority, cgroups) and implement a simple user‑space **Earliest Deadline First (EDF)** scheduler.

5.  Provide a scheduling plan that meets the deadlines of multiple tasks on the RPi5.

***

## 3. Hardware & Software Requirements

### Hardware

*   Raspberry Pi **5** (8 GB recommended)
*   64‑bit Raspberry Pi OS
*   USB webcam or Pi Camera Module 3
*   USB microphone
*   Internet connection

### Software

Install system packages:

```bash
sudo apt update
sudo apt install -y linux-perf time sysstat numactl util-linux \
                    cpufrequtils stress-ng python3-psutil \
                    mosquitto mosquitto-clients
```

Reuse your previous virtual environments:

*   `imgvid_env` (Image/Video)
*   `audio` (Sound)
*   `dlonedge` (DL on Edge)
*   MQTT clients (Python Paho)

***

## 4. Environment Setup

Create a new profiling environment (separate from earlier labs):

```bash
mkdir -p ~/inf2009
cd ~/inf2009

python3 -m venv profiling_env
source profiling_env/bin/activate

pip install psutil numpy matplotlib
```

Download the starter code folder (you will paste the snippets below).

***

## 5. Conceptual Background (Read Carefully)

### 5.1 Wall‑clock vs CPU time

*   **Wall‑clock** includes waiting, I/O, OS scheduling delays.
*   **CPU time** is actual time the CPU spent on your code.

### 5.2 Latency distribution

Mean is misleading—**tail latency** (p95, p99) determines deadline success.

### 5.3 RPi5 core characteristics

RPi5 has 4 high‑performance Cortex‑A76 cores. You can pin tasks to:

*   1 core → predictable but slower
*   Multiple cores → higher throughput but possible interference

***

## 6. Part A — Profiling Your Pipelines

### 6.1 Add timing wrappers to your code

Create `prof_utils.py`:

```python
# prof_utils.py
import os, time, csv, psutil
from contextlib import contextmanager

PROC = psutil.Process(os.getpid())

def _sample_proc():
    with PROC.oneshot():
        return {
            "cpu_percent": PROC.cpu_percent(interval=None),
            "rss_mb": PROC.memory_info().rss / (1024*1024),
            "threads": PROC.num_threads(),
            "affinity": PROC.cpu_affinity()
        }

@contextmanager
def measure(label, meta=None, log_path="profile_log.csv"):
    t0 = time.perf_counter_ns()
    t_cpu0 = time.process_time_ns()
    PROC.cpu_percent(interval=None)  # prime

    yield  # execute user block

    t1 = time.perf_counter_ns()
    t_cpu1 = time.process_time_ns()

    p = _sample_proc()
    row = {
        "label": label,
        "wall_ms": (t1 - t0)/1e6,
        "cpu_ms": (t_cpu1 - t_cpu0)/1e6,
        "cpu_percent": p["cpu_percent"],
        "rss_mb": p["rss_mb"],
        "threads": p["threads"],
        "affinity": ",".join(map(str, p["affinity"])),
        "timestamp_ns": t1
    }
    if meta:
        row.update(meta)
    write_header = not os.path.exists(log_path)
    with open(log_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            w.writeheader()
        w.writerow(row)
```

### 6.2 Apply the wrapper

Inside each of your previous lab scripts:

```python
from prof_utils import measure

with measure("frame_process", meta={"src":"camera"}):
    # your image/video/audio/DL processing block
    process_frame()
```

### 6.3 System‑level profiling

```bash
/usr/bin/time -v python script.py
perf stat -e cycles,instructions,cache-misses,cs,migrations \
          -- python script.py
perf record -F 99 -g -- python script.py
perf report
```

***

## 7. Part B — 1 Core vs Multi‑Core

### 7.1 Pin to a single core

```bash
taskset -c 0 python script.py
```

### 7.2 Pin to multiple cores

```bash
taskset -c 0-3 python script.py
```

### 7.3 Control library threading

Set before running DL pipelines:

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```

PyTorch (from your DL lab):

```python
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
```

### 7.4 Collect scaling curves

Measure latency & throughput for:

*   1 thread / 1 core
*   N threads / 4 cores
*   Quantized vs FP32 DL models

***

## 8. Part C — End‑to‑End Latency with MQTT

### 8.1 Producer → Processor → Consumer

Use your existing MQTT publisher/subscriber from the MQTT lab.

Publisher example:

```python
msg = {"id": frame_id, "t_pub_ns": time.time_ns()}
client.publish("lab/e2e", json.dumps(msg))
```

Processor:

*   Subscribe → run your pipeline → publish result with `t_done_ns`.

Subscriber:

```bash
mosquitto_sub -t "lab/e2e/processed" -v | ts '%s%N' >> mqtt_log.txt
```

### 8.2 Add network impairments (optional)

```bash
sudo tc qdisc add dev eth0 root netem delay 40ms loss 1%
```

Compare MQTT QoS 0 vs 1 vs 2.

***

## 9. Part D — Build Your Scheduling Plan

### 9.1 Create a **Profiling Card** (one per pipeline)

Example (`profiling_cards/mobilenet.yaml`):

```yaml
name: mobilenet_v2_infer
period_ms: 100
deadline_ms: 100
wcet_ms: 18
avg_ms: 12.5
rss_mb: 210
cpu_share_percent: 130
notes: "int8 quantized; torch threads=1; pinned to core 2"
```

### 9.2 OS controls

*   Priority:
    ```bash
    sudo chrt --rr 50 python task.py
    ```
*   CPU affinity:
    ```bash
    taskset -c 2 python audio_task.py
    ```
*   CPU quotas (cgroups v2):  
    Allocate CPU time to ensure DL does not starve audio.

### 9.3 User‑space EDF scheduler

Create `edf_runner.py`:

```python
# edf_runner.py
import time, heapq

class PeriodicTask:
    def __init__(self, name, period_ms, deadline_ms, fn):
        self.name = name
        self.P = period_ms/1000
        self.D = deadline_ms/1000
        self.fn = fn
        self.next_release = time.monotonic()

    def release(self):
        self.next_release += self.P
        return (self.next_release + self.D, self)

def run_edf(tasks):
    q = [(t.next_release + t.D, t) for t in tasks]
    heapq.heapify(q)

    while True:
        dl, t = heapq.heappop(q)
        now = time.monotonic()
        if now < t.next_release:
            time.sleep(t.next_release - now)

        t_start = time.monotonic()
        t.fn()
        exec_ms = (time.monotonic() - t_start)*1000
        miss = time.monotonic() > dl
        print(f"{t.name}: {exec_ms:.2f} ms | Missed deadline: {miss}")

        heapq.heappush(q, t.release())
```

Register at least 3 tasks:

*   Audio feature extractor
*   Image/video frame analyzer
*   DL inference block

***

## 10. Deliverables

### Submit the following:

1.  **Profiling Cards** for each pipeline
2.  **CSV logs** generated by `prof_utils.measure`
3.  **Scaling plots** (threads vs latency/throughput/jitter)
4.  **MQTT latency breakdown**
5.  **Scheduling plan** showing how you meet all deadlines
6.  **EDF run output** showing deadline misses before/after your plan
7.  Optional:
    *   `perf` flamegraphs
    *   Comparison of FP vs quantized models

***

## 11. Marking Criteria (100%)

*   **Correct use of profiling tools**
*   **Depth of analysis (p95/p99, jitter, interference)**
*   **Scheduling plan correctness & justification**
*   **Clarity & reproducibility of report**

***

## 12. Appendix — Quick Command Cheatsheet

```bash
# Profile CPU & memory
/usr/bin/time -v python app.py
pidstat -rud -p $(pgrep -f app.py) 1

# Pin cores
taskset -c 0 python app.py

# Perf
perf stat -e cycles,instructions,cache-misses -- python app.py
perf record -F 99 -g -- python app.py

# MQTT subscriber with timestamps
mosquitto_sub -t lab/e2e/# -v | ts '%s%N'

# Stress interference
stress-ng --cpu 4 --timeout 20s
```

***

If you'd like, I can also generate a **ZIP‑ready folder structure** with all starter files (`prof_utils.py`, `edf_runner.py`, sample pipelines, folders for logs, plots, profiling cards) so you can drop it directly into your course repo.
