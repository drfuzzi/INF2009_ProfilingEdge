# INF2009 — **Profiling for Edge Computing**

## 1. Overview

In the previous labs, you learned how to perform:

*   Image & Video analytics on the edge (OpenCV)
*   Sound analytics (streaming, FFT, feature extraction)
*   Deep Learning inference (FP vs quantized)
*   IoT communication using MQTT

This lab brings everything together.  
You will **measure**, **profile**, **compare**, and **schedule** computational tasks on the Raspberry Pi 5.

By the end of this lab, you will understand:

*   How your scripts behave on **1 core vs multiple cores**
*   How to measure **latency**, **resource usage**, and **system overhead**
*   How to identify **performance bottlenecks**
*   How to schedule tasks based on **deadlines** and **resource constraints**

***

## 2. Learning Objectives

By the end of this lab, you will be able to:

1.  Profile a script using:
    *   Wall‑clock vs CPU time
    *   CPU%, memory usage (RSS)
    *   Latency distribution (mean, p50, p95, p99)
    *   Context switches, instructions, cache misses (via `perf stat`)

2.  Compare **single‑core** vs **multi‑core** execution.

3.  Use OS‑level controls:
    *   `taskset` (CPU affinity)
    *   `chrt` (priority)
    *   cgroups (CPU quotas)

4.  Implement a **simple EDF scheduler** (Earliest Deadline First).

5.  Build a scheduling plan that meets task deadlines.

***

## 3. Hardware & Software Requirements

### Hardware

*   Raspberry Pi **5** (8GB recommended)
*   Raspberry Pi OS **64‑bit**
*   USB webcam or Pi Camera Module 3
*   USB microphone
*   Internet connection

### System Tools

Install required system-level packages:

```bash
sudo apt update
sudo apt install -y linux-perf time sysstat numactl util-linux \
                    linux-cpupower stress-ng python3-psutil \
                    mosquitto mosquitto-clients
```

***

## 4. Environment Setup

Create a new profiling environment:

```bash
mkdir -p ~/inf2009
cd ~/inf2009
python3 -m venv profiling_env
source profiling_env/bin/activate
```

Install general Python libraries:

```bash
pip install psutil numpy matplotlib
```

***

### 4.1 If You DO NOT Have Code From Previous Labs (Important)

If you did not complete earlier labs (Image, Audio, DL, MQTT), download the **Sample Script Pack**:

```bash
cd ~/inf2009
git clone https://github.com/drfuzzi/INF2009_ProfilingEdge.git
```

You should now have:

    ~/inf2009/INF2009_ProfilingEdge/profiling_package/
        sample_img.py
        sample_audio.py
        sample_dl.py
        sample_mqtt_pub.py
        sample_mqtt_proc.py
        sample_mqtt_sub.py

These scripts are **NOT profiling scripts**.  
They are **small standalone tasks** representing previous labs.  
**You will manually apply profiling tools to them in this lab.**

***

### 4.2 Required Python Libraries for Sample Scripts

Inside your profiling environment:

```bash
pip install numpy
pip install opencv-python
pip install sounddevice
pip install scipy
pip install paho-mqtt
pip install matplotlib
pip install psutil
```

### Optional — Deep Learning Sample

(If you want to run `sample_dl.py`)

```bash
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu
```

If PyTorch cannot be installed on your Pi, skip the DL task.

***

### 4.3 Running the Sample Scripts

Move into:

```bash
cd ~/inf2009/INF2009_ProfilingEdge/profiling_package/
```

### Image / Video

```bash
python sample_img.py
```

### Audio

```bash
python sample_audio.py
```

### Deep Learning (if installed)

```bash
python sample_dl.py
```

### MQTT

Terminal 1:

```bash
python sample_mqtt_proc.py
```

Terminal 2:

```bash
python sample_mqtt_sub.py
```

Terminal 3:

```bash
python sample_mqtt_pub.py
```

***

# 5. Conceptual Background

### 5.1 Wall‑clock vs CPU Time

*   **Wall‑clock time**: includes waiting, I/O delays, scheduling delays.
*   **CPU time**: actual time CPU spent running your code.

### 5.2 Why Percentiles Matter (p95, p99)

Your pipeline may average 10 ms but occasionally take 80 ms.  
Deadlines are violated by **tail latency** — not averages.

### 5.3 RPi5 Cores

RPi5 includes **4 Cortex‑A76 high‑performance cores**.

Use:

*   **1 core** for predictability
*   **4 cores** for throughput
*   Compare both

***

# 6. Part A — Profiling Your Pipelines

You will apply profiling **manually** using these tools.

### 6.1 `/usr/bin/time`

```bash
/usr/bin/time -v python sample_img.py
```

Useful fields:

*   Max RSS (memory)
*   User vs Sys CPU time

***

### 6.2 `perf stat`

```bash
perf stat -e cycles,instructions,cache-misses,cs,migrations -- python sample_audio.py
```

This reveals:

*   CPU cycles
*   Cache behavior
*   Context switch / migration count

***

### 6.3 `perf record`

```bash
perf record -F 99 -g -- python sample_img.py
perf report
```

Shows where CPU time is spent (functions, call stacks).

***

### 6.4 `pidstat`

```bash
pidstat -rud -p $(pgrep -f sample_img.py) 1
```

Monitors:

*   CPU%
*   RSS memory
*   I/O
*   Context switches (voluntary/involuntary)

***

# 7. Part B — Single Core vs Multi‑Core

### 7.1 Pin to Single Core

```bash
taskset -c 0 python sample_img.py
```

### 7.2 Use All 4 Cores

```bash
taskset -c 0-3 python sample_img.py
```

### 7.3 (For DL) Limit Library Threads

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```

***

# 8. Part C — End‑to‑End Latency with MQTT

Use sample scripts or your own.

### Subscriber with timestamps

```bash
mosquitto_sub -t "lab/e2e/processed" -v | ts '%s%N' >> mqtt_log.txt
```

### Optional: Network Impairment

```bash
sudo tc qdisc add dev eth0 root netem delay 40ms loss 1%
```

***

# 9. Part D — Scheduling Plan & EDF

You will:

*   Build profiling cards
*   Decide deadlines
*   Assign cores
*   Limit CPU quotas if necessary
*   Use EDF scheduler starter code (provided separately in lab)

### Example OS Controls

#### Priority

```bash
sudo chrt --rr 50 python task.py
```

#### Affinity

```bash
taskset -c 2 python sample_audio.py
```

#### Cgroups (optional)

Used to limit CPU shares of greedy tasks.

***

# 10. Deliverables

Submit:

1.  **Profiling Cards** (one per task)
2.  **CSV logs** from your profiling runs
3.  Plots of:
    *   latency distribution
    *   CPU% vs time
    *   1‑core vs multi‑core
4.  MQTT end‑to‑end latency analysis
5.  Your EDF scheduling plan
6.  EDF run output showing deadline hits/misses

***

# 11. Quick Command Cheatsheet

```bash
/usr/bin/time -v python app.py
perf stat -e cycles,instructions,cache-misses --
perf record -F 99 -g --
perf report
pidstat -rud -p $(pgrep -f app.py) 1
taskset -c 0 python app.py
taskset -c 0-3 python app.py
mosquitto_sub -t lab/e2e/# -v | ts '%s%N'
stress-ng --cpu 4 --timeout 20s
```

***
