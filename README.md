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

4.  Implement a **simple scheduler**.

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
                    mosquitto mosquitto-clients moreutils libcamera-apps alsa-utils
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
pip install scikit-image opencv-python matplotlib
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

### Deep Learning Sample

Needed to run `sample_dl.py`

```bash
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu
```

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
arecord -D hw:2,0 -d 5 -r 16000 -f S16_LE -c 2 test.wav
python sample_audio.py
```

### Deep Learning (if installed)

```bash
python sample_dl.py
```

***

# 5. Conceptual Background

### 5.1 Wall‑clock vs CPU Time

You can capture both metrics directly within your Python scripts (like sample_img.py or sample_audio.py) using the time library.

*   **Wall‑clock time**: includes waiting, I/O delays, scheduling delays.
*   **CPU time**: actual time CPU spent running your code.

```python
import time
import os

# Start counters
wall_start = time.perf_counter()    # Wall-clock (includes sleep/IO)
cpu_start = time.process_time()     # CPU-time (only active execution)

# --- Your Processing Code Here ---
# e.g., model(input_tensor)

# End counters
wall_end = time.perf_counter()
cpu_end = time.process_time()

print(f"Wall-clock Time: {wall_end - wall_start:.4f}s")
print(f"Total CPU Time:  {cpu_end - cpu_start:.4f}s")
```

### 5.2 Why Percentiles Matter (p95, p99)

To find tail latency, store the timing of each frame in a list and use numpy to calculate percentiles. This is more critical for edge devices than a simple average because a single "laggy" frame can break a real-time system. Your pipeline may average 10 ms but occasionally take 80 ms. Deadlines are violated by **tail latency** — not averages.

```python
import numpy as np

latencies = []
for i in range(100):
    start = time.perf_counter()
    # Run inference...
    latencies.append(time.perf_counter() - start)

# Calculate stats
p95 = np.percentile(latencies, 95)
p99 = np.percentile(latencies, 99)
avg = np.mean(latencies)

print(f"Average: {avg*1000:.2f}ms | p95: {p95*1000:.2f}ms | p99: {p99*1000:.2f}ms")
```

# 6. Part A — Profiling Your Pipelines

You will apply profiling **manually** using these tools.

### 6.1 `/usr/bin/time`

This command executes your script while the kernel tracks and reports a detailed summary of resource consumption, focusing heavily on memory usage and system-level interactions. By using the verbose (-v) flag, it reveals critical edge computing metrics like Maximum Resident Set Size (the peak RAM used) and the number of page faults, which helps you understand the physical hardware footprint of your image analytics code.

```bash
/usr/bin/time -v python sample_img.py
```

Useful fields:

*   CPU vs wall time sanity check
*   Memory footprint
*   Scheduling / contention indicators
*   I/O indicators

***

### 6.2 `cProfile`

cProfile gives function-level CPU breakdown. It runs a deterministic software profiler that counts every single function call and its precise execution time, giving you a detailed breakdown of the internal Python logic. Unlike hardware-based tools, it allows you to save these metrics to a file and interactively sort them to find exactly which line of code, rather than just which hardware resource that cause the delay.

```bash
python -m cProfile sample_img.py
```

Saving profile to file (recommended)
```bash
python -m cProfile -o output.prof sample_img.py
```

Then inspect interactively. Try to explore the various stats that are useful for analysis:
```bash
python -m pstats output.prof
```

***

### 6.3 `perf stat`

This command executes your script while collecting high-level hardware performance counters to provide a quantitative summary of how the CPU handled the workload. By measuring metrics like cache-misses, instructions per cycle (IPC), and context switches (cs), it allows you to determine if your audio processing is limited by raw calculation speed or memory access inefficiencies.

```bash
perf stat -e cycles,instructions,cache-misses,cs,migrations -- python sample_audio.py
```

This reveals:

*   CPU cycles
*   Cache behavior
*   Context switch / migration count

***

### 6.4 `perf record`

This command records exactly which functions are consuming your CPU resources by sampling the script's execution 99 times per second and tracking the sequence of function calls. After the script finishes, the report provides an interactive breakdown that reveals whether your performance bottlenecks lie in image processing, mathematical computations, or library overhead.

```bash
perf record -F 99 -g -- python sample_img.py
perf report
```

Shows where CPU time is spent (functions, call stacks).

***

### 6.5 `pidstat`

This command monitors the real-time resource usage—including CPU, memory, and disk I/O—of your specific script by locating its Process ID (PID) and refreshing the statistics every second. It is particularly useful for observing the dynamic behavior of your code, such as whether memory usage creeps up during image processing or if disk access spikes when exporting header files.

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

### 7.1 Pin to Single Core 0

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

***

# 9. Part D — Scheduling Plan

You will:

*   Build profiling cards
*   Decide deadlines
*   Assign cores
*   Limit CPU quotas if necessary

### Example OS Controls

#### Priority
This command runs your script using the Round-Robin (RR) real-time scheduling policy with a priority level of 50, ensuring it always preempts standard background tasks. It forces the CPU to give your code a dedicated time slice, significantly reducing "jitter" and making your execution latency much more predictable for edge processing.

```bash
sudo chrt --rr 50 python task.py
```

#### Affinity
This command restricts your script to a single, specific CPU core (Core 2), preventing the Linux scheduler from "bouncing" the process between different cores. By isolating the task this way, you minimize cache misses and scheduling overhead, which provides a more consistent and predictable baseline for measuring your audio processing latency.

```bash
taskset -c 2 python sample_audio.py
```

***

# 10. Deliverables

Submit a concise summary showing your ability to measure and control edge performance. Use the following guides to extract the required data.

### 10.1 Bottleneck Insight

Identify why your code is slow and how efficiently it uses the CPU.

* **Top 3 Functions**: Run `python -m cProfile -s tottime sample_img.py`. Look at the `tottime` (total time) column to find the functions consuming the most resources.
* **IPC (Instructions Per Cycle)**: Run `perf stat python sample_img.py`.
* **Guide:** Look for the "insn per cycle" metric.
* *Note: A value < 1.0 often suggests the CPU is stalled waiting for memory (I/O bound), while > 1.5 suggests efficient computation.*

### 10.2 Tail Latency

Analyze if your pipeline is "jittery" or stable.

* **The Table**: Compare **Average** vs. **p99** latency.
* **Guide**: In your script, save frame times to a list and use:
```python
import numpy as np
print(f"Average: {np.mean(latencies)}")
print(f"p99: {np.percentile(latencies, 99)}")

```

### 10.3 Optimization Impact (DL)

Measure the tangible benefit of Quantization on the RPi5.

* **Metrics**: Compare **FPS** and **RAM (RSS)** for `sample_dl.py` (Standard vs. Quantized).
* **Guide**:
* Get **FPS** from the script’s terminal output.
* Get **RAM** by running `pidstat -r 1` or `ps -o rss -p $(pgrep -f sample_dl.py)` while the script is active.

### 10.4 System Control Verification

Prove you can take control of the OS scheduler and network.

* **(1) Real-Time Priority**: Run `chrt -p [PID]` while your script is running under `sudo chrt --rr 50`.
* **Guide**: Ensure the output confirms `policy: SCHED_RR` and `priority: 50`.

* **(2) Network Stress**: Record the **End-to-End latency** before and after applying `tc qdisc`.
* **Guide**: Subtract the "sensor" timestamp from the "processed" timestamp in your `mqtt_log.txt`. You should see an increase of roughly 40ms.

---

# 11. Quick Command Cheatsheet

* Audio Capture: arecord -D hw:2,0 -d 5 -r 16000 -f S16_LE -c 2 test.wav
* Real-Time Execution: sudo chrt --rr 50 taskset -c 2 python task.py
* Hardware Counters: perf stat -e cycles,instructions,cache-misses,cs -- python app.py
* Function Profiling: python -m cProfile -o output.prof app.py
* Resource Monitoring: pidstat -rud -p $(pgrep -f app.py) 1
* Network Stress: sudo tc qdisc add dev eth0 root netem delay 40ms loss 1%
* MQTT Timing: mosquitto_sub -t "lab/e2e/processed" -v | ts '%s%N'

***
