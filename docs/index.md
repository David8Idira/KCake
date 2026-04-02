# KCake Architecture Documentation

## Overview

KCake integrates two powerful projects:
- **ktransformers** (CPU-GPU heterogeneous optimization)
- **cake** by evilsocket (distributed device clustering)

## System Architecture

```
                    ┌─────────────────────────────────────────┐
                    │              KCake Cluster               │
                    ├─────────────────────────────────────────┤
                    │                                         │
                    │  ┌─────────────────────────────────┐    │
                    │  │       Cluster Manager           │    │
                    │  │  • mDNS Node Discovery          │    │
                    │  │  • Shard Routing & Load Balance │    │
                    │  │  • Weight Streaming & Cache     │    │
                    │  │  • Heartbeat Health Check       │    │
                    │  └─────────────────────────────────┘    │
                    │                  │                       │
                    │  ┌───────────────┴───────────────┐    │
                    │  │                               │        │
                    │  ▼                               ▼        │
                    │  ┌─────────────┐     ┌─────────────────┐  │
                    │  │   Master    │     │     Workers     │  │
                    │  │   Node      │◄───►│   (iOS/Android │  │
                    │  │ (Has Model) │     │  Mac/PC/etc.)  │  │
                    │  └──────┬──────┘     └─────────────────┘  │
                    │         │                                  │
                    │         ▼                                  │
                    │  ┌─────────────────────────────────┐      │
                    │  │     Heterogeneous Scheduler     │      │
                    │  │  • Expert Placement (Hot/Warm/  │      │
                    │  │    Cold)                        │      │
                    │  │  • NUMA-aware Memory            │      │
                    │  │  • Dynamic Migration             │      │
                    │  └─────────────────────────────────┘      │
                    │                  │                       │
                    │         ┌────────┴────────┐               │
                    │         ▼                 ▼               │
                    │  ┌────────────┐    ┌────────────────┐      │
                    │  │  GPU/CUDA  │    │   CPU/NUMA    │      │
                    │  │  (Hot Exp) │    │  (Cold Exp)   │      │
                    │  └────────────┘    └────────────────┘      │
                    │                                         │
                    └─────────────────────────────────────────┘
```

## Core Modules

### 1. `kcake.core` - Inference Engine
- Model loading and sharding
- Token generation
- KV Cache management

### 2. `kcake.heterogeneous` - Hetero Scheduler
- Expert classification (Hot/Warm/Cold)
- NUMA-aware memory allocation
- Dynamic expert migration

### 3. `kcake.cluster` - Cluster Manager
- mDNS-based node discovery
- Shard routing and distribution
- Weight streaming with compression

### 4. `kcake.devices` - Device Abstraction
- CUDA, MPS, Vulkan, CPU backends
- NPU, TPU support (planned)
- Device capability detection

### 5. `kcake.api` - API Layer
- OpenAI-compatible REST API
- Ollama-compatible API
- Streaming response support

## Key Features from ktransformers

1. **Expert Scheduling**: MoE models have experts distributed across devices based on call frequency
2. **NUMA Optimization**: Multi-socket CPU servers get optimized memory access patterns
3. **Quantization**: INT4/INT8 CPU weights with GPTQ GPU support

## Key Features from cake

1. **Zero-config Clustering**: mDNS auto-discovers nodes on the network
2. **Weight Streaming**: Model shards streamed on-demand with zstd compression
3. **Cross-platform**: Runs on iOS, Android, macOS, Linux, Windows

## Data Flow

```
User Request → API Server → Cluster Manager → Hetero Scheduler
                                                      │
                                          ┌───────────┼───────────┐
                                          ▼           ▼           ▼
                                      GPU Device  CPU Device  Remote Device
                                          │           │           │
                                          └───────────┴───────────┘
                                                      │
                                        ◄─────────────┘
                                        │
                        ◄──────────────┘
                        │
                Generated Tokens ← Inference Engine
                        │
                        ▼
                API Response (Stream/Complete)
```

## Usage Modes

### Single Node Mode
```bash
python -m kcake serve --model meta-llama/Llama-3.1-70B
```

### Cluster Mode (Master)
```bash
python -m kcake serve --model meta-llama/Llama-3.1-70B --cluster-key secret123
```

### Cluster Mode (Worker)
```bash
python -m kcake worker --cluster-key secret123 --name iphone-12
```
