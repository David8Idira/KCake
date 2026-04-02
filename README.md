# KCake

<div align="center">

![KCake Logo](docs/logo.png)

**异构设备集群化超大规模模型推理引擎**

*让闲置的手机、电脑等设备成为超大规模模型的算力节点*

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-green.svg)](https://python.org)
[![Rust](https://img.shields.io/badge/rust-1.70%2B-orange.svg)](https://rust-lang.org)
[![Stars](https://img.shields.io/github/stars/liq_idira/KCake?style=social)](https://github.com/liq_idira/KCake/stargazers)

</div>

---

## 🎯 项目愿景

KCake 是一个融合 **ktransformers** (高性能CPU-GPU异构计算) 与 **cake** (evilsocket) (分布式设备集群推理) 优势的**超大规模模型推理引擎**。

**核心理念**：将闲置的移动设备（手机、平板）、个人电脑、服务器等异构设备组成集群，以协同运行超出单一设备能力范围的超大规模语言模型。

---

## ✨ 核心特性

### 🔧 异构计算优化 (源自 ktransformers)

- **CPU-GPU 混合推理**：智能调度热门专家到GPU、冷门专家到CPU/Disk
- **NUMA-aware 内存管理**：针对多插槽服务器优化
- **AMX/AVX 加速**：Intel AMX、AVX512/AVX2 硬件加速支持
- **量化支持**：INT4/INT8 CPU量化权重，GPTQ GPU支持
- **MoE 专家调度**：高效混合专家模型推理

### 🌐 分布式集群 (源自 cake)

- **零配置集群**：mDNS 自动发现设备
- **异构设备支持**：手机(iOS/Android)、电脑(macOS/Linux/Windows)、服务器
- **Transformer 分片**：按层分片到不同设备
- **权重流式传输**：压缩传输(zstd) + CRC校验
- **设备缓存**：已传输权重本地缓存复用

### 🔌 兼容性

- **OpenAI 兼容 API**：无缝接入现有应用
- **Ollama 兼容 API**：兼容 Ollama 模型格式
- **RESTful API**：标准化接口设计
- **Web UI**：内置可视化界面
- **TUI 客户端**：终端交互界面

### 💻 任意平台支持

| 平台 | CUDA | Metal | Vulkan | CPU | NPU | TPU |
|------|------|-------|--------|-----|-----|-----|
| Linux | ✅ | - | ✅ | ✅ | ✅ | 🔜 |
| macOS | - | ✅ | - | ✅ | - | - |
| Windows | ✅ | - | ✅ | ✅ | - | - |
| iOS | - | ✅ | - | ✅ | - | - |
| Android | ✅ | - | ✅ | ✅ | ✅ | - |

---

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                         KCake Cluster                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │  iPhone  │  │  Android │  │   Mac    │  │  Linux   │       │
│  │   Phone  │  │   Phone  │  │   PC     │  │  Server  │       │
│  │  (Edge)  │  │  (Edge)  │  │  (Hybrid)│  │  (Core)  │       │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘       │
│       │              │              │              │            │
│       └──────────────┴──────────────┴──────────────┘            │
│                              │                                   │
│                    ┌─────────▼─────────┐                        │
│                    │   Cluster Layer   │                        │
│                    │   (mDNS Discovery) │                        │
│                    │   (Weight Router) │                        │
│                    └─────────┬─────────┘                        │
│                              │                                   │
│       ┌──────────────────────┼──────────────────────┐           │
│       │                      │                      │           │
│  ┌────▼────┐           ┌────▼────┐           ┌────▼────┐       │
│  │Hetero    │           │Inference│           │ Model   │       │
│  │Scheduler │           │ Engine  │           │ Loader  │       │
│  │(Experts) │           │(Tokens) │           │(GGUF)   │       │
│  └────┬────┘           └────┬────┘           └────┬────┘       │
│       │                      │                      │           │
│       └──────────────────────┼──────────────────────┘           │
│                              │                                   │
│                    ┌─────────▼─────────┐                        │
│                    │    API Layer      │                        │
│                    │ (OpenAI/Ollama)   │                        │
│                    └─────────┬─────────┘                        │
│                              │                                   │
│       ┌──────────────────────┼──────────────────────┐           │
│       │                      │                      │           │
│  ┌────▼────┐           ┌─────▼─────┐           ┌────▼────┐     │
│  │  Web UI │           │   REST    │           │   TUI   │     │
│  │(Gradio) │           │   API     │           │(Terminal│     │
│  └─────────┘           └───────────┘           └─────────┘     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🚀 快速开始

### 安装

```bash
# 克隆项目
git clone https://github.com/liq_idira/KCake.git
cd KCake

# 安装 Python 依赖
pip install -r requirements.txt

# 安装 Rust 组件 (可选，用于高性能推理)
cargo build --release
```

### 启动集群节点

```bash
# 在主服务器上启动 (拥有完整模型)
python -m kcake serve --model meta-llama/Llama-3.1-70B --cluster-key mysecret

# 在闲置设备上启动为 worker
python -m kcake worker --cluster-key mysecret --name iphone-12
python -m kcake worker --cluster-key mysecret --name android-pixel
python -m kcake worker --cluster-key mysecret --name macbook-pro
```

### API 调用

```bash
# OpenAI 兼容格式
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-70B",
    "messages": [{"role": "user", "content": "解释量子计算"}]
  }'

# Ollama 兼容格式
curl http://localhost:8000/api/generate \
  -d '{
    "model": "llama3.1",
    "prompt": "解释量子计算"
  }'
```

---

## 📦 模块设计

### `kcake.core` - 核心推理引擎

- 模型加载与分片
- Token 生成
- KV Cache 管理

### `kcake.heterogeneous` - 异构调度器

- CPU-GPU 专家调度
- NUMA-aware 内存分配
- 量化权重管理

### `kcake.cluster` - 集群管理

- mDNS 设备发现
- 权重路由与传输
- 分片调度策略

### `kcake.devices` - 设备适配层

- CUDA/MPS/Metal/Vulkan/CPU 后端
- NPU/TPU 抽象接口
- 设备能力探测

### `kcake.api` - API 服务

- OpenAI 兼容接口
- Ollama 兼容接口
- 流式响应支持

### `kcake.ui` - 用户界面

- Gradio Web UI
- TUI 终端界面

---

## 📊 性能对比

| 配置 | 模型 | 吞吐 | 备注 |
|------|------|------|------|
| 单机 8×L20 | DeepSeek-R1 (FP8) | 227 tok/s | ktransformers 基线 |
| 集群 (1服务器+2手机) | Llama-3.1-70B | ~50 tok/s | 概念验证 |
| 单 MacBook M3 | Llama-3.1-8B | 30 tok/s | Metal 加速 |

---

## 🔬 技术融合

| 特性 | ktransformers | cake | KCake |
|------|--------------|------|-------|
| CPU-GPU 异构 | ✅ | ❌ | ✅ |
| 分布式集群 | ❌ | ✅ | ✅ |
| MoE 专家调度 | ✅ | ❌ | ✅ |
| 手机/移动端 | ❌ | ✅ | ✅ |
| NUMA 优化 | ✅ | ❌ | ✅ |
| mDNS 发现 | ❌ | ✅ | ✅ |
| OpenAI API | ❌ | ✅ | ✅ |
| Ollama API | ❌ | ❌ | ✅ |
| 量化优化 | ✅ | ❌ | ✅ |
| 多模态 | ❌ | ✅ | 🔜 |

---

## 📄 许可证

Apache License 2.0 - 详见 [LICENSE](LICENSE) 文件

---

## 🙏 致谢

- [ktransformers](https://github.com/kvcache-ai/ktransformers) - 高性能 CPU-GPU 异构推理框架
- [cake](https://github.com/evilsocket/cake) - 分布式推理服务器

---

*KCake - 让每一台设备都能参与 AI 推理*
