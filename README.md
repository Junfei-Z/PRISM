# PRISM: Privacy-Aware Routing for Adaptive Cloud–Edge LLM Inference via Semantic Sketch Collaboration

[![AAAI 2026](https://img.shields.io/badge/AAAI-2026-blue)](https://ojs.aaai.org/index.php/AAAI/article/view/40041)
[![Paper](https://img.shields.io/badge/Paper-PDF-red)](https://ojs.aaai.org/index.php/AAAI/article/view/40041/44002)
[![DOI](https://img.shields.io/badge/DOI-10.1609%2Faaai.v40i33.40041-orange)](https://doi.org/10.1609/aaai.v40i33.40041)
[![Project Page](https://img.shields.io/badge/Project-Page-purple)](https://junfei-z.github.io/prism/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-brightgreen.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**PRISM** is a context-aware cloud–edge inference framework that dynamically balances privacy and inference quality for LLM deployments. It executes in four stages: (1) edge-side entity-level sensitivity profiling, (2) entropy-regularised soft gating to select an execution path, (3) adaptive two-layer local differential privacy for collaborative paths, and (4) cloud–edge semantic sketch collaboration for final response generation.

---

## Paper

**Title:** PRISM: Privacy-Aware Routing for Adaptive Cloud–Edge LLM Inference via Semantic Sketch Collaboration  
**Conference:** AAAI 2026  
**Authors:** Junfei Zhan, Haoxun Shen, Zheng Lin, Tengjiao He  
**DOI:** [10.1609/aaai.v40i33.40041](https://doi.org/10.1609/aaai.v40i33.40041)

| Resource | Link |
|----------|------|
| Paper (PDF) | [AAAI Proceedings](https://ojs.aaai.org/index.php/AAAI/article/view/40041/44002) |
| Project Page | [junfei-z.github.io/prism](https://junfei-z.github.io/prism/) |
| Poster | [AAAI Poster](https://ojs.aaai.org/index.php/AAAI/article/view/40041/49562) |

---

## Overview

PRISM routes each user prompt to one of three execution paths based on its assessed privacy risk:

| Mode | When used | Privacy |
|------|-----------|---------|
| **Cloud-only** | Low-risk prompts | None (sent as-is) |
| **Edge-only** | High-risk prompts | Full (SLM generates locally) |
| **Collaborative** | Medium-risk prompts | Adaptive LDP + sketch refinement |

In collaborative mode, sensitive entities are perturbed with adaptive two-layer LDP before the prompt is sent to the cloud. The cloud LLM generates a *semantic sketch* from the obfuscated prompt; the edge SLM then reconstructs the final response by conditioning on both the original prompt (available locally) and the sketch.

---

## Repository Structure

```
PRISM/
├── Code/
│   ├── edge_detection.py           # Sensitivity profiling (NER + risk scoring)
│   ├── soft_gating.py              # Entropy-regularised soft gating router
│   ├── two_layer_ldp.py            # Adaptive two-layer LDP mechanism
│   ├── cloud_sketch_generator.py   # Cloud-side semantic sketch generation
│   ├── edge_denoising.py           # Edge-side SLM inference (G_edge)
│   ├── prism_pipeline.py           # End-to-end PRISM pipeline (Algorithm 1)
│   ├── windows_energy_monitor.py   # Energy measurement (Windows / NVML)
│   ├── few_shot_examples_cloud.txt # D_cloud: cloud few-shot demonstrations
│   └── few_shot_examples_edge.txt  # D_edge: edge few-shot demonstrations
├── Dataset/
│   ├── prism_dataset.xlsx          # Evaluation dataset (4 domains, 400 prompts)
│   └── route_result.xlsx           # Routing experiment results
├── prism.pdf                       # Paper
├── Appendix_PRISM.pdf              # Supplementary material
└── requirements_prism.txt          # Python dependencies
```

---

## Installation

### 1. Clone and install Python dependencies

```bash
git clone https://github.com/Junfei-Z/PRISM.git
cd PRISM
pip install -r requirements_prism.txt
python -m spacy download en_core_web_lg
```

### 2. Install llama-cpp-python with GPU support

The edge SLM is served via [llama-cpp-python](https://github.com/abetlen/llama-cpp-python). Install the CUDA-enabled build for GPU offloading:

```bash
# CUDA 12.x
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir

# CPU-only (slower, no GPU required)
pip install llama-cpp-python
```

### 3. Download a GGUF edge SLM

The paper evaluates four edge SLMs. Download any GGUF-quantised variant from HuggingFace and place it in the `models/` directory.

| Model | Paper label | HuggingFace repo | Recommended GGUF |
|-------|-------------|------------------|-----------------|
| Phi-3.5-mini-3.5B | S1 | [bartowski/Phi-3.5-mini-instruct-GGUF](https://huggingface.co/bartowski/Phi-3.5-mini-instruct-GGUF) | `Phi-3.5-mini-instruct-Q6_K_L.gguf` |
| Qwen1.5-1.8B-Chat | S2 | [Qwen/Qwen1.5-1.8B-Chat-GGUF](https://huggingface.co/Qwen/Qwen1.5-1.8B-Chat-GGUF) | `qwen1_5-1_8b-chat-q4_k_m.gguf` |
| StableLM-2-Zephyr-1.6B | S3 | [second-state/stablelm-2-zephyr-1.6b-GGUF](https://huggingface.co/second-state/stablelm-2-zephyr-1.6b-GGUF) | `stablelm-2-zephyr-1_6b-Q4_K_M.gguf` |
| TinyLLaMA-1.1B | S4 | [TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF](https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF) | `tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf` |

```bash
mkdir -p models
# Example: TinyLLaMA
wget -P models https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
```

### 4. Set cloud API key

```bash
export OPENAI_API_KEY="your-api-key"
```

---

## Quick Start

### Python API

```python
from Code.prism_pipeline import PRISMPipeline

pipeline = PRISMPipeline(
    slm_model_path="models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
    slm_n_gpu_layers=32,   # set to 0 for CPU-only
    epsilon_total=2.0,     # total LDP privacy budget
    alpha=0.5,             # budget allocation parameter
    lambda_entropy=0.4,    # entropy regularisation weight
)

result = pipeline.process_prompt_end_to_end(
    "I plan to travel solo to Tokyo for three days; help me design my itinerary."
)
print(result["routing"]["mode"])              # e.g. "collaborative"
print(result["edge_refinement"]["final_response"])
```

### Command-line demo

```bash
cd Code
python prism_pipeline.py \
    --slm-model ../models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
    --n-gpu-layers 32 \
    --epsilon 2.0
```

Or via environment variable:

```bash
export PRISM_SLM_MODEL_PATH=../models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
python prism_pipeline.py
```

### Edge-only SLM test

```bash
cd Code
python edge_denoising.py \
    --model ../models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
    --n-gpu-layers 32
```

---

## Inference Parameters

The following defaults match the experimental setup in the paper (edge device: NVIDIA RTX 3070 laptop GPU, Windows 10):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `slm_n_gpu_layers` | 32 | Transformer layers offloaded to GPU |
| `slm_n_ctx` | 2048 | Context window (tokens) |
| `slm_n_batch` | 512 | Prompt batch size |
| `slm_temperature` | 0.7 | Sampling temperature |
| `slm_top_p` | 0.9 | Nucleus sampling threshold |
| `slm_max_tokens` | 512 | Max new tokens per call |
| `epsilon_total` | 2.0 | Total LDP privacy budget |
| `alpha` | 0.5 | Category/value budget split |
| `lambda_entropy` | 0.4 | Entropy regularisation weight |

---

## Performance

Results from Table 1 of the paper (cloud LLM: GPT-4o, edge SLM: Qwen1.5-1.8B):

| Method | Completion Time (s) | Energy (J) | Inference Quality |
|--------|-------------------|------------|------------------|
| **PRISM** | **7.92** | **687** | **6.88** |
| Uniform LDP | 20.56 | 1708 | 5.72 |
| Selective LDP | 21.22 | 1771 | 5.94 |
| Cloud-Only | 5.13 | 296 | 8.14 |
| Edge-Only | 17.84 | 1574 | 5.09 |

PRISM achieves 40–50% lower latency and energy than uniform/selective LDP baselines while maintaining strong privacy guarantees.

---

## Evaluation Dataset

The dataset covers four domains (100 prompts each):

- **Tourism** – travel plans, budgets, destinations, group compositions
- **Medical** – symptoms, demographics, diagnoses (partially adapted from PrivacyRestore)
- **Banking** – transaction histories, account identifiers, dispute requests
- **General knowledge** – non-sensitive factual queries (from MT-Bench)

---

## Citation

```bibtex
@inproceedings{zhan2026prism,
  title     = {PRISM: Privacy-Aware Routing for Adaptive Cloud-Edge
               LLM Inference via Semantic Sketch Collaboration},
  author    = {Zhan, Junfei and Shen, Haoxun and Lin, Zheng and He, Tengjiao},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  volume    = {40},
  number    = {33},
  pages     = {28150--28158},
  year      = {2026},
  doi       = {10.1609/aaai.v40i33.40041},
  url       = {https://ojs.aaai.org/index.php/AAAI/article/view/40041}
}
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
