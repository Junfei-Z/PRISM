# 🔐 PRISM: Privacy-Aware Routing for Adaptive Cloud–Edge LLM Inference

[![AAAI 2026](https://img.shields.io/badge/AAAI-2026-blue)](https://aaai.org/conference/aaai/aaai-26/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-brightgreen.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-red.svg)](LICENSE)

> **Privacy-aware Routing for Inference with Semantic Modulation (PRISM)** - A context-aware cloud-edge framework that dynamically balances privacy and inference quality for LLM deployments.

## 📄 Paper

**Title:** PRISM: Privacy-Aware Routing for Adaptive Cloud–Edge LLM Inference via Semantic Sketch Collaboration
**Conference:** AAAI 2026
**Authors:** Junfei Zhan, Haoxun Shen, Zheng Lin, Tengjiao He

---

## 🎯 Overview

PRISM addresses privacy challenges in cloud-based LLM inference through intelligent routing and adaptive privacy protection:

- 🔍 **Sensitivity Profiling**: Computes risk scores and identifies sensitive entities
- 🚦 **Soft Gating Router**: Entropy-regularized neural routing for context-aware execution
- 🛡️ **Adaptive Two-Layer LDP**: Category-aware differential privacy with automatic budget allocation
- 🤝 **Semantic Sketch Collaboration**: Privacy-preserving cloud-edge communication

### Three Execution Modes

| Mode | Privacy | Performance | Use Case |
|------|---------|-------------|----------|
| 🌩️ **Cloud-Only** | ❌ Low | ⚡ Fast | General queries |
| 📱 **Edge-Only** | ✅ High | 🐢 Slower | Sensitive data (medical) |
| 🔄 **Collaborative** | ⚖️ Balanced | 🚀 Optimal | Most queries (tourism, banking) |

---

## 📂 Repository Structure

```
PRISM/
├── Code/
│   ├── edge_detection.py          # 🔍 Sensitivity profiling module
│   ├── soft_gating.py              # 🚦 Entropy-regularized routing
│   ├── two_layer_ldp.py           # 🛡️ Adaptive differential privacy
│   ├── cloud_sketch_generator.py  # ☁️ Cloud-side semantic sketch generation
│   ├── edge_denoising.py          # 📱 Edge-side response refinement
│   ├── prism_pipeline.py          # 🔧 End-to-end PRISM framework
│   ├── windows_energy_monitor.py  # ⚡ Energy consumption monitor
│   ├── few_shot_examples_cloud.txt
│   └── few_shot_examples_edge.txt
├── Dataset/
│   ├── prism_dataset.xlsx         # Evaluation dataset (4 domains)
│   └── route_result.xlsx          # Routing experiment results
├── prism.pdf                      # 📖 Paper
├── Appendix_PRISM.pdf             # 📚 Supplementary materials
└── requirements_prism.txt         # 📦 Dependencies
```

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/Junfei-Z/PRISM.git
cd PRISM

# Install dependencies
pip install -r requirements_prism.txt

# Download spaCy model for entity detection
python -m spacy download en_core_web_sm

# Set OpenAI API key for cloud inference
export OPENAI_API_KEY="your-api-key-here"
```

### Basic Usage

```python
from Code.prism_pipeline import PRISMPipeline

# Initialize PRISM
pipeline = PRISMPipeline(
    edge_model_path="models/phi-3.5-mini",
    cloud_api_key="your-api-key",
    epsilon_total=1.0,  # Privacy budget
    lambda_entropy=0.4  # Routing entropy weight
)

# Process prompt with automatic routing
prompt = "I plan to travel to Tokyo for three days"
result = pipeline.process_prompt_end_to_end(prompt)

print(f"Routing: {result['routing_decision']}")
print(f"Response: {result['edge_refinement']['final_response']}")
```

---

## ⚙️ Configuration

```python
config = {
    "epsilon_total": 1.0,        # 🛡️ Total privacy budget
    "alpha": 0.5,                # ⚖️ Budget allocation parameter
    "lambda_entropy": 0.4,       # 🎲 Entropy regularization coefficient
    "routing_threshold": 0.5,    # 🚦 Sensitivity routing threshold
    "entity_weights": {          # 🏷️ Domain-specific sensitivity weights
        "PERSON": 0.9,
        "LOCATION": 0.6,
        "ORGANIZATION": 0.4,
        "DATE_TIME": 0.3
    }
}
```

---

## 📊 Performance Highlights

| Method | Completion Time (s) | Energy (J) | Quality (1-10) |
|--------|-------------------|-----------|---------------|
| **PRISM** | **7.92** | **687** | **6.88** |
| Uniform LDP | 20.56 | 1708 | 5.72 |
| Selective LDP | 21.22 | 1771 | 5.94 |
| Cloud-Only | 5.13 | 296 | 8.14 |
| Edge-Only | 17.84 | 1574 | 5.09 |

**Key Results:**
- ⚡ **40-50% reduction** in energy/latency vs. baseline LDP methods
- 🛡️ **Strong privacy guarantees** (ε-LDP compliance)
- 📈 **Superior quality** under privacy constraints

---

## 🔬 Evaluation Domains

- 🏥 **Medical**: Demographic data, symptom descriptions
- ✈️ **Tourism**: Travel plans, budgets, destinations
- 💳 **Banking**: Transaction histories, account info
- 📚 **General**: Non-sensitive knowledge queries

---

## 📖 Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{zhan2026prism,
  title     = {PRISM: Privacy-Aware Routing for Adaptive Cloud–Edge LLM Inference via Semantic Sketch Collaboration},
  author    = {Junfei Zhan and Haoxun Shen and Zheng Lin and Tengjiao He},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence (AAAI 2026)},
  year      = {2026}
}
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📧 Contact

For questions or collaborations, please contact:
- Junfei Zhan: [junfeiz@seas.upenn.edu](mailto:junfeiz@seas.upenn.edu)
- Haoxun Shen: [haoxuns@seas.upenn.edu](mailto:haoxuns@seas.upenn.edu)

---

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.
