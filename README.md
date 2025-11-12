# PRISM: Privacy-Aware Routing for Adaptive Cloud–Edge LLM Inference via Semantic Sketch Collaboration

This repository contains the implementation of PRISM, a privacy-aware cloud-edge collaborative inference framework designed to balance privacy protection, computational efficiency, and response quality in LLM deployments.

## 📄 Paper Information
**Title:** PRISM: Privacy-Aware Routing for Adaptive Cloud–Edge LLM Inference via Semantic Sketch Collaboration 
**Conference:** AAAI Conference on Artificial Intelligence (AAAI 2026)  
**Authors:**  Junfei Zhan\, Haoxun Shen\, Zheng Lin, Tengjiao He  


## Abstract
PRISM addresses privacy challenges in cloud-based LLM inference through a novel cloud-edge collaborative framework. The system performs sensitivity profiling to intelligently route user prompts, applies adaptive two-layer differential privacy for entity protection, and uses semantic sketch-based collaboration to balance privacy, utility, and efficiency.

## Key Features

1. **Sensitivity Profiling**: Computes risk score R(P) and sensitivity mask d for intelligent routing
2. **Soft Gating**: Entropy-regularized neural routing mechanism (π = softmax(f_θ(z)), λ = 0.4)
3. **Adaptive Two-Layer LDP**: Category-aware differential privacy with automatic budget allocation
4. **Semantic Sketch Collaboration**: Privacy-preserving cloud-edge communication using structured sketches
5. **Few-Shot Learning**: Domain-specific demonstration sets for cloud (P*, S) and edge (P, S, R) components

## System Architecture

PRISM is a privacy-aware cloud-edge inference framework consisting of three components:
- **User**: Initiates prompts and receives final responses
- **Local Edge Device**: Hosts an SLM (Small Language Model) and privacy protection modules
- **Remote Cloud Server**: Provides access to an LLM (Large Language Model)

### Inference Process
1. **User sends prompt** to the edge device
2. **Sensitivity Profiling** (edge_detection.py) computes risk score R(P) and sensitivity mask d
3. **Soft Gating** (soft_gating.py) uses [R(P); d] feature vector to select execution path
4. **Three execution paths available**:
   - **Edge-only**: High privacy sensitivity (e.g., medical queries)
   - **Cloud-only**: Low privacy risk (e.g., general knowledge)
   - **Collaborative**: Medium risk with cloud-edge cooperation
5. **For collaborative cases**:
   - **Adaptive Two-Layer LDP** perturbs sensitive entities
   - **Cloud-Edge Semantic Sketch Collaboration** generates abstract response on cloud
   - **Edge refinement** produces final response returned to user

### Routing Strategies
- **Medical queries** → **Edge-only** (high privacy sensitivity)  
- **Common/General queries** → **Cloud-only** (low privacy risk)
- **Tourism & Banking queries** → **Collaborative** (medium risk, sketch-based collaboration)

## Installation

### Basic Installation

```bash
# Clone the repository
git clone [repository-url]
cd prism-framework

# Install dependencies
pip install -r requirements.txt

# Download spaCy language model (required for entity detection)
python -m spacy download en_core_web_sm

# Set OpenAI API key (for cloud sketch generation)
export OPENAI_API_KEY="your-api-key-here"

# Create results directory for experiments
mkdir results
```

### GPU Support (Optional - For Local Edge Models)

For optimal performance with local edge models, install `llama-cpp-python` with CUDA support:

#### Prerequisites (Windows)
1. **NVIDIA CUDA Toolkit** (11.8 or 12.x recommended):
   - Download from [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads)
   - Install and add CUDA to your PATH

2. **Visual Studio Build Tools** (required for CMake compilation):
   - Download [Visual Studio Build Tools](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022)
   - Install with "C++ build tools" workload
   - **OR** Install full Visual Studio Community Edition

3. **CMake** (if not included with Visual Studio):
   - Download from [CMake Downloads](https://cmake.org/download/)
   - Add CMake to your PATH

#### Install llama-cpp-python with CUDA

```bash
# Method 1: Install with CUDA support (Windows)
set CMAKE_ARGS=-DLLAMA_CUBLAS=on
set FORCE_CMAKE=1
pip install llama-cpp-python --no-cache-dir

# Method 2: Install pre-built wheel (if available)
pip install llama-cpp-python[cuda]

# Method 3: Install CPU-only version (fallback)
pip install llama-cpp-python
```

#### Prerequisites (Linux)
```bash
# Install CUDA (Ubuntu/Debian)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda

# Install build tools
sudo apt-get install build-essential cmake

# Install llama-cpp-python with CUDA
CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python --no-cache-dir
```

#### Verify GPU Support
```python
from llama_cpp import Llama

# Test CUDA availability
llm = Llama(
    model_path="./models/your-model.gguf",
    n_gpu_layers=32,  # Number of layers to offload to GPU
    verbose=True
)
# Should show GPU layers being loaded
```

### Model Files Setup

Place your GGUF model files in the `./models/` directory:

```bash
mkdir models
# Download and place your model files:
# - Phi-3.5-mini-instruct (Q6_K quantization)
# - Qwen1.5-1.8B-chat (Q6_K quantization)  
# - StableLM-2-Zephyr-1.6B (Q6_K quantization)
# - TinyLLaMA-1.1B-chat (Q6_K quantization)
```

### Troubleshooting

**Common Issues:**
- **"Microsoft Visual C++ 14.0 is required"**: Install Visual Studio Build Tools
- **CUDA not found**: Ensure CUDA is installed and in your PATH
- **CMake errors**: Verify CMake is installed and accessible
- **Out of memory**: Reduce `n_gpu_layers` or use smaller quantized models

**Performance Tips:**
- Use Q6_K quantization for optimal quality/speed balance
- Set `n_gpu_layers` based on your GPU memory (typically 20-40 layers)
- Monitor GPU memory usage with `nvidia-smi`
- Ensure sufficient RAM for model loading (8GB+ recommended)

## Quick Start

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (optional, for local edge models)
- OpenAI API key (for cloud integration)

### File Structure
```
prism-framework/
├── soft_gating.py              # Entropy-regularized routing module
├── prism_pipeline.py           # Main PRISM framework integration
├── edge_detection.py           # Sensitivity profiling (risk score R(P) + mask d generation)
├── two_layer_ldp.py           # Adaptive two-layer differential privacy
├── cloud_sketch_generator.py  # Cloud-side semantic sketch generation
├── edge_denoising.py          # Edge-side response refinement
├── few_shot_examples_cloud.txt # Cloud demonstration set (P*, S)
├── few_shot_examples_edge.txt  # Edge demonstration set (P, S, R)
└── models/
    └── soft_gating_pretrained.pth # Pre-trained soft gating weights
```

## Usage

### Basic Example

```python
from prism_pipeline import PRISMPipeline
from soft_gating import RoutingMode

# Initialize PRISM pipeline
pipeline = PRISMPipeline(
    edge_model_path="models/phi-3.5-mini",
    cloud_api_key="your-api-key",  # or set OPENAI_API_KEY env var
    epsilon_total=1.0,
    lambda_entropy=0.4
)

# Process a user prompt - automatic routing based on sensitivity
prompt = "I plan to travel solo to Tokyo for three days"
result = pipeline.process_prompt_end_to_end(prompt)
print(f"Routing: {result['routing_decision']}")
print(f"Response: {result['edge_refinement']['final_response']}")
```


### Running the Framework

```bash
# Initialize and test core components
python soft_gating.py              # Initialize soft gating module
python edge_detection.py           # Initialize sensitivity profiling
python two_layer_ldp.py           # Initialize two-layer LDP
python cloud_sketch_generator.py  # Initialize cloud generator (requires API key)
python edge_denoising.py          # Initialize edge denoising

# Full pipeline 
python prism_pipeline.py          # End-to-end PRISM framework
python test_prism_integration.py  # Integration verification

# Experimental evaluation
python train_soft_gating.py       # Train soft gating module
python model_comparison_experiment.py  # Model evaluation experiments
python privacy_budget_experiment_windows.py  # Privacy budget analysis
```

## Core Modules

1. **soft_gating.py**: Entropy-regularized neural routing mechanism with feature extraction
2. **edge_detection.py**: Sensitivity profiling module (generates risk score R(P) and sensitivity mask d)
3. **two_layer_ldp.py**: Adaptive two-layer local differential privacy with budget allocation
4. **cloud_sketch_generator.py**: GPT-4o based semantic sketch generation with few-shot learning
5. **edge_denoising.py**: Edge-side sketch refinement and response synthesis
6. **prism_pipeline.py**: End-to-end PRISM framework integration


## Configuration

Adjust privacy and routing parameters in the configuration:

```python
config = {
    "epsilon_total": 1.0,        # Total privacy budget
    "alpha": 0.5,                # Budget allocation parameter for two-layer LDP
    "lambda_entropy": 0.4,       # Entropy regularization coefficient
    "routing_threshold": 0.5,     # Sensitivity threshold for routing decisions
    "entity_weights": {           # Domain-specific sensitivity weights
        "PERSON": 0.9,
        "LOCATION": 0.6,
        "ORGANIZATION": 0.4,
        "DATE_TIME": 0.3
    },
    "few_shot_examples": {
        "cloud_file": "few_shot_examples_cloud.txt",
        "edge_file": "few_shot_examples_edge.txt"
    }
}
```

## Evaluation Framework

The PRISM implementation includes comprehensive evaluation capabilities:

### Dataset Domains
- **Tourism Planning**: User identities, travel budgets, destinations
- **Medical Consultation**: Demographic attributes, symptom descriptions  
- **Banking Services**: Transaction histories, account identifiers
- **General Knowledge**: Non-sensitive queries for baseline comparison

### Evaluation Metrics
- **Inference Quality**: GPT-4o-based scoring (1-10 scale) for relevance and coherence
- **Energy Consumption**: Windows-based power monitoring in Joules
- **Completion Time**: End-to-end latency measurement
- **Privacy Preservation**: ε-LDP compliance scoring

### Performance Characteristics
- **Privacy Protection**: Formal ε-LDP guarantees with adaptive two-layer budget allocation
- **Intelligent Routing**: Entropy-regularized soft gating for context-aware execution path selection
- **Utility Preservation**: Semantic sketch collaboration maintains response quality
- **Efficiency**: Edge-cloud collaboration reduces end-to-end latency
- **Scalability**: Supports multiple domains (Tourism, Medical, Banking, General)

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{zhan2026prism,
  title     = {PRISM: Privacy-Aware Routing for Adaptive Cloud–Edge LLM Inference via Semantic Sketch Collaboration},
  author    = {Junfei Zhan and Haoxun Shen and Zheng Lin and Tengjiao He},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence (AAAI 2026)},
  year      = {2026},
}

```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or collaborations, please contact: [contact information]

## Acknowledgments


This work was supported by [funding information].





