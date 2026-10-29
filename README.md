# SmolLM2-135M Local Inference Engine (tiny.py)
# **Abstract**

_This repository houses a lightweight, localized command-line interface (CLI) inference script designed to execute instruction-tuned causal language models. Specifically, it implements the HuggingFaceTB/SmolLM2-135M-Instruct architecture—a compact 135-parameter transformer network optimized for resource-constrained edge environments. The implementation utilizes explicit CPU allocation with single-precision floating-point arithmetic (torch.float32) to ensure reproducible execution independent of dedicated hardware accelerators._


The underlying system relies on the Hugging Face transformers and PyTorch libraries. Key algorithmic features include:

    Explicit Device Mapping: Forces model execution onto the host CPU to circumvent hardware dependencies and preserve accelerator VRAM for larger parallel tasks.

    Stateful Dialogue Management: Maintains conversational context incrementally by appending role-based message dictionaries (system, user, assistant).

    Chat Template Formalization: Dynamically compiles structured message buffers into serialized input tokens via the tokenizer's native chat template API.

    Stochastic Decoding Strategy: Incorporates nucleus sampling (top_p = 0.9) combined with temperature scaling (temperature = 0.7) to balance exploratory lexical diversity and deterministic coherence.

### System Requirements & Dependencies

To execute this script, ensure the following Python packages are installed within your environment:
`pip install torch transformers`
### Execution Protocol
1. Clone or download the script as tiny.py.

2. Initialize the execution environment from your terminal:
`python tiny.py`

3. Upon successful weight instantiation, interact with the model via the interactive prompt. Terminate the session by inputting exit or quit.
