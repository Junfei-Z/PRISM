"""
Edge-Side Denoising and Reintegration for PRISM Framework

Implements G_edge from Algorithm 1 of the PRISM paper:
  - Edge-only mode  : R̂ = G_edge(P)
  - Collaborative   : R̂ = G_edge(C_edge),  C_edge = [D_edge, (P, S, _)]

Inference runs locally via llama-cpp-python using GGUF-quantised SLMs.
Supported models (any GGUF-quantised variant):
  - TinyLLaMA-1.1B       (S4 in paper)
  - Qwen1.5-1.8B-Chat    (S2 in paper)
  - StableLM-2-Zephyr-1.6B (S3 in paper)
  - Phi-3.5-mini-3.5B    (S1 in paper)
"""

import re
import os
import logging
from typing import Dict, List, Tuple, Optional

from llama_cpp import Llama


class EdgeDenoisingReintegrator:
    """
    Edge-side SLM component of the PRISM framework.

    Loads a GGUF-quantised small language model via llama-cpp-python and uses
    it for both direct edge-only generation (G_edge(P)) and collaborative
    sketch refinement (G_edge(C_edge)).  Few-shot demonstrations from D_edge
    are prepended to every prompt to guide the model.

    Parameters
    ----------
    model_path : str
        Path to the GGUF model file, e.g.
        ``"models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"``.
    n_gpu_layers : int
        Transformer layers to offload to GPU.  32 matches the RTX 3070
        configuration used in the paper experiments.
    n_ctx : int
        Context window length in tokens (default 2048).
    n_batch : int
        Batch size for prompt processing (default 512).
    temperature : float
        Sampling temperature (default 0.7).
    top_p : float
        Nucleus sampling threshold (default 0.9).
    max_tokens : int
        Maximum new tokens per generation call (default 512).
    examples_file : str
        Path to the few-shot demonstration file D_edge.
    """

    def __init__(
        self,
        model_path: str,
        n_gpu_layers: int = 32,
        n_ctx: int = 2048,
        n_batch: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 512,
        examples_file: str = "few_shot_examples_edge.txt",
    ):
        self.logger = logging.getLogger("EdgeDenoisingReintegrator")

        self.model_path = model_path
        self.n_gpu_layers = n_gpu_layers
        self.n_ctx = n_ctx
        self.n_batch = n_batch
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens

        self.logger.info(f"Loading edge SLM: {model_path}")
        self.llm = Llama(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            n_batch=n_batch,
            main_gpu=0,
            offload_kqv=True,
            verbose=False,
        )
        self.logger.info("Edge SLM loaded successfully")

        self.examples = self._load_few_shot_examples(examples_file)

        # Patterns for obfuscated entity placeholders produced by two_layer_ldp.py
        self.placeholder_patterns = [
            r'\[([A-Z_]+)\]_(\w+)',   # [PERSON]_Alice
            r'\[([A-Z_]+)\]',         # [PERSON]
            r'\[MASKED_([A-Z_]+)\]',  # [MASKED_PERSON]
            r'\[XXX\]',               # generic mask
            r'<([A-Z_]+)>',           # <PERSON>
        ]

    # ------------------------------------------------------------------
    # Public API  (Algorithm 1 in the paper)
    # ------------------------------------------------------------------

    def generate_direct_response(self, prompt: str) -> str:
        """
        Edge-only generation: R̂ = G_edge(P).

        The SLM receives the original, unperturbed prompt directly and
        generates the full response without any cloud involvement.

        Parameters
        ----------
        prompt : str
            Original user prompt P.

        Returns
        -------
        str
            Generated response R̂.
        """
        category = self._detect_category(prompt)
        llm_prompt = self._build_direct_prompt(prompt, category)
        response = self._run_inference(llm_prompt)
        self.logger.info(
            f"Edge-only generation complete ({category}): {len(response)} chars"
        )
        return response

    def refine_sketch(
        self,
        sketch: str,
        original_prompt: str,
        entities: List[Tuple[str, int, int, str]],
        protected_entities: List[Tuple[str, int, int, str]],
        category: str,
    ) -> str:
        """
        Collaborative sketch refinement: R̂ = G_edge(C_edge).

        The SLM conditions on both the original prompt P (available locally)
        and the cloud-generated semantic sketch S to reconstruct a coherent,
        privacy-preserving final response.  Private entities that were
        obfuscated before cloud transmission are restored in the sketch
        before being passed to the SLM.

        Parameters
        ----------
        sketch : str
            Semantic sketch S returned by the cloud LLM.
        original_prompt : str
            Original, unperturbed user prompt P.
        entities : list
            All entities detected during sensitivity profiling.
        protected_entities : list
            Entities that were LDP-obfuscated before cloud upload.
        category : str
            Detected domain (Tourism / Medical / Banking / Common).

        Returns
        -------
        str
            Final refined response R̂.
        """
        self.logger.info(
            f"Refining {category} sketch "
            f"({len(protected_entities)} protected entities)"
        )

        # Restore obfuscated entity placeholders so the SLM receives clean
        # semantic content aligned with the original prompt.
        denoised_sketch = self._replace_obfuscated_entities(
            sketch, original_prompt, entities, protected_entities
        )

        llm_prompt = self._build_refinement_prompt(
            denoised_sketch, original_prompt, category
        )
        response = self._run_inference(llm_prompt)

        # Append mandatory medical disclaimer when required.
        if category == "Medical" and "disclaimer" not in response.lower():
            response += (
                "\n\n*This information is for reference only and does not "
                "replace professional medical advice.*"
            )

        self.logger.info(f"Refinement complete: {len(response)} chars")
        return response

    # ------------------------------------------------------------------
    # SLM inference
    # ------------------------------------------------------------------

    def _run_inference(self, prompt: str) -> str:
        """Call the loaded SLM and return the generated text."""
        output = self.llm(
            prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            echo=False,
            stop=["###", "\n\n\n"],
        )
        return output["choices"][0]["text"].strip()

    # ------------------------------------------------------------------
    # Prompt construction  (C_edge = [D_edge, (P, S, _)])
    # ------------------------------------------------------------------

    def _build_direct_prompt(self, user_prompt: str, category: str) -> str:
        """
        Construct a few-shot prompt for edge-only generation G_edge(P).

        Each shot shows a (Prompt -> Response) pair from D_edge.
        """
        domain_examples = self.examples.get(category, self.examples.get("Common", []))

        shots = []
        for ex in domain_examples[:2]:
            shots.append(
                f"### Prompt:\n{ex['prompt']}\n"
                f"### Response:\n{ex['response']}"
            )

        system = (
            "You are a helpful assistant running on an edge device. "
            "Generate a fluent, context-specific response to the user prompt."
        )
        return (
            f"{system}\n\n"
            + "\n\n".join(shots)
            + f"\n\n### Prompt:\n{user_prompt}\n### Response:\n"
        )

    def _build_refinement_prompt(
        self,
        sketch: str,
        original_prompt: str,
        category: str,
    ) -> str:
        """
        Construct a few-shot prompt for sketch refinement G_edge(C_edge).

        Each shot shows a (Prompt, Sketch -> Response) triple from D_edge,
        following the demonstration set format described in the paper.
        The model uses P to recover private semantics and S as structural
        guidance (per the prompting guidelines in few_shot_examples_edge.txt).
        """
        domain_examples = self.examples.get(category, self.examples.get("Common", []))

        shots = []
        for ex in domain_examples[:2]:
            if "sketch" in ex:
                shots.append(
                    f"### Original Prompt:\n{ex['prompt']}\n"
                    f"### Sketch:\n{ex['sketch']}\n"
                    f"### Response:\n{ex['response']}"
                )

        system = (
            "You are a helpful assistant running on an edge device. "
            "Use the original prompt to recover personalized details lost during "
            "privacy protection, and use the sketch as structural guidance. "
            "Produce a fluent, context-specific final response."
        )
        return (
            f"{system}\n\n"
            + "\n\n".join(shots)
            + f"\n\n### Original Prompt:\n{original_prompt}\n"
            f"### Sketch:\n{sketch}\n"
            f"### Response:\n"
        )

    # ------------------------------------------------------------------
    # Entity denoising helpers
    # ------------------------------------------------------------------

    def _replace_obfuscated_entities(
        self,
        sketch: str,
        original_prompt: str,
        entities: List[Tuple[str, int, int, str]],
        protected_entities: List[Tuple[str, int, int, str]],
    ) -> str:
        """Replace LDP-obfuscated placeholders in the sketch with original values."""
        denoised = sketch

        entity_map: Dict[str, List[str]] = {}
        for entity_type, _, _, entity_text in protected_entities:
            entity_map.setdefault(entity_type, []).append(entity_text)

        for pattern in self.placeholder_patterns:
            for match in re.finditer(pattern, denoised):
                full_match = match.group(0)
                entity_type = match.group(1) if match.lastindex and match.lastindex >= 1 else None
                replacement = self._find_replacement(entity_type, entity_map, original_prompt)
                if replacement:
                    denoised = denoised.replace(full_match, replacement, 1)
                    self.logger.debug(f"Denoised: {full_match} -> {replacement}")

        return denoised

    def _find_replacement(
        self,
        entity_type: Optional[str],
        entity_map: Dict[str, List[str]],
        original_prompt: str,
    ) -> Optional[str]:
        """Return the original entity value for an obfuscated placeholder."""
        if entity_type and entity_type in entity_map:
            originals = entity_map[entity_type]
            if originals:
                return originals[0]

        # Heuristic fallback: find the entity type near a keyword in the prompt.
        if entity_type:
            type_keywords = {
                "PERSON": ["name", "patient", "customer", "user"],
                "LOCATION": ["city", "place", "destination", "location"],
                "ORGANIZATION": ["bank", "hospital", "company", "organization"],
                "DATE_TIME": ["age", "date", "time", "year", "day"],
            }
            for keyword in type_keywords.get(entity_type, []):
                if keyword in original_prompt.lower():
                    words = original_prompt.split()
                    for i, word in enumerate(words):
                        if keyword in word.lower() and i < len(words) - 1:
                            return words[i + 1]
        return None

    # ------------------------------------------------------------------
    # Category detection
    # ------------------------------------------------------------------

    def _detect_category(self, prompt: str) -> str:
        """Infer domain category from prompt keywords."""
        p = prompt.lower()
        if any(w in p for w in ["travel", "trip", "visit", "destination", "itinerary", "tour"]):
            return "Tourism"
        if any(w in p for w in ["patient", "symptom", "diagnosis", "treatment", "medical", "doctor"]):
            return "Medical"
        if any(w in p for w in ["bank", "account", "dispute", "charge", "transfer", "payment", "card"]):
            return "Banking"
        return "Common"

    # ------------------------------------------------------------------
    # Few-shot demonstration set loading
    # ------------------------------------------------------------------

    def _load_few_shot_examples(self, examples_file: str) -> Dict[str, List[Dict]]:
        """
        Parse the few-shot demonstration file D_edge.

        Accepts both plain ``Prompt:`` labels and the annotated
        ``Prompt (P):`` / ``Sketch (S):`` / ``Response (R̂):`` format used
        in few_shot_examples_edge.txt.
        """
        examples: Dict[str, List[Dict]] = {
            "Tourism": [], "Medical": [], "Banking": [], "Common": []
        }

        try:
            if not os.path.exists(examples_file):
                self.logger.warning(
                    f"Few-shot examples file not found: {examples_file}. "
                    "Using built-in defaults."
                )
                return self._default_examples()

            with open(examples_file, "r", encoding="utf-8") as fh:
                content = fh.read()

            current_domain: Optional[str] = None
            current_example: Dict = {}

            for line in content.splitlines():
                line = line.strip()

                if "TOURISM DOMAIN" in line.upper():
                    current_domain = "Tourism"
                elif "MEDICAL DOMAIN" in line.upper():
                    current_domain = "Medical"
                elif "BANKING DOMAIN" in line.upper():
                    current_domain = "Banking"
                elif "GENERAL" in line.upper() or "COMMON" in line.upper():
                    current_domain = "Common"
                elif re.match(r'Prompt\s*(?:\([^)]*\))?:', line) and current_domain:
                    if current_example:
                        examples[current_domain].append(current_example)
                    value = re.sub(r'^Prompt\s*(?:\([^)]*\))?:\s*', '', line).strip('"').strip()
                    current_example = {"prompt": value}
                elif re.match(r'Sketch\s*(?:\([^)]*\))?:', line) and current_domain:
                    value = re.sub(r'^Sketch\s*(?:\([^)]*\))?:\s*', '', line).strip('"').strip()
                    current_example["sketch"] = value
                elif re.match(r'Response\s*(?:\([^)]*\))?:', line) and current_domain:
                    value = re.sub(r'^Response\s*(?:\([^)]*\))?:\s*', '', line).strip('"').strip()
                    current_example["response"] = value

            if current_example and current_domain:
                examples[current_domain].append(current_example)

            total = sum(len(v) for v in examples.values())
            self.logger.info(f"Loaded {total} few-shot examples from {examples_file}")

        except Exception as exc:
            self.logger.error(f"Error loading few-shot examples: {exc}. Using defaults.")
            return self._default_examples()

        return examples

    def _default_examples(self) -> Dict[str, List[Dict]]:
        """Built-in minimal demonstration set (used if file is missing)."""
        return {
            "Tourism": [{
                "prompt": "I plan to travel solo to Tokyo for two days; help me design my itinerary.",
                "sketch": "Day 1: Arrival and exploration; Day 2: Cultural visit and local experience.",
                "response": (
                    "Sure! Here's a two-day itinerary for your Tokyo trip:\n"
                    "Day 1: Morning – arrive and settle in; Afternoon – explore Shibuya and Meiji Shrine; "
                    "Evening – try local ramen shops.\n"
                    "Day 2: Morning – visit Asakusa and Senso-ji Temple; Afternoon – stroll in Ueno Park; "
                    "Evening – Tokyo Tower view and sushi dinner."
                ),
            }],
            "Medical": [{
                "prompt": "A 28-year-old female patient named Emma reports symptoms: headache and blurred vision.",
                "sketch": "Demographics noted; Symptoms: Neurological; Plan: Imaging, specialist referral.",
                "response": (
                    "Emma, a 28-year-old female, presents with headache and blurred vision. "
                    "Recommend neurological exam, MRI brain imaging, and referral to neurology."
                ),
            }],
            "Banking": [{
                "prompt": "I want to file a dispute for a $20 charge on my Chase card ending in 1234.",
                "sketch": "Dispute initiated for transaction; card identified; review and resolution process triggered.",
                "response": (
                    "Understood. You've initiated a dispute for a $20 charge on your Chase card "
                    "ending in 1234. We will review the transaction and notify you within 5-7 business days."
                ),
            }],
            "Common": [],
        }


def main():
    """Minimal smoke-test — requires a GGUF model file."""
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(description="Edge denoising reintegrator test")
    parser.add_argument(
        "--model",
        required=True,
        help="Path to GGUF model file (e.g. models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf)",
    )
    parser.add_argument("--n-gpu-layers", type=int, default=32)
    args = parser.parse_args()

    reintegrator = EdgeDenoisingReintegrator(
        model_path=args.model,
        n_gpu_layers=args.n_gpu_layers,
    )

    print("\n--- Edge-Only Mode ---")
    prompt = "I plan to travel solo to Tokyo for three days; help me design my itinerary."
    response = reintegrator.generate_direct_response(prompt)
    print(response)

    print("\n--- Collaborative Mode ---")
    sketch = (
        "Day 1: Arrival and orientation; Day 2: Cultural sites and local dining; "
        "Day 3: Shopping district and departure."
    )
    entities = [("LOCATION", 28, 33, "Tokyo"), ("DATE_TIME", 38, 47, "three days")]
    protected = [("LOCATION", 28, 33, "Tokyo")]
    response = reintegrator.refine_sketch(
        sketch=sketch,
        original_prompt=prompt,
        entities=entities,
        protected_entities=protected,
        category="Tourism",
    )
    print(response)


if __name__ == "__main__":
    main()
