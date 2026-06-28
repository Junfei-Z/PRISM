"""
Complete PRISM Framework Integration Pipeline
Orchestrates end-to-end privacy-preserving cloud-edge collaborative inference.
"""

import os
import time
import logging
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn.functional as F
import numpy as np
from enum import Enum

from edge_detection import EdgeEntityDetector
from two_layer_ldp import TwoLayerLDP
from cloud_sketch_generator import CloudSketchGenerator
from edge_denoising import EdgeDenoisingReintegrator
from soft_gating import SoftGatingPredictor, RoutingMode


class PRISMPipeline:
    """
    Complete PRISM (Privacy-aware cloud-edge inference) Framework Pipeline.
    
    Implements the full methodology:
    1. Sensitivity Profiling for Context-Aware Routing
    2. Soft Gating with Entropy-Regularized Routing
    3. Adaptive Two-Layer Local Differential Privacy  
    4. Cloud-Edge Semantic Sketch Collaboration
    """
    
    def __init__(
        self,
        slm_model_path: str,
        risk_threshold: float = 0.5,
        privacy_threshold: float = 0.5,
        epsilon_total: float = 2.0,
        alpha: float = 0.5,
        lambda_entropy: float = 0.4,
        api_key: str = None,
        base_url: str = "https://api.chatanywhere.tech/v1",
        model: str = "gpt-4o",
        slm_n_gpu_layers: int = 32,
        slm_n_ctx: int = 2048,
        slm_n_batch: int = 512,
        slm_temperature: float = 0.7,
        slm_top_p: float = 0.9,
        slm_max_tokens: int = 512,
    ):
        """
        Initialize the complete PRISM pipeline (Algorithm 1 in the paper).

        Args:
            slm_model_path: Path to the GGUF-quantised edge SLM file, e.g.
                ``"models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"``.
                The edge SLM (G_edge) is used for both edge-only generation
                and collaborative sketch refinement.  Models evaluated in the
                paper: TinyLLaMA-1.1B (S4), Qwen1.5-1.8B-Chat (S2),
                StableLM-2-Zephyr-1.6B (S3), Phi-3.5-mini-3.5B (S1).
            risk_threshold: Threshold for entity risk assessment.
            privacy_threshold: Threshold for personal context detection.
            epsilon_total: Total LDP privacy budget epsilon.
            alpha: Budget allocation balance parameter (Section 'Adaptive
                Two-Layer LDP' in the paper).
            lambda_entropy: Entropy regularisation weight for soft gating.
            api_key: Cloud LLM API key (falls back to OPENAI_API_KEY env var).
            base_url: Cloud API base URL.
            model: Cloud LLM model name (G_cloud).
            slm_n_gpu_layers: GPU layers for the edge SLM (default 32,
                matching the RTX 3070 setup in the paper experiments).
            slm_n_ctx: SLM context window in tokens (default 2048).
            slm_n_batch: Prompt batch size (default 512).
            slm_temperature: Sampling temperature (default 0.7).
            slm_top_p: Nucleus sampling threshold (default 0.9).
            slm_max_tokens: Max new tokens per SLM call (default 512).
        """
        self.logger = logging.getLogger("PRISMPipeline")

        self.edge_detector = EdgeEntityDetector(
            risk_threshold=risk_threshold,
            privacy_threshold=privacy_threshold,
        )
        self.ldp_mechanism = TwoLayerLDP(alpha=alpha)
        self.cloud_generator = CloudSketchGenerator(
            api_key=api_key,
            base_url=base_url,
            model=model,
        )
        self.edge_reintegrator = EdgeDenoisingReintegrator(
            model_path=slm_model_path,
            n_gpu_layers=slm_n_gpu_layers,
            n_ctx=slm_n_ctx,
            n_batch=slm_n_batch,
            temperature=slm_temperature,
            top_p=slm_top_p,
            max_tokens=slm_max_tokens,
        )
        
        # Configuration
        self.epsilon_total = epsilon_total
        self.risk_threshold = risk_threshold
        self.lambda_entropy = lambda_entropy
        
        # Initialize soft gating predictor with the trained routing classifier.
        # Train it with `python train_soft_gating.py` if the checkpoint is absent.
        self.gating_predictor = SoftGatingPredictor("models/soft_gating_pretrained.pth")
        
        self.logger.info("PRISM Pipeline initialized successfully")
    
    def soft_gating(self, risk_score: float, sensitivity_mask: List[int]) -> RoutingMode:
        """
        Perform soft gating with entropy regularization to determine routing.
        
        Args:
            risk_score: Overall privacy risk score
            sensitivity_mask: Binary mask for entity protection
            
        Returns:
            Selected routing mode
        """
        # Use the soft gating predictor
        routing_mode, probs = self.gating_predictor.route(
            risk_score, 
            sensitivity_mask, 
            return_probs=True
        )
        
        # Log routing information
        self.logger.info(f"Routing decision: {routing_mode.name}")
        self.logger.info(f"Probabilities - Edge: {probs['edge_only']:.3f}, "
                        f"Collab: {probs['collaborative']:.3f}, "
                        f"Cloud: {probs['cloud_only']:.3f}, "
                        f"Entropy: {probs['entropy']:.3f}")
        
        return routing_mode
    
    def process_prompt_end_to_end(self, user_prompt: str) -> Dict:
        """
        Complete end-to-end PRISM processing pipeline (Algorithm 1 in paper).
        
        Args:
            user_prompt: Original user prompt
            
        Returns:
            Dictionary containing complete processing results
        """
        self.logger.info(f"Starting end-to-end processing: '{user_prompt[:50]}...'")
        start_time = time.time()
        
        try:
            # Step 1: Sensitivity Profiling for Context-Aware Routing
            step1_start = time.time()
            detection_results = self.edge_detector.detect_and_classify(user_prompt)
            step1_time = time.time() - step1_start
            
            self.logger.info(f"Step 1 complete: {len(detection_results['entities'])} entities, "
                           f"risk={detection_results['risk_score']:.2f}, "
                           f"protection_needed={detection_results['needs_protection']}")
            
            # Step 2: Soft Gating with Entropy-Regularized Routing
            step2_start = time.time()
            routing_mode = self.soft_gating(
                detection_results['risk_score'],
                detection_results['sensitivity_labels']
            )
            step2_time = time.time() - step2_start
            
            self.logger.info(f"Step 2 complete: routing_mode={routing_mode.value}")
            
            # Execute based on routing decision
            if routing_mode == RoutingMode.EDGE_ONLY:
                # Edge-only processing
                step3_start = time.time()
                # Use edge model directly
                final_response = self.edge_reintegrator.generate_direct_response(user_prompt)
                step3_time = time.time() - step3_start
                step4_time = 0  # No cloud processing
                
                cloud_input = None
                is_obfuscated = False
                semantic_sketch = None
                detected_category = "edge_only"
                
            elif routing_mode == RoutingMode.CLOUD_ONLY:
                # Cloud-only processing (no privacy protection)
                cloud_input = user_prompt
                is_obfuscated = False
                
                step3_start = time.time()
                cloud_results = self.cloud_generator.process_prompt(
                    cloud_input, is_obfuscated, None
                )
                semantic_sketch = cloud_results['semantic_sketch']
                detected_category = cloud_results['detected_category']
                final_response = semantic_sketch  # Direct cloud response
                step3_time = time.time() - step3_start
                step4_time = 0  # No edge refinement
                
            else:  # RoutingMode.COLLABORATIVE
                # Step 3: Adaptive Two-Layer LDP (if collaborative)
                step3_start = time.time()
                if detection_results['protected_entities']:
                    protected_prompt = self._apply_privacy_protection(
                        user_prompt, detection_results['protected_entities']
                    )
                    cloud_input = protected_prompt
                    is_obfuscated = True
                else:
                    cloud_input = user_prompt
                    is_obfuscated = False
                step3_time = time.time() - step3_start
            
                self.logger.info(f"Step 3 complete: privacy_applied={is_obfuscated}")
                
                # Step 4: Cloud-Edge Semantic Sketch Collaboration
                step4_start = time.time()
                cloud_results = self.cloud_generator.process_prompt(
                    cloud_input, is_obfuscated, detection_results.get('detected_category')
                )
                semantic_sketch = cloud_results['semantic_sketch']
                detected_category = cloud_results['detected_category']
                step4_time = time.time() - step4_start
                
                self.logger.info(f"Step 4 complete: sketch generated ({len(semantic_sketch)} chars)")
                
                # Step 5: Edge-Side Refinement
                step5_start = time.time()
                final_response = self.edge_reintegrator.refine_sketch(
                    sketch=semantic_sketch,
                    original_prompt=user_prompt,
                    entities=detection_results['entities'],
                    protected_entities=detection_results['protected_entities'],
                    category=detected_category
                )
                step5_time = time.time() - step5_start
                step3_time += step5_time  # Combine for backward compatibility
            
            total_time = time.time() - start_time
            
            self.logger.info(f"PRISM processing complete in {total_time:.3f}s")
            
            # Compile comprehensive results
            results = {
                # Input
                "original_prompt": user_prompt,
                
                # Step 1: Sensitivity Profiling
                "edge_detection": {
                    "entities": detection_results['entities'],
                    "risk_score": detection_results['risk_score'],
                    "privacy_context_score": detection_results['privacy_context_score'],
                    "needs_protection": detection_results['needs_protection'],
                    "protected_entities": detection_results['protected_entities'],
                    "factual_entities": detection_results['factual_entities']
                },
                
                # Step 2: Routing Decision
                "routing": {
                    "mode": routing_mode.value if 'routing_mode' in locals() else "unknown",
                    "risk_score": detection_results['risk_score'],
                    "privacy_applied": is_obfuscated,
                    "cloud_input": cloud_input,
                    "epsilon_total": self.epsilon_total if is_obfuscated else None
                },
                
                # Step 3/4: Cloud Processing (if applicable)
                "cloud_processing": {
                    "detected_category": detected_category if 'detected_category' in locals() else None,
                    "semantic_sketch": semantic_sketch if 'semantic_sketch' in locals() else None,
                    "processing_status": cloud_results['processing_status'] if 'cloud_results' in locals() else "skipped"
                },
                
                # Step 5: Edge Refinement
                "edge_refinement": {
                    "final_response": final_response
                },
                
                # Performance Metrics
                "performance": {
                    "total_time": total_time,
                    "step1_sensitivity_profiling": step1_time,
                    "step2_soft_gating": step2_time, 
                    "step3_cloud_generation": step3_time,
                    "step4_edge_refinement": step4_time,
                    "privacy_overhead": step2_time + step4_time,
                    "cloud_communication_time": step3_time
                },
                
                # Status
                "status": "success",
                "privacy_preserved": is_obfuscated
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            return self._create_error_result(user_prompt, str(e), time.time() - start_time)
    
    def _apply_privacy_protection(self, 
                                prompt: str, 
                                protected_entities: List[Tuple[str, int, int, str]]) -> str:
        """
        Apply two-layer LDP protection to entities.
        
        Args:
            prompt: Original prompt
            protected_entities: Entities requiring protection
            
        Returns:
            Privacy-protected prompt
        """
        if not protected_entities:
            return prompt
        
        protected_prompt = prompt
        
        # Apply LDP obfuscation to each protected entity
        for entity_type, start, end, entity_value in protected_entities:
            obfuscated_value = self.ldp_mechanism.two_layer_dp(
                entity_type=entity_type,
                entity_value=entity_value,
                epsilon_total=self.epsilon_total
            )
            
            # Replace entity in prompt with obfuscated version
            protected_prompt = protected_prompt.replace(entity_value, obfuscated_value, 1)
            
            self.logger.debug(f"Protected entity: {entity_value} -> {obfuscated_value}")
        
        return protected_prompt
    
    def _create_error_result(self, prompt: str, error: str, elapsed_time: float) -> Dict:
        """Create error result structure."""
        return {
            "original_prompt": prompt,
            "status": "error",
            "error_message": error,
            "final_response": "I apologize, but I encountered an error processing your request. Please try again.",
            "performance": {
                "total_time": elapsed_time,
                "error_occurred": True
            }
        }
    
    def get_pipeline_info(self) -> Dict:
        """Get information about the pipeline configuration."""
        slm = self.edge_reintegrator
        return {
            "components": {
                "edge_detector": "EdgeEntityDetector",
                "ldp_mechanism": "TwoLayerLDP",
                "cloud_generator": "CloudSketchGenerator",
                "edge_reintegrator": "EdgeDenoisingReintegrator",
            },
            "configuration": {
                "risk_threshold": self.risk_threshold,
                "epsilon_total": self.epsilon_total,
                "alpha": self.ldp_mechanism.alpha,
                "cloud_model": self.cloud_generator.model,
                "edge_slm": {
                    "model_path": slm.model_path,
                    "n_gpu_layers": slm.n_gpu_layers,
                    "n_ctx": slm.n_ctx,
                    "temperature": slm.temperature,
                    "top_p": slm.top_p,
                    "max_tokens": slm.max_tokens,
                },
            },
            "methodology": "PRISM: Privacy-Aware Cloud-Edge Inference Framework",
        }
    
    def benchmark_performance(self, test_prompts: List[str]) -> Dict:
        """
        Benchmark pipeline performance across multiple prompts.
        
        Args:
            test_prompts: List of test prompts
            
        Returns:
            Performance benchmark results
        """
        self.logger.info(f"Running performance benchmark on {len(test_prompts)} prompts")
        
        results = []
        total_start = time.time()
        
        for i, prompt in enumerate(test_prompts):
            self.logger.info(f"Benchmark {i+1}/{len(test_prompts)}: {prompt[:30]}...")
            result = self.process_prompt_end_to_end(prompt)
            results.append(result)
        
        total_time = time.time() - total_start
        
        # Calculate aggregate metrics
        successful_runs = [r for r in results if r['status'] == 'success']
        privacy_protected = [r for r in successful_runs if r['privacy_preserved']]
        
        avg_latency = sum(r['performance']['total_time'] for r in successful_runs) / len(successful_runs) if successful_runs else 0
        avg_privacy_overhead = sum(r['performance']['privacy_overhead'] for r in successful_runs) / len(successful_runs) if successful_runs else 0
        
        benchmark_results = {
            "summary": {
                "total_prompts": len(test_prompts),
                "successful_runs": len(successful_runs),
                "privacy_protected_prompts": len(privacy_protected),
                "success_rate": len(successful_runs) / len(test_prompts) * 100,
                "privacy_protection_rate": len(privacy_protected) / len(test_prompts) * 100
            },
            "performance": {
                "total_benchmark_time": total_time,
                "average_latency_per_prompt": avg_latency,
                "average_privacy_overhead": avg_privacy_overhead,
                "throughput_prompts_per_second": len(test_prompts) / total_time
            },
            "detailed_results": results
        }
        
        self.logger.info(f"Benchmark complete: {len(successful_runs)}/{len(test_prompts)} successful, "
                        f"avg_latency={avg_latency:.3f}s")
        
        return benchmark_results


def main():
    """End-to-end demo of the PRISM pipeline."""
    import os
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(description="PRISM Pipeline Demo")
    parser.add_argument(
        "--slm-model",
        default=os.environ.get("PRISM_SLM_MODEL_PATH"),
        required=not os.environ.get("PRISM_SLM_MODEL_PATH"),
        help=(
            "Path to GGUF edge SLM file. "
            "Supported: TinyLLaMA-1.1B, Qwen1.5-1.8B-Chat, "
            "StableLM-2-Zephyr-1.6B, Phi-3.5-mini-3.5B"
        ),
    )
    parser.add_argument("--n-gpu-layers", type=int, default=32)
    parser.add_argument("--epsilon", type=float, default=2.0, help="Total LDP budget")
    args = parser.parse_args()

    print("PRISM Framework Pipeline Demo")
    print("=" * 70)

    pipeline = PRISMPipeline(
        slm_model_path=args.slm_model,
        slm_n_gpu_layers=args.n_gpu_layers,
        epsilon_total=args.epsilon,
    )

    info = pipeline.get_pipeline_info()
    slm_cfg = info["configuration"]["edge_slm"]
    print(f"\nPipeline Configuration:")
    print(f"   Cloud LLM  : {info['configuration']['cloud_model']}")
    print(f"   Edge SLM   : {slm_cfg['model_path']}")
    print(f"   GPU layers : {slm_cfg['n_gpu_layers']}")
    print(f"   Privacy    : epsilon={info['configuration']['epsilon_total']}")

    test_prompts = [
        # (prompt, expected_mode)
        ("I want to plan a trip to Tokyo for 3 days with my family", "Collaborative"),
        ("My name is Alice and I need to find a doctor in Boston.", "Edge-only"),
        ("I want to file a dispute regarding a charge of $10 on my Chase card", "Collaborative"),
        ("What is the capital of Japan?", "Cloud-only"),
    ]

    print(f"\nProcessing {len(test_prompts)} test prompts")
    print("=" * 70)

    for i, (prompt, expected) in enumerate(test_prompts, 1):
        print(f"\nTest {i} [{expected}]: {prompt}")
        print("-" * 50)
        result = pipeline.process_prompt_end_to_end(prompt)
        if result["status"] == "success":
            e = result["edge_detection"]
            p = result["performance"]
            print(f"  Entities  : {len(e['entities'])}, risk={e['risk_score']:.2f}")
            print(f"  Mode      : {result['routing']['mode']}")
            print(f"  Latency   : {p['total_time']:.2f}s")
            print(f"  Response  : {result['edge_refinement']['final_response'][:120]}...")
        else:
            print(f"  Error: {result['error_message']}")


if __name__ == "__main__":
    main()