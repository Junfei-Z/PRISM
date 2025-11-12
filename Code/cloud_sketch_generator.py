"""
Cloud-Side Semantic Sketch Generation for PRISM Framework
Implements GPT-4o integration with few-shot learning for robust sketch generation.
"""

import os
import re
import logging
from typing import Dict, List, Optional
from openai import OpenAI


class CloudSketchGenerator:
    """
    Cloud-side semantic sketch generator using GPT-4o with few-shot learning.
    Generates structured sketches from original or obfuscated prompts.
    """
    
    def __init__(self, 
                 api_key: str = None,
                 base_url: str = "https://api.chatanywhere.tech/v1",
                 model: str = "gpt-4o",
                 examples_file: str = "few_shot_examples_cloud.txt"):
        """
        Initialize cloud sketch generator.
        
        Args:
            api_key: OpenAI API key (use environment variable OPENAI_API_KEY if None)
            base_url: API base URL
            model: Model name (gpt-4o)
            examples_file: Path to few-shot examples file
        """
        # Get API key from environment or parameter
        final_api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not final_api_key:
            raise ValueError("API key must be provided via parameter or OPENAI_API_KEY environment variable")
        
        self.client = OpenAI(
            api_key=final_api_key,
            base_url=base_url
        )
        self.model = model
        self.logger = logging.getLogger("CloudSketchGenerator")
        
        # Load few-shot examples
        self.examples = self._load_few_shot_examples(examples_file)
        
        # Category detection patterns
        self.category_patterns = {
            "Tourism": [
                "travel", "trip", "vacation", "holiday", "visit", "tour", "itinerary",
                "sightseeing", "explore", "destination", "flight", "hotel"
            ],
            "Medical": [
                "patient", "symptoms", "doctor", "medical", "health", "pain", "fever",
                "treatment", "diagnosis", "hospital", "clinic", "medication"
            ],
            "Banking": [
                "bank", "account", "card", "charge", "dispute", "transfer", "payment",
                "transaction", "deposit", "withdraw", "balance", "credit"
            ],
            "Common": [
                "blog", "email", "report", "document", "write", "draft", "compare",
                "analysis", "review", "feedback", "presentation"
            ]
        }
    
    def _load_few_shot_examples(self, examples_file: str) -> Dict[str, List[Dict]]:
        """Load few-shot examples from text file."""
        examples = {
            "Tourism": [],
            "Medical": [],
            "Banking": [],
            "Common": []
        }
        
        try:
            file_path = os.path.join(os.path.dirname(__file__), examples_file)
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse examples by category
            current_category = None
            
            for line in content.split('\n'):
                line = line.strip()
                
                # Detect category headers
                if "## TOURISM CATEGORY" in line:
                    current_category = "Tourism"
                elif "## MEDICAL CATEGORY" in line:
                    current_category = "Medical"
                elif "## BANKING CATEGORY" in line:
                    current_category = "Banking"
                elif "## COMMON CATEGORY" in line:
                    current_category = "Common"
                elif "## GENERAL INSTRUCTIONS" in line:
                    break
                
                # Parse examples
                if current_category and line.startswith("Input:"):
                    input_text = line.replace("Input:", "").strip().strip('"')
                elif current_category and line.startswith("Sketch:"):
                    sketch_text = line.replace("Sketch:", "").strip()
                    
                    examples[current_category].append({
                        "input": input_text,
                        "sketch": sketch_text
                    })
            
            self.logger.info(f"Loaded few-shot examples: "
                           f"Tourism={len(examples['Tourism'])}, "
                           f"Medical={len(examples['Medical'])}, "
                           f"Banking={len(examples['Banking'])}, "
                           f"Common={len(examples['Common'])}")
            
        except Exception as e:
            self.logger.error(f"Failed to load examples file: {e}")
            # Fallback examples
            examples = self._get_fallback_examples()
        
        return examples
    
    def _get_fallback_examples(self) -> Dict[str, List[Dict]]:
        """Fallback examples if file loading fails."""
        return {
            "Tourism": [
                {
                    "input": "I plan to travel solo to Tokyo for two days; help me design my itinerary.",
                    "sketch": "Day 1: Morning – arrival and local exploration; Afternoon – museum or park; Evening – local dining. Day 2: Morning – cultural/historical visit; Afternoon – outdoor activity; Evening – local performance."
                }
            ],
            "Medical": [
                {
                    "input": "A 28-year-old female patient named Emma reports symptoms: focal_neurological_symptoms.",
                    "sketch": "Patient Assessment: Age and gender noted; Symptom Documentation: Neurological symptoms recorded; Recommended Actions: Neurological examination, imaging studies, specialist consultation; Follow-up: Monitor symptom progression, schedule return visit."
                }
            ],
            "Banking": [
                {
                    "input": "I want to file a dispute regarding a charge of $10 on my Chase card ending in 1234.",
                    "sketch": "Dispute Request: Charge amount and card identification; Documentation Needed: Transaction details, supporting evidence; Process Steps: Initial review, investigation period, provisional credit consideration; Timeline: Standard dispute resolution timeframe, status updates."
                }
            ],
            "Common": [
                {
                    "input": "Compose an engaging travel blog post about a recent trip to Hawaii.",
                    "sketch": "Blog Structure: Introduction with destination overview; Content Sections: Travel experiences, local attractions, cultural insights, practical tips; Writing Style: Engaging narrative, descriptive language, personal anecdotes; Conclusion: Overall impressions and recommendations."
                }
            ]
        }
    
    def detect_category(self, prompt: str) -> str:
        """
        Detect the category of the input prompt.
        
        Args:
            prompt: Input prompt text
            
        Returns:
            Detected category (Tourism, Medical, Banking, or Common)
        """
        prompt_lower = prompt.lower()
        category_scores = {}
        
        for category, keywords in self.category_patterns.items():
            score = sum(1 for keyword in keywords if keyword in prompt_lower)
            category_scores[category] = score
        
        # Return category with highest score, default to Common
        detected = max(category_scores.items(), key=lambda x: x[1])
        category = detected[0] if detected[1] > 0 else "Common"
        
        self.logger.debug(f"Category detection: {category} (scores: {category_scores})")
        return category
    
    def build_few_shot_prompt(self, category: str, input_prompt: str) -> str:
        """
        Build few-shot prompt with examples from the detected category.
        
        Args:
            category: Detected category
            input_prompt: User's input prompt
            
        Returns:
            Complete prompt with few-shot examples
        """
        system_message = """You are a semantic sketch generator for a privacy-preserving system. Your task is to generate structured outlines/sketches rather than full responses.

Key Requirements:
1. Generate concise, structured sketches with clear organization
2. Use semicolons or bullet points for structure
3. Focus on intent and key action items, not detailed content
4. Handle obfuscated/noisy entities gracefully (treat them as placeholders)
5. Maintain consistent format within each category
6. Keep sketches comprehensive enough for later refinement but not full responses

Generate sketches that capture the essential structure and intent of the request."""

        # Get examples for the category
        examples = self.examples.get(category, [])
        
        # Build few-shot examples
        few_shot_text = f"\nHere are examples for {category} category:\n\n"
        
        for i, example in enumerate(examples[:3]):  # Use up to 3 examples
            few_shot_text += f"Example {i+1}:\n"
            few_shot_text += f"Input: \"{example['input']}\"\n"
            few_shot_text += f"Sketch: {example['sketch']}\n\n"
        
        # Add the current input
        few_shot_text += f"Now generate a sketch for this input:\n"
        few_shot_text += f"Input: \"{input_prompt}\"\n"
        few_shot_text += f"Sketch:"
        
        return system_message + few_shot_text
    
    def generate_semantic_sketch(self, 
                                prompt: str, 
                                category: Optional[str] = None,
                                temperature: float = 0.3,
                                max_tokens: int = 200) -> str:
        """
        Generate semantic sketch using GPT-4o with few-shot learning.
        
        Args:
            prompt: Input prompt (original P or obfuscated P*)
            category: Optional category override
            temperature: Generation temperature
            max_tokens: Maximum tokens in response
            
        Returns:
            Generated semantic sketch
        """
        try:
            # Auto-detect category if not provided
            if category is None:
                category = self.detect_category(prompt)
            
            self.logger.info(f"Generating sketch for category: {category}")
            
            # Build few-shot prompt
            full_prompt = self.build_few_shot_prompt(category, prompt)
            
            # Call GPT-4o API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": full_prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=0.9
            )
            
            # Extract sketch from response
            sketch = response.choices[0].message.content.strip()
            
            # Clean up the sketch (remove any "Sketch:" prefix if model added it)
            sketch = re.sub(r'^Sketch:\s*', '', sketch)
            
            self.logger.info(f"Generated sketch: {sketch[:100]}...")
            return sketch
            
        except Exception as e:
            self.logger.error(f"Failed to generate sketch: {e}")
            return self._generate_fallback_sketch(prompt, category)
    
    def _generate_fallback_sketch(self, prompt: str, category: str) -> str:
        """Generate fallback sketch if API call fails."""
        fallback_sketches = {
            "Tourism": "Travel Planning: Destination and duration noted; Itinerary Structure: Daily activities outline; Key Elements: Accommodation, activities, dining; Logistics: Transportation and timing considerations.",
            "Medical": "Patient Assessment: Demographics and symptoms documented; Clinical Approach: Examination and diagnostic steps; Treatment Plan: Therapeutic interventions; Follow-up: Monitoring and next steps.",
            "Banking": "Service Request: Account and transaction details; Process Steps: Verification and investigation; Resolution: Corrective actions and timeline; Communication: Status updates and follow-up.",
            "Common": "Content Structure: Main topic and format identified; Key Elements: Primary sections and components; Approach: Writing style and methodology; Deliverables: Expected outcomes and format."
        }
        
        fallback = fallback_sketches.get(category, fallback_sketches["Common"])
        self.logger.warning(f"Using fallback sketch for {category}")
        return fallback
    
    def process_prompt(self, 
                      prompt: str, 
                      is_obfuscated: bool = False,
                      category: Optional[str] = None) -> Dict:
        """
        Complete cloud-side processing: G_cloud(x; D).
        
        Args:
            prompt: Input prompt (P or P*)
            is_obfuscated: Whether prompt contains obfuscated entities
            category: Optional category override
            
        Returns:
            Dictionary with processing results
        """
        self.logger.info(f"Processing prompt (obfuscated={is_obfuscated}): {prompt[:50]}...")
        
        # Generate semantic sketch
        sketch = self.generate_semantic_sketch(prompt, category)
        
        # Detect category if not provided
        if category is None:
            category = self.detect_category(prompt)
        
        results = {
            "input_prompt": prompt,
            "is_obfuscated": is_obfuscated,
            "detected_category": category,
            "semantic_sketch": sketch,
            "processing_status": "success"
        }
        
        self.logger.info(f"Cloud processing complete for {category} prompt")
        return results



if __name__ == "__main__":
    # Initialize cloud sketch generator for testing
    try:
        generator = CloudSketchGenerator()
        print("PRISM Cloud Sketch Generator initialized successfully.")
    except Exception as e:
        print(f"Note: API key required for full functionality: {e}")