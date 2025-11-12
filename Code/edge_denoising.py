"""
Edge-Side Denoising and Reintegration for PRISM Framework
Implements sketch refinement and privacy-preserving response synthesis.
"""

import re
import os
import logging
from typing import Dict, List, Tuple, Optional


class EdgeDenoisingReintegrator:
    """
    Edge-side denoising and reintegration component.
    Refines cloud-generated sketches with original context and entities.
    """
    
    def __init__(self, examples_file: str = "few_shot_examples_edge.txt"):
        """Initialize edge denoising reintegrator."""
        self.logger = logging.getLogger("EdgeDenoisingReintegrator")
        
        # Load few-shot examples for edge refinement
        self.examples = self._load_few_shot_examples(examples_file)
        
        # Entity placeholder patterns for different obfuscation formats
        self.placeholder_patterns = [
            r'\[([A-Z_]+)\]_(\w+)',  # [PERSON]_Alice format
            r'\[([A-Z_]+)\]',        # [PERSON] format
            r'\[MASKED_([A-Z_]+)\]', # [MASKED_PERSON] format
            r'\[XXX\]',              # Generic mask
            r'<([A-Z_]+)>',          # <PERSON> format
        ]
        
        # Category-specific refinement strategies
        self.refinement_strategies = {
            "Tourism": self._refine_tourism_sketch,
            "Medical": self._refine_medical_sketch,
            "Banking": self._refine_banking_sketch,
            "Common": self._refine_common_sketch
        }
        
        # Edge-only generation strategies (for PRISM)
        self.edge_generation_strategies = {
            "Tourism": self._generate_tourism_response,
            "Medical": self._generate_medical_response,
            "Banking": self._generate_banking_response,
            "Common": self._generate_common_response
        }
    
    def refine_sketch(self, 
                     sketch: str,
                     original_prompt: str,
                     entities: List[Tuple[str, int, int, str]],
                     protected_entities: List[Tuple[str, int, int, str]],
                     category: str) -> str:
        """
        Complete edge-side denoising and reintegration.
        
        Args:
            sketch: Cloud-generated semantic sketch
            original_prompt: Original user prompt
            entities: All detected entities
            protected_entities: Entities that were privacy-protected
            category: Detected category
            
        Returns:
            Final refined response
        """
        self.logger.info(f"Refining {category} sketch with {len(protected_entities)} protected entities")
        
        # Step 1: Replace obfuscated entities with originals
        denoised_sketch = self._replace_obfuscated_entities(
            sketch, original_prompt, entities, protected_entities
        )
        
        # Step 2: Apply category-specific refinement
        refinement_func = self.refinement_strategies.get(category, self._refine_common_sketch)
        refined_response = refinement_func(
            denoised_sketch, original_prompt, entities, protected_entities
        )
        
        # Step 3: Post-processing and quality checks
        final_response = self._post_process_response(
            refined_response, original_prompt, category
        )
        
        self.logger.info(f"Refinement complete: {len(final_response)} characters")
        return final_response
    
    def _replace_obfuscated_entities(self, 
                                   sketch: str,
                                   original_prompt: str,
                                   entities: List[Tuple[str, int, int, str]],
                                   protected_entities: List[Tuple[str, int, int, str]]) -> str:
        """
        Replace obfuscated entities in sketch with original values.
        
        Args:
            sketch: Cloud-generated sketch
            original_prompt: Original prompt
            entities: All detected entities
            protected_entities: Protected entities
            
        Returns:
            Sketch with obfuscated entities replaced
        """
        denoised_sketch = sketch
        
        # Create entity mapping: type -> original values
        entity_mapping = {}
        for entity_type, start, end, entity_text in protected_entities:
            if entity_type not in entity_mapping:
                entity_mapping[entity_type] = []
            entity_mapping[entity_type].append(entity_text)
        
        # Replace placeholders with original entities
        for pattern in self.placeholder_patterns:
            matches = re.finditer(pattern, denoised_sketch)
            
            for match in matches:
                full_match = match.group(0)
                
                if len(match.groups()) >= 2:
                    # Format: [TYPE]_value
                    entity_type = match.group(1)
                    obfuscated_value = match.group(2)
                elif len(match.groups()) == 1:
                    # Format: [TYPE] or <TYPE>
                    entity_type = match.group(1)
                    obfuscated_value = None
                else:
                    # Generic [XXX]
                    entity_type = None
                    obfuscated_value = None
                
                # Find replacement
                replacement = self._find_replacement_entity(
                    entity_type, obfuscated_value, entity_mapping, original_prompt
                )
                
                if replacement:
                    denoised_sketch = denoised_sketch.replace(full_match, replacement, 1)
                    self.logger.debug(f"Replaced {full_match} -> {replacement}")
        
        return denoised_sketch
    
    def _find_replacement_entity(self, 
                               entity_type: Optional[str],
                               obfuscated_value: Optional[str],
                               entity_mapping: Dict[str, List[str]],
                               original_prompt: str) -> Optional[str]:
        """
        Find the best replacement for an obfuscated entity.
        
        Args:
            entity_type: Type of entity to replace
            obfuscated_value: Obfuscated value (if any)
            entity_mapping: Mapping of types to original values
            original_prompt: Original prompt for context
            
        Returns:
            Best replacement entity or None
        """
        if entity_type and entity_type in entity_mapping:
            originals = entity_mapping[entity_type]
            
            if originals:
                # If we have multiple options, prefer the first one
                # In a more sophisticated implementation, we could use
                # semantic similarity or position in the original prompt
                return originals[0]
        
        # Fallback: try to find any entity in the original prompt
        # that might be contextually appropriate
        if entity_type:
            # Look for entity type keywords in original prompt
            type_keywords = {
                "PERSON": ["name", "patient", "customer", "user"],
                "LOCATION": ["city", "place", "destination", "location"],
                "ORGANIZATION": ["bank", "hospital", "company", "organization"],
                "DATE_TIME": ["age", "date", "time", "year", "day"],
            }
            
            keywords = type_keywords.get(entity_type, [])
            for keyword in keywords:
                if keyword in original_prompt.lower():
                    # Extract potential entity near the keyword
                    words = original_prompt.split()
                    for i, word in enumerate(words):
                        if keyword in word.lower() and i < len(words) - 1:
                            return words[i + 1]
        
        return None
    
    def _refine_tourism_sketch(self, 
                             sketch: str,
                             original_prompt: str,
                             entities: List[Tuple[str, int, int, str]],
                             protected_entities: List[Tuple[str, int, int, str]]) -> str:
        """Refine tourism category sketches."""
        
        # Extract key information from original prompt
        duration_info = self._extract_duration(original_prompt)
        location_info = self._extract_locations(entities)
        group_info = self._extract_group_info(original_prompt)
        
        # Build refined response
        refined_response = f"Here's a personalized itinerary for your trip:\n\n"
        
        # Add context from original prompt
        if location_info:
            refined_response += f"**Destination:** {location_info[0]}\n"
        if duration_info:
            refined_response += f"**Duration:** {duration_info}\n"
        if group_info:
            refined_response += f"**Travel Style:** {group_info}\n\n"
        
        # Add the refined sketch
        refined_response += "**Detailed Itinerary:**\n"
        refined_response += sketch
        
        # Add practical tips
        refined_response += "\n\n**Additional Tips:**\n"
        refined_response += "- Book accommodations in advance\n"
        refined_response += "- Check local weather and pack accordingly\n"
        refined_response += "- Keep important documents and emergency contacts handy"
        
        return refined_response
    
    def _refine_medical_sketch(self, 
                             sketch: str,
                             original_prompt: str,
                             entities: List[Tuple[str, int, int, str]],
                             protected_entities: List[Tuple[str, int, int, str]]) -> str:
        """Refine medical category sketches."""
        
        # Extract medical context
        age_info = self._extract_age(original_prompt)
        symptoms_info = self._extract_symptoms(original_prompt)
        
        # Build refined medical response
        refined_response = "**Clinical Assessment Summary:**\n\n"
        
        if age_info:
            refined_response += f"**Patient Demographics:** {age_info}\n"
        if symptoms_info:
            refined_response += f"**Presenting Symptoms:** {', '.join(symptoms_info)}\n\n"
        
        # Add the clinical sketch
        refined_response += "**Recommended Clinical Approach:**\n"
        refined_response += sketch
        
        # Add medical disclaimers
        refined_response += "\n\n**Important Notes:**\n"
        refined_response += "- This is a general clinical framework for reference\n"
        refined_response += "- Actual diagnosis and treatment should involve qualified medical professionals\n"
        refined_response += "- Emergency cases require immediate medical attention"
        
        return refined_response
    
    def _refine_banking_sketch(self, 
                             sketch: str,
                             original_prompt: str,
                             entities: List[Tuple[str, int, int, str]],
                             protected_entities: List[Tuple[str, int, int, str]]) -> str:
        """Refine banking category sketches."""
        
        # Extract banking context
        amount_info = self._extract_amounts(original_prompt)
        bank_info = self._extract_banks(entities)
        
        # Build refined banking response
        refined_response = "**Banking Service Request Summary:**\n\n"
        
        if bank_info:
            refined_response += f"**Financial Institution:** {bank_info[0]}\n"
        if amount_info:
            refined_response += f"**Transaction Amount:** {amount_info[0]}\n\n"
        
        # Add the banking sketch
        refined_response += "**Service Process Overview:**\n"
        refined_response += sketch
        
        # Add banking notices
        refined_response += "\n\n**Important Information:**\n"
        refined_response += "- Processing times may vary based on transaction type\n"
        refined_response += "- Keep all transaction receipts and confirmations\n"
        refined_response += "- Contact customer service for urgent issues"
        
        return refined_response
    
    def _refine_common_sketch(self, 
                            sketch: str,
                            original_prompt: str,
                            entities: List[Tuple[str, int, int, str]],
                            protected_entities: List[Tuple[str, int, int, str]]) -> str:
        """Refine common category sketches."""
        
        # Build general refined response
        refined_response = "**Content Creation Framework:**\n\n"
        
        # Add the sketch
        refined_response += "**Structured Approach:**\n"
        refined_response += sketch
        
        # Add general tips
        refined_response += "\n\n**Implementation Tips:**\n"
        refined_response += "- Adapt the structure to your specific needs\n"
        refined_response += "- Consider your target audience\n"
        refined_response += "- Review and refine the content before finalizing"
        
        return refined_response
    
    def _post_process_response(self, 
                             response: str,
                             original_prompt: str,
                             category: str) -> str:
        """
        Final post-processing and quality checks.
        
        Args:
            response: Refined response
            original_prompt: Original prompt
            category: Category
            
        Returns:
            Final processed response
        """
        # Clean up formatting
        response = re.sub(r'\n{3,}', '\n\n', response)  # Remove excessive newlines
        response = response.strip()
        
        # Ensure minimum response quality
        if len(response) < 100:
            # If response is too short, add fallback content
            fallback = self._generate_fallback_response(original_prompt, category)
            response = fallback + "\n\n" + response
        
        # Add category-specific footer if needed
        if "disclaimer" not in response.lower() and category == "Medical":
            response += "\n\n*This information is for reference only and should not replace professional medical advice.*"
        
        return response
    
    def _generate_fallback_response(self, original_prompt: str, category: str) -> str:
        """Generate fallback response if sketch is too short."""
        fallbacks = {
            "Tourism": f"Based on your travel request, here's a comprehensive approach to planning your trip:",
            "Medical": f"Regarding your medical inquiry, here's a structured clinical approach:",
            "Banking": f"For your banking service request, here's the recommended process:",
            "Common": f"Based on your request, here's a structured approach:"
        }
        return fallbacks.get(category, "Here's a structured approach to your request:")
    
    # Helper methods for extracting information
    def _extract_duration(self, text: str) -> Optional[str]:
        """Extract duration information from text."""
        duration_patterns = [
            r'(\d+)\s+days?',
            r'(\d+)\s+nights?',
            r'for\s+(\d+)\s+days?',
            r'(\d+)-day',
        ]
        
        for pattern in duration_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return f"{match.group(1)} days"
        return None
    
    def _extract_locations(self, entities: List[Tuple[str, int, int, str]]) -> List[str]:
        """Extract location entities."""
        return [entity_text for entity_type, _, _, entity_text in entities if entity_type == "LOCATION"]
    
    def _extract_group_info(self, text: str) -> Optional[str]:
        """Extract group/travel style information."""
        if "solo" in text.lower():
            return "Solo travel"
        elif "family" in text.lower():
            return "Family travel"
        elif "group" in text.lower():
            return "Group travel"
        elif "couple" in text.lower():
            return "Couple travel"
        return None
    
    def _extract_age(self, text: str) -> Optional[str]:
        """Extract age information."""
        age_pattern = r'(\d+)-year-old'
        match = re.search(age_pattern, text)
        return f"{match.group(1)} years old" if match else None
    
    def _extract_symptoms(self, text: str) -> List[str]:
        """Extract symptom information."""
        # Simple symptom extraction - could be enhanced
        symptom_keywords = [
            "pain", "fever", "cough", "headache", "nausea", "fatigue",
            "symptoms", "altered_appetite", "focal_neurological_symptoms"
        ]
        
        found_symptoms = []
        text_lower = text.lower()
        for symptom in symptom_keywords:
            if symptom in text_lower:
                found_symptoms.append(symptom)
        
        return found_symptoms
    
    def _extract_amounts(self, text: str) -> List[str]:
        """Extract monetary amounts."""
        amount_pattern = r'\$(\d+(?:\.\d{2})?)'
        matches = re.findall(amount_pattern, text)
        return [f"${amount}" for amount in matches]
    
    def _extract_banks(self, entities: List[Tuple[str, int, int, str]]) -> List[str]:
        """Extract bank/organization entities."""
        banks = []
        for entity_type, _, _, entity_text in entities:
            if entity_type == "ORGANIZATION" and any(word in entity_text.lower() 
                                                   for word in ["bank", "chase", "wells", "pnc"]):
                banks.append(entity_text)
        return banks
    
    def _load_few_shot_examples(self, examples_file: str) -> Dict[str, List[Dict]]:
        """
        Load few-shot examples for edge refinement from file.
        
        Args:
            examples_file: Path to few-shot examples file
            
        Returns:
            Dictionary of examples organized by domain
        """
        examples = {
            "Tourism": [],
            "Medical": [], 
            "Banking": []
        }
        
        try:
            if os.path.exists(examples_file):
                with open(examples_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Parse examples from file
                current_domain = None
                current_example = {}
                
                for line in content.split('\n'):
                    line = line.strip()
                    
                    if "TOURISM DOMAIN" in line:
                        current_domain = "Tourism"
                    elif "MEDICAL DOMAIN" in line:
                        current_domain = "Medical"
                    elif "BANKING DOMAIN" in line:
                        current_domain = "Banking"
                    elif line.startswith("Prompt:") and current_domain:
                        if current_example:  # Save previous example
                            examples[current_domain].append(current_example)
                        current_example = {"prompt": line.replace("Prompt:", "").strip().strip('"')}
                    elif line.startswith("Sketch:") and current_domain:
                        current_example["sketch"] = line.replace("Sketch:", "").strip().strip('"')
                    elif line.startswith("Response:") and current_domain:
                        current_example["response"] = line.replace("Response:", "").strip().strip('"')
                
                # Add last example
                if current_example and current_domain:
                    examples[current_domain].append(current_example)
                    
                self.logger.info(f"Loaded {sum(len(v) for v in examples.values())} few-shot examples")
            else:
                self.logger.warning(f"Few-shot examples file not found: {examples_file}")
                # Use default examples if file not found
                examples = self._get_default_examples()
                
        except Exception as e:
            self.logger.error(f"Error loading few-shot examples: {e}")
            examples = self._get_default_examples()
            
        return examples
    
    def _get_default_examples(self) -> Dict[str, List[Dict]]:
        """Get default few-shot examples if file loading fails."""
        return {
            "Tourism": [
                {
                    "prompt": "I plan to travel solo to Tokyo for two days; help me design my itinerary.",
                    "sketch": "Day 1: Arrival and exploration; Day 2: Cultural visit and local experience.",
                    "response": "Sure! Here's a two-day itinerary for your Tokyo trip: Day 1: Morning – arrive and settle in; Afternoon – explore Shibuya and Meiji Shrine; Evening – try local ramen shops. Day 2: Morning – visit Asakusa and Senso-ji Temple; Afternoon – stroll in Ueno Park; Evening – Tokyo Tower view and sushi dinner."
                }
            ],
            "Medical": [
                {
                    "prompt": "A 28-year-old female patient named Emma reports symptoms: headache and blurred vision.",
                    "sketch": "Demographics noted; Symptoms: Neurological; Plan: Imaging, specialist referral.",
                    "response": "Emma, a 28-year-old female, presents with headache and blurred vision. Recommend neurological exam, MRI brain imaging, and referral to neurology."
                }
            ],
            "Banking": [
                {
                    "prompt": "I want to file a dispute for a $20 charge on my Chase card ending in 1234.",
                    "sketch": "Dispute initiated for transaction; card identified; review and resolution process triggered.",
                    "response": "Understood. You've initiated a dispute for a $20 charge on your Chase card ending in 1234. We will review the transaction and notify you within 5–7 business days."
                }
            ]
        }
    
    def generate_direct_response(self, prompt: str) -> str:
        """
        Generate response directly on edge device (for PRISM edge-only mode).
        
        Args:
            prompt: Original user prompt
            
        Returns:
            Edge-generated response
        """
        # Detect category from prompt
        category = self._detect_category(prompt)
        
        # Use appropriate generation strategy
        generation_func = self.edge_generation_strategies.get(
            category, self._generate_common_response
        )
        
        response = generation_func(prompt)
        
        self.logger.info(f"Edge-only generation complete for {category} category")
        return response
    
    def _detect_category(self, prompt: str) -> str:
        """Detect category from prompt content."""
        prompt_lower = prompt.lower()
        
        # Tourism keywords
        if any(word in prompt_lower for word in ["travel", "trip", "visit", "destination", "itinerary"]):
            return "Tourism"
        
        # Medical keywords  
        elif any(word in prompt_lower for word in ["patient", "symptom", "diagnosis", "treatment", "medical"]):
            return "Medical"
        
        # Banking keywords
        elif any(word in prompt_lower for word in ["bank", "account", "dispute", "charge", "transfer", "payment"]):
            return "Banking"
        
        # Default
        return "Common"
    
    def _generate_tourism_response(self, prompt: str) -> str:
        """Generate tourism response on edge."""
        response = "**Travel Planning Assistant**\n\n"
        
        # Extract basic information
        duration = self._extract_duration(prompt)
        
        if duration:
            response += f"I'll help you plan your {duration} trip.\n\n"
        else:
            response += "I'll help you plan your trip.\n\n"
        
        response += "**Essential Planning Steps:**\n"
        response += "1. **Destination Research**: Check visa requirements, weather, and local customs\n"
        response += "2. **Accommodation**: Book hotels or rentals in advance for better rates\n"
        response += "3. **Transportation**: Arrange flights and local transport options\n"
        response += "4. **Activities**: Research and book popular attractions early\n"
        response += "5. **Budget Planning**: Estimate daily expenses including meals and shopping\n\n"
        
        response += "**Travel Checklist:**\n"
        response += "- Valid passport and visa (if required)\n"
        response += "- Travel insurance\n"
        response += "- Copies of important documents\n"
        response += "- Local currency and payment methods\n"
        response += "- Weather-appropriate clothing\n\n"
        
        response += "Would you like specific recommendations for any of these areas?"
        
        return response
    
    def _generate_medical_response(self, prompt: str) -> str:
        """Generate medical response on edge."""
        response = "**Medical Information Assistant**\n\n"
        
        response += "Based on your medical query, here's general guidance:\n\n"
        
        response += "**Important Steps:**\n"
        response += "1. **Document Symptoms**: Keep a detailed record of symptoms and their timeline\n"
        response += "2. **Professional Consultation**: Schedule an appointment with a healthcare provider\n"
        response += "3. **Medical History**: Prepare your medical history and current medications list\n"
        response += "4. **Follow-up Care**: Ensure proper follow-up after initial consultation\n\n"
        
        response += "**General Health Tips:**\n"
        response += "- Maintain regular check-ups\n"
        response += "- Follow prescribed treatment plans\n"
        response += "- Keep emergency contacts readily available\n"
        response += "- Document any changes in condition\n\n"
        
        response += "**Disclaimer**: This information is for educational purposes only. "
        response += "Please consult qualified healthcare professionals for medical advice."
        
        return response
    
    def _generate_banking_response(self, prompt: str) -> str:
        """Generate banking response on edge."""
        response = "**Banking Service Assistant**\n\n"
        
        # Check for dispute-related keywords
        if "dispute" in prompt.lower():
            response += "**Dispute Resolution Process:**\n\n"
            response += "1. **Document the Issue**: Gather all relevant transaction details\n"
            response += "2. **Contact Your Bank**: Call the customer service number on your card\n"
            response += "3. **File Formal Dispute**: Submit dispute form within 60 days of statement\n"
            response += "4. **Provide Evidence**: Include receipts, emails, or other documentation\n"
            response += "5. **Track Progress**: Note your dispute reference number\n\n"
            
            response += "**Expected Timeline:**\n"
            response += "- Initial acknowledgment: 2-3 business days\n"
            response += "- Investigation period: 30-90 days\n"
            response += "- Provisional credit: May be issued during investigation\n\n"
        else:
            response += "**Banking Services Overview:**\n\n"
            response += "1. **Account Management**: Online and mobile banking options\n"
            response += "2. **Transaction Services**: Transfers, payments, and deposits\n"
            response += "3. **Security Features**: Fraud alerts and account monitoring\n"
            response += "4. **Customer Support**: 24/7 helpline for urgent issues\n\n"
        
        response += "**Security Reminders:**\n"
        response += "- Never share your PIN or passwords\n"
        response += "- Monitor account activity regularly\n"
        response += "- Report suspicious transactions immediately"
        
        return response
    
    def _generate_common_response(self, prompt: str) -> str:
        """Generate general response on edge."""
        response = "**Information Assistant**\n\n"
        
        response += "I can help you with your request. Here's a structured approach:\n\n"
        
        response += "**Key Considerations:**\n"
        response += "1. **Define Your Goal**: Clearly identify what you want to achieve\n"
        response += "2. **Gather Information**: Research relevant background information\n"
        response += "3. **Plan Your Approach**: Create a step-by-step action plan\n"
        response += "4. **Execute and Monitor**: Implement your plan and track progress\n"
        response += "5. **Review and Adjust**: Evaluate results and make improvements\n\n"
        
        response += "**Best Practices:**\n"
        response += "- Document your process for future reference\n"
        response += "- Seek expert advice when needed\n"
        response += "- Consider multiple perspectives\n"
        response += "- Allow time for thorough completion\n\n"
        
        response += "Feel free to provide more specific details for tailored guidance."
        
        return response


def main():
    """Test the edge denoising reintegrator."""
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize reintegrator
    reintegrator = EdgeDenoisingReintegrator()
    
    print("🔧 PRISM Edge-Side Denoising and Reintegration Demo")
    print("=" * 60)
    
    # Test cases
    test_cases = [
        {
            "category": "Tourism",
            "original_prompt": "I want to plan a trip to Tokyo for 3 days with my family.",
            "sketch": "Day 1: Morning – arrival and accommodation setup; Afternoon – city orientation and major landmark visit; Evening – family-friendly dining. Day 2: Morning – cultural/historical site tour; Afternoon – interactive museum or park; Evening – local performance or entertainment. Day 3: Morning – local market or shopping district; Afternoon – scenic outdoor activity; Evening – departure preparation.",
            "entities": [("LOCATION", 25, 30, "Tokyo"), ("DATE_TIME", 35, 41, "3 days")],
            "protected_entities": [("LOCATION", 25, 30, "Tokyo")]
        },
        {
            "category": "Medical", 
            "original_prompt": "A 28-year-old female patient named Emma reports symptoms: focal_neurological_symptoms",
            "sketch": "Patient Assessment: Age and gender noted; Symptom Documentation: Neurological symptoms recorded; Recommended Actions: Neurological examination, imaging studies, specialist consultation; Follow-up: Monitor symptom progression, schedule return visit.",
            "entities": [("DATE_TIME", 2, 13, "28-year-old"), ("PERSON", 34, 38, "Emma")],
            "protected_entities": [("PERSON", 34, 38, "Emma")]
        },
        {
            "category": "Banking",
            "original_prompt": "I want to file a dispute regarding a charge of $10 on my Chase card",
            "sketch": "Dispute Request: Charge amount and card identification; Documentation Needed: Transaction details, supporting evidence; Process Steps: Initial review, investigation period, provisional credit consideration; Timeline: Standard dispute resolution timeframe, status updates.",
            "entities": [("ORGANIZATION", 56, 61, "Chase")],
            "protected_entities": [("ORGANIZATION", 56, 61, "Chase")]
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n Test {i}: {test_case['category']}")
        print(f"Prompt: {test_case['original_prompt'][:60]}...")
        print(f"Sketch: {test_case['sketch'][:80]}...")
        print("-" * 40)
        
        refined_response = reintegrator.refine_sketch(
            sketch=test_case['sketch'],
            original_prompt=test_case['original_prompt'], 
            entities=test_case['entities'],
            protected_entities=test_case['protected_entities'],
            category=test_case['category']
        )
        
        print(f"Refined Response:\n{refined_response}")


if __name__ == "__main__":
    # Initialize edge denoising reintegrator for testing
    reintegrator = EdgeDenoisingReintegrator()
    print("PRISM Edge Denoising Reintegrator initialized successfully.")