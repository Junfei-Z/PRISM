"""
Edge-Side Entity Detection for PRISM Framework
Implements category-weighted risk scoring and lightweight personal context detection.
"""

import re
from typing import List, Dict, Tuple, Set
from presidio_analyzer import AnalyzerEngine
import logging

class EdgeEntityDetector:
    """
    Edge-side entity detection with category-weighted risk scoring 
    and lightweight personal context detection.
    """
    
    def __init__(self, 
                 category_weights: Dict[str, float] = None,
                 risk_threshold: float = 0.5,
                 privacy_threshold: float = 0.5):
        """
        Initialize edge entity detector.
        
        Args:
            category_weights: Sensitivity weights for entity categories
            risk_threshold: Threshold for overall risk score
            privacy_threshold: Threshold for privacy context detection
        """
        # Use smaller spaCy model for faster processing
        from presidio_analyzer.nlp_engine import NlpEngineProvider
        nlp_configuration = {
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
        }
        nlp_engine = NlpEngineProvider(nlp_configuration=nlp_configuration).create_engine()
        self.analyzer = AnalyzerEngine(nlp_engine=nlp_engine)
        self.risk_threshold = risk_threshold
        self.privacy_threshold = privacy_threshold
        
        # Optimized category weights based on your dataset analysis
        # Dataset contains: Medical, Tourism, Bank, Common categories
        # Entity types found: PERSON, LOCATION, DATE_TIME, ORGANIZATION, NRP
        self.category_weights = category_weights or {
            # High-risk personal identifiers (most common in your dataset)
            "PERSON": 0.9,              # 77 instances - names in medical/banking contexts
            
            # Medium-risk location data  
            "LOCATION": 0.4,            # 86 instances - travel destinations, medical facilities
            
            # Organization names - context dependent
            "ORGANIZATION": 0.5,        # 72 instances - banks, hospitals, companies
            
            # Temporal data - lower risk but still sensitive in context
            "DATE_TIME": 0.3,           # 185 instances - ages, dates, times
            
            # Nationality/Race/Political - sensitive demographic info
            "NRP": 0.7,                 # 2 instances - demographic information
            
            # Additional entity types that might appear
            "EMAIL_ADDRESS": 0.9,
            "PHONE_NUMBER": 0.8,
            "CREDIT_CARD": 0.95,
            "SSN": 1.0,
            "MEDICAL_LICENSE": 0.85,
            "US_PASSPORT": 0.9,
            "IP_ADDRESS": 0.6,
            "URL": 0.3,
            
            # spaCy entities that appear in your dataset
            "MONEY": 0.6,               # Banking amounts - moderate risk
            "CARDINAL": 0.1,            # Numbers - low risk unless in sensitive context
            "ORDINAL": 0.1,             # Ordinal numbers - low risk
            "QUANTITY": 0.2,            # Quantities - low risk
            "PRODUCT": 0.3,             # Product names - low-medium risk
            "WORK_OF_ART": 0.2,         # Art/media references - low risk
            "LANGUAGE": 0.2,            # Language references - low risk
            "PERCENT": 0.2              # Percentages - low risk
        }
        
        # First-person personal indicators (lowercase for matching)
        self.personal_indicators = {
            "I", "my", "me", "our", "we", "mine", "us", "myself", "ourselves"
        }
        
        self.logger = logging.getLogger("EdgeEntityDetector")
    
    def run_ner(self, text: str) -> List[Tuple[str, int, int, str]]:
        """
        Run Named Entity Recognition on input text.
        
        Args:
            text: Input prompt text
            
        Returns:
            List of tuples (entity_type, start, end, entity_text)
        """
        try:
            results = self.analyzer.analyze(text=text, language="en")
            entities = []
            
            for result in results:
                entity_text = text[result.start:result.end]
                entities.append((
                    result.entity_type,
                    result.start, 
                    result.end,
                    entity_text
                ))
            
            self.logger.info(f"Detected {len(entities)} entities: {[e[0] for e in entities]}")
            return entities
            
        except Exception as e:
            self.logger.error(f"NER failed: {e}")
            return []
    
    def compute_category_weighted_risk_score(self, entities: List[Tuple[str, int, int, str]]) -> float:
        """
        Compute category-weighted risk score: R(P) = Σ w_{t_i} · I(e_i)
        
        Args:
            entities: List of detected entities
            
        Returns:
            Category-weighted risk score
        """
        risk_score = 0.0
        
        for entity_type, start, end, entity_text in entities:
            weight = self.category_weights.get(entity_type, 0.2)  # Default weight for unknown types
            risk_score += weight
            self.logger.debug(f"Entity {entity_text} ({entity_type}): weight={weight}")
        
        self.logger.info(f"Total risk score: {risk_score}")
        return risk_score
    
    def detect_personal_context(self, text: str) -> float:
        """
        Lightweight personal context detection: Δ_i = max_{w ∈ W_personal} I(w ∈ P)
        
        Args:
            text: Input prompt text
            
        Returns:
            Binary privacy context score (0 or 1)
        """
        # Convert to lowercase for case-insensitive matching
        text_lower = text.lower()
        
        # Split into tokens (simple whitespace + punctuation split)
        tokens = re.findall(r'\b\w+\b', text_lower)
        
        # Check for personal indicators
        has_personal_context = any(token in self.personal_indicators for token in tokens)
        
        privacy_score = 1.0 if has_personal_context else 0.0
        
        if has_personal_context:
            found_indicators = [token for token in tokens if token in self.personal_indicators]
            self.logger.info(f"Personal context detected: {found_indicators}")
        else:
            self.logger.info("No personal context detected - factual query")
            
        return privacy_score
    
    def classify_entity_sensitivity(self, 
                                   entities: List[Tuple[str, int, int, str]], 
                                   privacy_context_score: float) -> List[int]:
        """
        Classify each entity as privacy-sensitive (1) or commonsense (0).
        
        Args:
            entities: List of detected entities
            privacy_context_score: Personal context score from detect_personal_context
            
        Returns:
            List of binary sensitivity labels (0 or 1) for each entity
        """
        sensitivity_labels = []
        
        for entity_type, start, end, entity_text in entities:
            # If personal context detected, mark as privacy-sensitive
            # Otherwise, mark as commonsense/factual
            if privacy_context_score > self.privacy_threshold:
                sensitivity_label = 1  # Privacy-sensitive
                self.logger.debug(f"Entity '{entity_text}' marked as privacy-sensitive (personal context)")
            else:
                sensitivity_label = 0  # Commonsense/factual
                self.logger.debug(f"Entity '{entity_text}' marked as commonsense (factual context)")
                
            sensitivity_labels.append(sensitivity_label)
        
        return sensitivity_labels
    
    def detect_and_classify(self, prompt: str) -> Dict:
        """
        Complete edge-side entity detection and classification pipeline.
        
        Args:
            prompt: Input user prompt
            
        Returns:
            Dictionary containing detection results
        """
        self.logger.info(f"Processing prompt: '{prompt[:50]}...'")
        
        # Step 1: Run NER to extract entities
        entities = self.run_ner(prompt)
        
        # Step 2: Compute category-weighted risk score
        risk_score = self.compute_category_weighted_risk_score(entities)
        
        # Step 3: Check if risk threshold is exceeded
        if risk_score < self.risk_threshold:
            self.logger.info(f"Risk score {risk_score} < threshold {self.risk_threshold}, no protection needed")
            sensitivity_labels = [0] * len(entities)  # All entities marked as non-sensitive
            privacy_context_score = 0.0
        else:
            # Step 4: Detect personal context
            privacy_context_score = self.detect_personal_context(prompt)
            
            # Step 5: Classify entity sensitivity based on context
            sensitivity_labels = self.classify_entity_sensitivity(entities, privacy_context_score)
        
        # Prepare results
        results = {
            "prompt": prompt,
            "entities": entities,
            "risk_score": risk_score,
            "privacy_context_score": privacy_context_score,
            "sensitivity_labels": sensitivity_labels,
            "needs_protection": risk_score >= self.risk_threshold,
            "has_personal_context": privacy_context_score > self.privacy_threshold,
            "protected_entities": [
                entities[i] for i, label in enumerate(sensitivity_labels) if label == 1
            ],
            "factual_entities": [
                entities[i] for i, label in enumerate(sensitivity_labels) if label == 0
            ]
        }
        
        self.logger.info(f"Detection complete: {len(results['protected_entities'])} protected, "
                        f"{len(results['factual_entities'])} factual")
        
        return results


if __name__ == "__main__":
    # Initialize detector for testing
    detector = EdgeEntityDetector()
    print("PRISM Sensitivity Profiling Module initialized successfully.")