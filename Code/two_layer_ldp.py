"""
Two-Layer Local Differential Privacy for PRISM Framework
Implements category-weighted privacy budget allocation and entity obfuscation.
"""

import random
import numpy as np
from typing import Dict, List, Tuple
import logging


class TwoLayerLDP:
    """
    Two-layer Local Differential Privacy mechanism with adaptive budget allocation.
    Implements category-weighted privacy protection as described in PRISM methodology.
    """
    
    def __init__(self, 
                 type_domain: List[str] = None,
                 value_domain_dict: Dict[str, List[str]] = None,
                 alpha: float = 0.5):
        """
        Initialize Two-Layer LDP mechanism.
        
        Args:
            type_domain: List of entity categories
            value_domain_dict: Dictionary mapping entity types to their value domains
            alpha: Balance parameter for budget allocation (0 < alpha <= 1)
        """
        # Enhanced type domain based on your dataset analysis
        self.type_domain = type_domain or [
            "PERSON", "LOCATION", "ORGANIZATION", "DATE_TIME", "NRP",
            "EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD", "SSN", 
            "MEDICAL_LICENSE", "US_PASSPORT", "IP_ADDRESS", "URL"
        ]
        
        # Enhanced value domains optimized for your dataset
        self.value_domain_dict = value_domain_dict or {
            # Personal identifiers - common names from your dataset
            "PERSON": [
                "Alex", "Emma", "John", "Sophia", "Michael", "Olivia", "James", 
                "Charlotte", "Robert", "Amelia", "William", "Isabella", "David",
                "Mia", "Richard", "Evelyn", "Joseph", "Harper", "Thomas", "Luna"
            ],
            
            # Locations - cities/places from your dataset
            "LOCATION": [
                "Paris", "Tokyo", "New York", "London", "Beijing", "Sydney",
                "Boston", "Chicago", "Los Angeles", "Miami", "Seattle", "Dublin",
                "Singapore", "Toronto", "Vienna", "Barcelona", "Rome", "Amsterdam"
            ],
            
            # Organizations - banks, hospitals, companies from your dataset  
            "ORGANIZATION": [
                "Chase Bank", "Wells Fargo", "Bank of America", "PNC Bank", 
                "US Bank", "TD Bank", "Citibank", "Capital One",
                "General Hospital", "City Medical Center", "Health Clinic",
                "Google", "Microsoft", "Apple", "Amazon", "Meta"
            ],
            
            # Date/Time values - ages, years, dates
            "DATE_TIME": [
                "25-year-old", "30-year-old", "45-year-old", "2024-01-15",
                "2024-02-20", "2024-03-10", "Monday", "Tuesday", "January",
                "February", "2023", "2024", "morning", "afternoon", "evening"
            ],
            
            # Nationality/Race/Political
            "NRP": [
                "American", "European", "Asian", "Canadian", "British", 
                "German", "French", "Japanese", "Chinese", "Australian"
            ],
            
            # Contact information
            "EMAIL_ADDRESS": [
                "user@example.com", "test@gmail.com", "contact@company.com",
                "info@hospital.org", "support@bank.com", "admin@clinic.net"
            ],
            
            "PHONE_NUMBER": [
                "123-456-7890", "555-555-5555", "800-123-4567", "617-555-0123",
                "415-555-7890", "212-555-1234", "305-555-9876", "206-555-4321"
            ],
            
            # Financial
            "CREDIT_CARD": [
                "4111-1111-1111-1111", "5555-5555-5555-4444", "3782-822463-10005"
            ],
            
            "SSN": [
                "123-45-6789", "987-65-4321", "555-44-3333"
            ],
            
            # Technical
            "IP_ADDRESS": [
                "192.168.1.1", "10.0.0.1", "172.16.0.1", "203.0.113.1"
            ],
            
            "URL": [
                "https://example.com", "https://company.org", "https://hospital.net"
            ]
        }
        
        self.alpha = alpha
        self.logger = logging.getLogger("TwoLayerLDP")
        
        # Category weights for adaptive budget allocation (from edge detection)
        self.category_weights = {
            "PERSON": 0.9,
            "EMAIL_ADDRESS": 0.9,
            "PHONE_NUMBER": 0.8,
            "CREDIT_CARD": 0.95,
            "SSN": 1.0,
            "MEDICAL_LICENSE": 0.85,
            "US_PASSPORT": 0.9,
            "LOCATION": 0.4,
            "ORGANIZATION": 0.5,
            "DATE_TIME": 0.3,
            "NRP": 0.7,
            "IP_ADDRESS": 0.6,
            "URL": 0.3
        }
    
    def adaptive_budget_allocation(self, 
                                 entity_type: str, 
                                 epsilon_total: float) -> Tuple[float, float]:
        """
        Adaptive privacy budget allocation based on category weights.
        
        Args:
            entity_type: Type of the entity
            epsilon_total: Total privacy budget
            
        Returns:
            Tuple of (epsilon1, epsilon2) for category and value protection
        """
        # Get category weight
        w_c = self.category_weights.get(entity_type, 0.5)
        
        # Calculate adaptive allocation: ε1 = ε_total * w_c / (w_c + α(1-w_c))
        epsilon1 = epsilon_total * w_c / (w_c + self.alpha * (1 - w_c))
        epsilon2 = epsilon_total - epsilon1
        
        self.logger.debug(f"Budget allocation for {entity_type}: "
                         f"weight={w_c}, ε1={epsilon1:.3f}, ε2={epsilon2:.3f}")
        
        return epsilon1, epsilon2
    
    def two_layer_dp(self, 
                    entity_type: str, 
                    entity_value: str, 
                    epsilon_total: float = 2.0) -> str:
        """
        Two-layer DP mechanism with adaptive budget allocation.
        
        Args:
            entity_type: Original entity type, e.g., "PERSON"
            entity_value: Entity value, e.g., "Alice"  
            epsilon_total: Total privacy budget
            
        Returns:
            Obfuscated value, e.g., "[PERSON]_Alice" or "[LOCATION]_Paris"
        """
        # Step 1: Adaptive budget allocation based on category weight
        epsilon1, epsilon2 = self.adaptive_budget_allocation(entity_type, epsilon_total)
        
        # Step 2: Category-level LDP (ε1 budget)
        noisy_type = self._category_level_ldp(entity_type, epsilon1)
        
        # Step 3: Value-level LDP (ε2 budget)  
        noisy_value = self._value_level_ldp(entity_value, noisy_type, epsilon2)
        
        # Step 4: Generate obfuscated result
        obfuscated_result = f"[{noisy_type}]_{noisy_value}"
        
        self.logger.info(f"DP obfuscation: {entity_type}:{entity_value} -> {obfuscated_result}")
        return obfuscated_result
    
    def _category_level_ldp(self, entity_type: str, epsilon1: float) -> str:
        """
        Category-level LDP: M1(c) with probability p1.
        
        Args:
            entity_type: True entity type
            epsilon1: Privacy budget for category protection
            
        Returns:
            Noisy entity type
        """
        K1 = len(self.type_domain)
        p1 = np.exp(epsilon1) / (np.exp(epsilon1) + K1 - 1)
        
        if random.random() < p1:
            noisy_type = entity_type
            self.logger.debug(f"Category preserved: {entity_type} (p={p1:.3f})")
        else:
            candidates = [t for t in self.type_domain if t != entity_type]
            noisy_type = random.choice(candidates) if candidates else entity_type
            self.logger.debug(f"Category obfuscated: {entity_type} -> {noisy_type} (p={1-p1:.3f})")
        
        return noisy_type
    
    def _value_level_ldp(self, entity_value: str, noisy_type: str, epsilon2: float) -> str:
        """
        Value-level LDP: M2(v) with probability p2.
        
        Args:
            entity_value: True entity value
            noisy_type: Noisy entity type from category-level LDP
            epsilon2: Privacy budget for value protection
            
        Returns:
            Noisy entity value
        """
        # Get value domain for the noisy type
        value_domain = self.value_domain_dict.get(noisy_type, [])
        
        if not value_domain:
            self.logger.warning(f"No value domain for type {noisy_type}, using original value")
            return entity_value
        
        # Handle case where original value is not in domain
        if entity_value not in value_domain:
            true_value = random.choice(value_domain)
            self.logger.debug(f"Value not in domain, using random: {entity_value} -> {true_value}")
        else:
            true_value = entity_value
        
        K2 = len(value_domain)
        if K2 <= 1:
            return true_value
        
        p2 = np.exp(epsilon2) / (np.exp(epsilon2) + K2 - 1)
        
        if random.random() < p2:
            noisy_value = true_value
            self.logger.debug(f"Value preserved: {true_value} (p={p2:.3f})")
        else:
            candidates = [v for v in value_domain if v != true_value]
            noisy_value = random.choice(candidates) if candidates else true_value
            self.logger.debug(f"Value obfuscated: {true_value} -> {noisy_value} (p={1-p2:.3f})")
        
        return noisy_value
    
    def batch_obfuscate(self, 
                       entities: List[Tuple[str, str]], 
                       epsilon_total: float = 2.0) -> List[str]:
        """
        Batch obfuscation for multiple entities.
        
        Args:
            entities: List of (entity_type, entity_value) tuples
            epsilon_total: Total privacy budget per entity
            
        Returns:
            List of obfuscated values
        """
        results = []
        
        for entity_type, entity_value in entities:
            obfuscated = self.two_layer_dp(entity_type, entity_value, epsilon_total)
            results.append(obfuscated)
        
        self.logger.info(f"Batch obfuscation completed: {len(entities)} entities processed")
        return results
    
    def get_budget_allocation_info(self, entity_type: str, epsilon_total: float = 2.0) -> Dict:
        """
        Get detailed information about budget allocation for an entity type.
        
        Args:
            entity_type: Entity type to analyze
            epsilon_total: Total privacy budget
            
        Returns:
            Dictionary with allocation details
        """
        epsilon1, epsilon2 = self.adaptive_budget_allocation(entity_type, epsilon_total)
        weight = self.category_weights.get(entity_type, 0.5)
        
        return {
            "entity_type": entity_type,
            "category_weight": weight,
            "epsilon_total": epsilon_total,
            "epsilon1_category": epsilon1,
            "epsilon2_value": epsilon2,
            "category_ratio": epsilon1 / epsilon_total,
            "value_ratio": epsilon2 / epsilon_total,
            "alpha_parameter": self.alpha
        }


if __name__ == "__main__":
    # Initialize two-layer LDP for testing
    ldp = TwoLayerLDP()
    print("PRISM Two-Layer LDP Module initialized successfully.")