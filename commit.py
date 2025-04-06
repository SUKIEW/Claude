import hashlib
from typing import List, Tuple
import secrets
from dataclasses import dataclass

@dataclass
class LHHValue:
    """Linear Homomorphic Hash value representation"""
    value: int
    random: int

class CommitmentScheme:
    def generate_commitment(self, lhh_value: LHHValue) -> str:
        # Convert LHH value to bytes and hash it
        lhh_bytes = str(lhh_value.value).encode()
        commitment = hashlib.sha256(lhh_bytes).hexdigest()
        return commitment
    
    def verify_commitment(self, commitment: str, lhh_value: LHHValue) -> bool:
        # Generate commitment from revealed value and compare
        computed_commitment = self.generate_commitment(lhh_value)
        return commitment == computed_commitment