import time
from cryptography.fernet import Fernet
from typing import Tuple
from .config import cfg

class SimulatedEncryptor:
    """
    Uses Fernet for real encryption while simulating algorithmic latency
    via sleep based on profile.base_latency_ms. This keeps encryption
    realistic while allowing latency/energy benchmarking.
    """
    def __init__(self, profile):
        self.key = Fernet.generate_key()
        self.cipher = Fernet(self.key)
        self.profile = profile

    def encrypt(self, plaintext: str) -> Tuple[bytes, float]:
        start = time.perf_counter()
        time.sleep(self.profile.base_latency_ms / 1000.0)
        ct = self.cipher.encrypt(plaintext.encode())
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return ct, elapsed_ms

def estimate_energy(elapsed_ms: float, cpu_coeff: float, energy_unit_per_ms: float):
    return elapsed_ms * cpu_coeff * energy_unit_per_ms
