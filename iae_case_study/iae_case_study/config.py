from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class EncryptionProfile:
    base_latency_ms: float
    cpu_coeff: float

@dataclass
class Config:
    random_seed: int = 42
    n_samples: int = 1000
    test_size: float = 0.3
    intent_set: List[str] = field(default_factory=lambda: [
        "emergency",
        "confidential-transfer",
        "routine-report",
        "diagnostic",
        "maintenance"
    ])
    intent_prior: List[float] = field(default_factory=lambda: [0.12, 0.20, 0.45, 0.13, 0.10])
    label_noise_prob: float = 0.05
    n_runs_per_policy: int = 30
    energy_unit_per_ms: float = 0.001
    encryption_profiles: Dict[str, EncryptionProfile] = field(default_factory=lambda: {
        "low": EncryptionProfile(base_latency_ms=2.0, cpu_coeff=0.8),
        "medium": EncryptionProfile(base_latency_ms=4.0, cpu_coeff=1.2),
        "high": EncryptionProfile(base_latency_ms=7.0, cpu_coeff=2.0),
        "abe_sim": EncryptionProfile(base_latency_ms=20.0, cpu_coeff=6.0),
    })

cfg = Config()
