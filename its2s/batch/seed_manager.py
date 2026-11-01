# Description: Re-export of its2s.seeds.derive_seed for the batch runner;
#   deleted with run_batch in P11.04.07 (spec_P11.04.03 OQ-10).
# Usage: from its2s.batch.seed_manager import derive_seed
# Dependencies: none

from ..seeds import derive_seed

__all__ = ["derive_seed"]
