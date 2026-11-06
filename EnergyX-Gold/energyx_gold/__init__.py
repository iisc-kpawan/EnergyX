"""Reference implementations and gold answers for the EnergyX-Gold benchmark."""

import os

# One BLAS/OpenMP thread, so ML tasks give the same numbers on every run.
# This has to happen before numpy is first imported.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
