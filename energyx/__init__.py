"""EnergyX - a multi-agent framework for conversational energy data analytics."""
import os as _os

# Pin the BLAS thread pools before numpy/scipy are imported anywhere in the
# package -- they size their pools once, at import, and nothing can resize
# them afterwards.
#
# This looks like giving up parallelism and is the opposite. The heavy numeric
# work here is small-matrix: a SARIMAX fit over 28 days of hourly data, a
# k-means over a hundred 24-point profiles. Spread across every core, the
# threads spend their time synchronising rather than computing -- one measured
# forecast origin costs 33 s unpinned and 7.7 s pinned, on the same machine
# and the same data. Where real parallelism is available it belongs across
# buildings, which are independent (see energyx_truth/common.pmap).
#
# setdefault, so an operator who has deliberately set these keeps their value.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    _os.environ.setdefault(_var, "1")

__version__ = "0.1.0"
