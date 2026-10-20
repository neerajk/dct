"""DCT. The env vars below must be set before torch is imported anywhere in the package."""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")        # OpenMP clash on this machine
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")    # unsupported MPS ops fall back to CPU
