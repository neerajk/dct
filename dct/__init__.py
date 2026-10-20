"""DCT. The env vars below must be set before torch is imported anywhere in the package."""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")        # OpenMP clash on this machine
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")    # unsupported MPS ops fall back to CPU
# Cloud-optimised GeoTIFF reads: skip directory listings, retry transient HTTP errors (speeds up STAC chip reads).
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "3")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "1")
os.environ.setdefault("VSI_CACHE", "TRUE")
