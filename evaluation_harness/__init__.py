# Modified from the original version: Import from our new helper functions that use original provider system
from .evaluators import *
import os
import sys
# Add src to path to import our helper functions
src_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from utils.webarena_evaluation_helper_functions import (
    shopping_get_latest_order_url,
    shopping_get_sku_latest_review_author,
    shopping_get_sku_latest_review_rating,
)
