"""Put `paperfigs` on the path so `paper/` runs without installing it."""

import os
import sys
from pathlib import Path

# Chronos-2 never detects a device (`chronos2.torch_device`): the tests
# run on the CPU unless the environment names one, as CI does.
os.environ.setdefault("DEVICE", "cpu")

sys.path.insert(0, str(Path(__file__).parent))
