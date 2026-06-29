import os
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

from .xlit_src import XlitEngine
from .__metadata import *
