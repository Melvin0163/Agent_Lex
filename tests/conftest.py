"""Make the repository root importable, so `import legal_agent` works when pytest runs from anywhere."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
