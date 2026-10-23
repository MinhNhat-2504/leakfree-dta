"""Run the patched DeepDTA source under WSL2 TensorFlow GPU.

TensorFlow 2.16 ships Keras 3 by default; the published DeepDTA source uses
the legacy Keras 2 session API.  ``tf_keras`` is injected as the ``keras``
module without changing the model source or its architecture.
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: run_deepdta_wsl.py SOURCE [ARGS ...]")
    source = Path(sys.argv[1]).resolve()
    os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

    import tf_keras  # type: ignore

    sys.modules["keras"] = tf_keras
    sys.path.insert(0, str(source.parent))
    sys.argv = [str(source), *sys.argv[2:]]
    runpy.run_path(str(source), run_name="__main__")


if __name__ == "__main__":
    main()
