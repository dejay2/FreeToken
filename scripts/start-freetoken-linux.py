#!/usr/bin/env python3
"""Start a FreeToken server on Linux / WSL with the Windows launcher's parameter spelling.

    python scripts/start-freetoken-linux.py -ModelPath ~/models/Qwen3.8-Flash-Next-NVFP4 \
        -Port 2020 -ContextTokens 65536 -KVCacheTokens 65536 -MaxRunningRequests 1 \
        -MoECacheSize 6750 -MoEVramReserveBytes 0 -MoECacheHeadroomBytes 0 \
        -DenseQuant int8 -EmbedHost -ExpertLoad parallel -EnableCacheReport [-DryRun]

The settings helper uses the same mapping (``freetoken.daemon.settings.linux_launch``) when it
boots a profile on Linux, so a profile that boots from the page boots identically from here.
Run it from the checkout's venv (``.venv/bin/python``); it execs ``ft serve`` in place.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from freetoken.daemon.settings.linux_launch import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
