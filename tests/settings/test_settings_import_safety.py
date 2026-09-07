from __future__ import annotations

import os
import subprocess
import sys


FORBIDDEN = ("torch", "torchvision", "triton", "transformers", "flashinfer", "sgl_kernel")


def test_settings_package_imports_without_gpu_libraries():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    body = f'''
import importlib
import sys

FORBIDDEN = {FORBIDDEN!r}
class Blocker:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in FORBIDDEN:
            raise ImportError("forbidden import: " + name)
        return None
sys.meta_path.insert(0, Blocker())
for name in [
    "freetoken.daemon.settings",
    "freetoken.daemon.settings.dials",
    "freetoken.daemon.settings.boot_parser",
    "freetoken.daemon.settings.process_manager",
    "freetoken.daemon.settings.governor",
    "freetoken.daemon.settings.memory_fit",
    "freetoken.daemon.settings.profiles_manager",
    "freetoken.daemon.settings.app",
    "freetoken.daemon.settings.server",
]:
    importlib.import_module(name)
assert not any(name.split(".")[0] in FORBIDDEN for name in sys.modules)
'''
    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.join(root, "python")
    proc = subprocess.run([sys.executable, "-c", body], cwd=root, env=env, text=True, capture_output=True)
    assert proc.returncode == 0, proc.stderr
