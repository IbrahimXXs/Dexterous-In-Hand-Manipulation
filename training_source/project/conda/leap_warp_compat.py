"""Compatibility hook: restore ``warp.context`` for MJLab v1.1.1 on Warp >= 1.12.

MJLab v1.1.1 reads ``wp.context.runtime.driver_version`` (``mjlab/sim/sim.py``)
and annotates ``wp.context.Device``.  Warp 1.12+ moved that module to
``warp._src.context`` and no longer binds ``warp.context`` on import, so every
CUDA run crashes with ``AttributeError: module 'warp' has no attribute
'context'``.  (CPU runs never reach that line, which is why the original macOS
setup did not notice.)

Installed into the conda env as ``leap_warp_compat.pth`` + this module.  It
patches lazily: Warp is *not* imported at interpreter start-up; the attribute
is bound right after the first ``import warp`` finishes.
"""

import importlib.abc
import importlib.util
import sys


class _WarpContextFinder(importlib.abc.MetaPathFinder):
  def find_spec(self, name, path=None, target=None):
    if name != "warp":
      return None
    sys.meta_path.remove(self)
    spec = importlib.util.find_spec("warp")
    if spec is None or spec.loader is None:
      return spec
    original = spec.loader.exec_module

    def exec_module(module):
      original(module)
      if "context" not in module.__dict__:
        from warp._src import context as _context

        module.context = _context

    spec.loader.exec_module = exec_module
    return spec


if not any(isinstance(f, _WarpContextFinder) for f in sys.meta_path):
  sys.meta_path.insert(0, _WarpContextFinder())
