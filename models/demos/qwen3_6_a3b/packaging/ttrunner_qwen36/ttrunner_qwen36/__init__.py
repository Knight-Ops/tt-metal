# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Self-contained tt-kernel dispatch runner for Qwen3.6-35B-A3B on Tenstorrent Blackhole."""

__version__ = "0.1.0"

# NOTE: importing Qwen36Runner here would transitively `import ttnn`/`import ttl` at package
# import time. Keep this module import-light; import the runner explicitly from its module:
#     from ttrunner_qwen36.runner import Qwen36Runner
