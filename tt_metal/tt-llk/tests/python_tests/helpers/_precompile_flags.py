# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# Set True by conftest.py's controller-side pre-compile pass (around the
# ThreadPoolExecutor in pytest_collection_finish).  Checked by generate_stimuli()
# and get_golden_generator() to return cheap dummy values — the real stimulus
# content is irrelevant during compilation because variant_stimuli is excluded
# from generate_variant_hash() in non-SPEED_OF_LIGHT mode.
ACTIVE: bool = False
