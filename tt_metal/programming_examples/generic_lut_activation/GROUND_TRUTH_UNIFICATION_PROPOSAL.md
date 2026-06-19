# Ground Truth Unification Proposal

## Problem Statement

Ground truth activation function implementations exist in 3 places:
1. **Polynomial Fitter**: `/localdev/nkapre/tt-polynomial-fitter/sollya_expressions.py` (LUT generation)
2. **Polynomial Fitter**: `/localdev/nkapre/tt-polynomial-fitter/ground_truth.py` (outdated, only 5 functions)
3. **TT-Metal**: `tt_metal/programming_examples/generic_lut_activation/sweep_best.sh` (hardware validation)

**Issues**:
- Formula mismatches cause incorrect error measurements
- Maintaining 3 separate implementations is error-prone
- No single source of truth

## Design Goals

1. **Single Source of Truth**: One canonical implementation
2. **Easy to Update**: Adding new activations should be straightforward
3. **Language Agnostic**: Usable from Python (polynomial fitter) and Bash (sweep scripts)
4. **Version Control**: Changes tracked in git
5. **Low Friction**: Minimal build/install complexity

## Proposed Solutions

### Option 1: Shared Python Package (Recommended for Production)

**Structure**:
```
tt-activation-reference/
├── setup.py
├── pyproject.toml
├── README.md
└── tt_activation_reference/
    ├── __init__.py
    ├── activations.py          # Core implementations
    └── activations_config.dat  # Configuration (ranges, etc.)
```

**Implementation**:
```python
# tt_activation_reference/activations.py
import numpy as np
from scipy.special import erf

class ActivationReference:
    """Canonical reference implementations for activation functions."""

    @staticmethod
    def gelu(x):
        """GELU (Gaussian Error Linear Unit) - exact formula"""
        return x * 0.5 * (1.0 + erf(x / np.sqrt(2.0)))

    @staticmethod
    def hardsigmoid(x):
        """HardSigmoid - formula: clip(0.2*x + 0.5, 0, 1)"""
        return np.maximum(0, np.minimum(1, 0.2 * x + 0.5))

    # ... all other activations ...

    @classmethod
    def get_function(cls, name):
        """Get activation function by name."""
        return getattr(cls, name)
```

**Usage in Polynomial Fitter**:
```python
pip install tt-activation-reference
from tt_activation_reference import ActivationReference

func = ActivationReference.get_function('gelu')
y = func(x)
```

**Usage in TT-Metal sweep_best.sh**:
```bash
# Install package in virtual environment
pip install tt-activation-reference

# Use in Python embedded in bash
python3 << EOF
from tt_activation_reference import ActivationReference
import sys

activation = sys.argv[1]
x = float(sys.argv[2])
func = ActivationReference.get_function(activation)
print(func(x))
EOF
```

**Pros**:
- Clean separation of concerns
- Pip-installable, version-pinnable
- Works across repos without duplication
- Can be published to PyPI or private registry

**Cons**:
- Requires packaging setup
- Need to publish/install for usage
- Versioning coordination between repos

---

### Option 2: Git Submodule (Recommended for Development)

**Structure**:
```
tt-polynomial-fitter/         # Main repo
├── ground_truth/
│   ├── activations.py
│   ├── activations_config.dat
│   └── README.md
└── ...

tt-metal/                     # Uses submodule
├── external/
│   └── activation-ground-truth/  # <- Git submodule
└── tt_metal/programming_examples/generic_lut_activation/
    └── sweep_best.sh             # Sources from ../../../external/
```

**Setup**:
```bash
# In tt-metal repo
cd /localdev/nkapre/tt-metal
git submodule add ../tt-polynomial-fitter/ground_truth external/activation-ground-truth

# In sweep_best.sh
GROUND_TRUTH_DIR="$TT_METAL_HOME/external/activation-ground-truth"
source "$GROUND_TRUTH_DIR/activations.sh"  # Bash wrapper
```

**Pros**:
- Simple git-native solution
- Automatic version tracking
- No packaging complexity
- Easy to develop/update

**Cons**:
- Submodules can be confusing
- Requires `git submodule update` after pulls
- Both repos must be on same filesystem

---

### Option 3: Symlink (Quick Development Hack)

**Implementation**:
```bash
# In tt-metal repo
cd tt_metal/programming_examples/generic_lut_activation
ln -s /localdev/nkapre/tt-polynomial-fitter/ground_truth.py ground_truth.py

# In sweep_best.sh
python3 << EOF
import sys
sys.path.insert(0, '.')
from ground_truth import ActivationReference
# ...
EOF
```

**Pros**:
- Instant, zero setup
- Changes immediately reflected

**Cons**:
- Breaks on different machines/paths
- Doesn't work for deployment
- Git doesn't track symlink targets
- Only works on Unix-like systems

---

### Option 4: Automated Sync Script

**Implementation**:
```bash
#!/bin/bash
# sync_ground_truth.sh

SOURCE="/localdev/nkapre/tt-polynomial-fitter/ground_truth.py"
DEST="/localdev/nkapre/tt-metal/tt_metal/programming_examples/generic_lut_activation/ground_truth.py"

if [[ "$SOURCE" -nt "$DEST" ]]; then
    cp "$SOURCE" "$DEST"
    echo "✓ Synced ground truth from polynomial-fitter"
else
    echo "✓ Ground truth already up to date"
fi
```

**Pros**:
- Simple, no external dependencies
- Both repos have independent copies
- Works on any filesystem

**Cons**:
- Must remember to run sync script
- Easy to forget and diverge
- Manual conflict resolution if both change

---

## Recommended Hybrid Approach

**Development** (Now):
1. Use **Option 2 (Git Submodule)** or **Option 3 (Symlink)** for immediate iteration
2. Both repos point to single source in polynomial-fitter

**Production** (Later):
1. Package as **Option 1 (Shared Python Package)**
2. Publish to internal PyPI or as git+https dependency
3. Pin versions in requirements.txt

## Implementation Plan

### Phase 1: Consolidate in Polynomial Fitter (This Week)

**Step 1**: Create canonical ground_truth module in polynomial-fitter
```bash
cd /localdev/nkapre/tt-polynomial-fitter

# Create new ground_truth module
mkdir -p tt_activation_reference
cat > tt_activation_reference/__init__.py << 'EOF'
from .activations import ActivationReference
__all__ = ['ActivationReference']
EOF

cat > tt_activation_reference/activations.py << 'EOF'
import numpy as np
from scipy.special import erf

class ActivationReference:
    # Implementations from sollya_expressions ACTIVATION_FUNCTIONS
    # ...
EOF

# Copy activations_config.dat
cp activations_config.dat tt_activation_reference/
```

**Step 2**: Update polynomial fitter to use new module
```python
# In segmentation/*.py
from tt_activation_reference import ActivationReference

# Replace: ACTIVATION_FUNCTIONS['gelu']
# With: ActivationReference.gelu
```

**Step 3**: Create symlink in tt-metal for development
```bash
cd /localdev/nkapre/tt-metal/tt_metal/programming_examples/generic_lut_activation
ln -s /localdev/nkapre/tt-polynomial-fitter/tt_activation_reference ground_truth

# Update sweep_best.sh to import from ground_truth
```

### Phase 2: Package for Distribution (Next Month)

**Step 1**: Add packaging files to polynomial-fitter
```bash
cd /localdev/nkapre/tt-polynomial-fitter

cat > pyproject.toml << 'EOF'
[build-system]
requires = ["setuptools>=45", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "tt-activation-reference"
version = "0.1.0"
description = "Reference implementations for activation functions"
dependencies = [
    "numpy>=1.20",
    "scipy>=1.7",
]
EOF
```

**Step 2**: Install as editable package
```bash
pip install -e /localdev/nkapre/tt-polynomial-fitter
```

**Step 3**: Update tt-metal requirements.txt
```
tt-activation-reference @ file:///localdev/nkapre/tt-polynomial-fitter
```

### Phase 3: Make Package Portable (When Needed)

**Option A**: Publish to PyPI
```bash
python -m build
twine upload dist/*
```

**Option B**: Use git dependency
```txt
# requirements.txt
tt-activation-reference @ git+https://github.com/tenstorrent/tt-activation-reference.git@v0.1.0
```

## File Organization

### Polynomial Fitter Repo
```
tt-polynomial-fitter/
├── tt_activation_reference/         # NEW: Shared package
│   ├── __init__.py
│   ├── activations.py               # Core implementations
│   ├── activations_config.dat       # Configuration
│   └── README.md                    # API documentation
├── sollya_expressions.py            # LEGACY: Kept for backward compat
├── ground_truth.py                  # LEGACY: Deprecated
├── pyproject.toml                   # NEW: Packaging config
└── setup.py                         # NEW: Packaging config
```

### TT-Metal Repo
```
tt-metal/
├── external/                        # NEW: External dependencies
│   └── activation-ground-truth/     # Submodule OR symlink
├── tt_metal/programming_examples/generic_lut_activation/
│   ├── sweep_best.sh                # UPDATED: Import from ground_truth
│   ├── activations_config_loader.sh # UPDATED: Use shared config
│   └── ground_truth -> ../../external/activation-ground-truth/
└── requirements.txt                 # UPDATED: Add tt-activation-reference
```

## API Design

### Core API
```python
from tt_activation_reference import ActivationReference

# Get function by name
func = ActivationReference.get_function('gelu')
y = func(x)  # Numpy array or scalar

# Direct access
y = ActivationReference.gelu(x)

# List all activations
names = ActivationReference.list_activations()

# Get configuration
from tt_activation_reference import get_activation_config
config = get_activation_config('gelu')
# Returns: {'range_min': -10.0, 'range_max': 10.0, 'has_native_sfpu': True}
```

### Bash Helper API
```bash
# Source bash wrapper
source $(python3 -c "import tt_activation_reference; print(tt_activation_reference.__path__[0])")/bash_wrapper.sh

# Evaluate activation
result=$(eval_activation "gelu" 1.5)

# Get range
range_min=$(get_activation_range_min "gelu")
range_max=$(get_activation_range_max "gelu")
```

## Migration Path

1. **Week 1**: Create tt_activation_reference module in polynomial-fitter ✓
2. **Week 1**: Symlink from tt-metal for immediate use ✓
3. **Week 2**: Update all references in both repos
4. **Week 2**: Add tests to verify implementations match
5. **Week 3**: Add packaging config (pyproject.toml)
6. **Week 3**: Test editable install workflow
7. **Week 4**: Deprecate old ground_truth.py and sollya_expressions.ACTIVATION_FUNCTIONS
8. **Month 2**: Publish package (PyPI or git+https)

## Testing Strategy

```python
# tests/test_ground_truth_consistency.py
import numpy as np
from tt_activation_reference import ActivationReference

def test_gelu_consistency():
    """Verify GELU matches expected formula."""
    x = np.array([-2, -1, 0, 1, 2])
    y = ActivationReference.gelu(x)

    # Expected values computed independently
    expected = np.array([...])

    np.testing.assert_allclose(y, expected, rtol=1e-6)

def test_all_activations_exist():
    """Verify all activations in config are implemented."""
    required = ['gelu', 'hardsigmoid', 'tanh', 'sigmoid', ...]

    for name in required:
        assert hasattr(ActivationReference, name)
        func = ActivationReference.get_function(name)
        assert callable(func)
```

## Documentation

Create comprehensive docs:
- API reference
- Formula definitions with LaTeX
- Range specifications
- Usage examples
- Migration guide

## Backward Compatibility

During transition:
1. Keep old `ground_truth.py` and `sollya_expressions.py`
2. Add deprecation warnings
3. Internal imports redirect to new module
4. Remove after 1 month grace period
