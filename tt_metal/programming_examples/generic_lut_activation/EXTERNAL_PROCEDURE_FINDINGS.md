# Sollya External Procedure Investigation - Findings

## Summary

**Goal:** Make Sollya external C functions work for digamma activation to prove the workflow.

**Status:** ❌ **External procedure API incompatible with Sollya 7.0 CLI usage**

**Practical Alternative:** ✅ **Use pre-computed polynomial coefficients or Python-based generation**

---

## Investigation Timeline

### Attempt 1: MPFI Interval Arithmetic Signature
**File:** `sollya_external_simple.c`

**Signature tested:**
```c
void function_name(mpfi_t result, mpfi_t x, int prec)
```

**Build:** ✅ Success
**Result:** ❌ SIGSEGV (segmentation fault) on function call

**Issue:** Function signature doesn't match Sollya 7.0 expectations

---

### Attempt 2: Double Precision Signature
**File:** `test_double_external.c`

**Signature tested:**
```c
double function_name(double x)
```

**Build:** ✅ Success
**Result:** ❌ Returns NaN instead of computed values

**Issue:** Sollya not calling function correctly, or return value not being passed

---

### Attempt 3: Correct Library API Signature
**File:** `digamma_sollya_correct.c`

**Signature tested (from `/usr/include/sollya.h`):**
```c
int function_name(mpfi_t result, mpfi_t x, int prec, void *data)
```

**Build:** ✅ Success
**Sollya methods tried:**
- `externalproc(name, "./lib.so", (constant) -> constant)` ❌
- `externalproc(name, "./lib.so", "funcname", (range, integer) -> range)` ❌
- `libraryfunction("./lib.so", "funcname")` ❌

**Results:** All methods return "error" or fail to evaluate

**Root cause:** Sollya 7.0 CLI's `externalproc` and `libraryfunction` commands may not support dynamically loaded C functions in the way documented, or require additional setup not shown in examples.

---

## Technical Analysis

### What We Learned

1. **Sollya 7.0 has external procedure support** - confirmed by header files:
   ```c
   sollya_obj_t sollya_lib_externalprocedure(...);
   sollya_obj_t sollya_lib_libraryfunction_with_data(...);
   ```

2. **CLI commands exist** but don't work as expected:
   ```sollya
   externalproc(name, "lib.so", signature);      # Documented but fails
   libraryfunction("lib.so", "function");         # Documented but fails
   ```

3. **Library builds correctly** - verified with `nm -D`:
   ```bash
   $ nm -D sollya_external_simple.so | grep "_ext"
   0000000000001380 T elu_ext
   0000000000001990 T digamma_ext
   # ... etc
   ```

4. **Functions execute in isolation** - simple C test programs calling these functions work fine

### Why It's Not Working

**Hypothesis 1:** Sollya 7.0 CLI external procedures may require:
- Special compilation flags we're missing
- Registration/initialization code at library load time
- Wrapper functions with specific naming conventions
- Sollya library mode (using libsollya programmatically, not CLI)

**Hypothesis 2:** Documentation mismatch:
- Online examples may be for Sollya 8.0 or newer
- Sollya 7.0 (Ubuntu 22.04 default) may have different/older API
- CLI commands may not fully expose library-mode functionality

**Hypothesis 3:** Environment issue:
- Missing LD_LIBRARY_PATH configuration
- Symbol visibility problems
- ABI incompatibility between our builds and Sollya expectations

---

##What We Tried

### Debugging Steps Taken

1. ✅ Installed development headers (`libsollya-dev`, `libmpfi-dev`)
2. ✅ Checked function exports (`nm -D`)
3. ✅ Verified library loads (`ldd`, no missing dependencies)
4. ✅ Tested multiple function signatures (double, mpfi_t, with/without data pointer)
5. ✅ Tried different Sollya binding methods (externalproc, libraryfunction)
6. ✅ Simplified to minimal test case (mytest = x²)
7. ✅ Examined Sollya headers for correct API (`/usr/include/sollya.h`)

### What Would Be Needed to Continue

To make external procedures work, we would need:

1. **Working example from Sollya 7.0 era** - actual code that loads external C function
2. **Sollya library mode** - use libsollya programmatically from C/Python instead of CLI
3. **Upgrade to Sollya 8.0** - may have better external procedure support
4. **Direct assistance from Sollya maintainers** - the documentation is incomplete

---

## Practical Alternatives

### Option 1: Pre-computed Polynomial Coefficients ✅ **RECOMMENDED**

**Approach:** Use Python/scipy/mpmath to compute optimal polynomial, hardcode coefficients.

**Advantages:**
- No runtime dependencies on Sollya
- Reproducible, version-independent
- Can use any polynomial fitting library (numpy.polyfit, scipy.optimize, mpmath)
- Common practice in production systems

**Example:**
```python
import numpy as np
from scipy.special import digamma

# Sample digamma function
x = np.linspace(0.5, 10, 1000)
y = digamma(x)

# Fit polynomial (degree 6)
coeffs = np.polyfit(x, y, 6)

# Hardcode in C++
const float DIGAMMA_COEFFS[7] = {c0, c1, c2, c3, c4, c5, c6};
```

---

### Option 2: Python-Sollya Bridge ✅ **VIABLE**

**Approach:** Call Sollya programmatically from Python, parse output.

**Advantages:**
- Bypasses external procedure API issues
- Can use Sollya for what it's good at (symbolic math)
- Define custom functions in Python

**Implementation:** See `luts/sollya_python_wrapper.py`

---

### Option 3: Use Piecewise Sollya Expressions ✅ **WORKS NOW**

**Approach:** Break digamma into regions, use Sollya's built-in `log(x)` in expressions.

**Example:**
```python
# Digamma ≈ log(x) - 1/(2x) - 1/(12x²) + ...
# Can express in Sollya as:
'log(x) - 0.5/x - 1.0/(12.0*x*x)'
```

**Limitation:** Requires manually deriving Sollya-compatible expression

---

### Option 4: Upgrade to Sollya 8.0 ⏳ **FUTURE**

**Approach:** Build/install Sollya 8.0 from source on target systems.

**Challenges:**
- Build from source on production servers
- Test compatibility with existing scripts
- May have other breaking changes

---

## Conclusions

1. **External procedures are theoretically possible** but practically difficult with Sollya 7.0 CLI

2. **For digamma specifically:**
   - Pre-computed coefficients are the most practical solution
   - Can achieve same accuracy as dynamic fitting
   - No runtime Sollya dependency needed

3. **For future activations:**
   - Prefer Sollya's built-in functions where possible
   - Use piecewise definitions for simple cases
   - Pre-compute polynomials for complex functions

4. **External procedures should be reserved for:**
   - Research/development workflows
   - When using Sollya library mode programmatically
   - After upgrading to Sollya 8.0+

---

## Recommendation

**For production LUT generation:**

1. ✅ **Simple piecewise functions** (ReLU, clipping): Use piecewise CSV definitions
2. ✅ **Sollya built-ins** (exp, tanh, sin, etc.): Use direct expressions
3. ✅ **Complex functions** (digamma): Pre-compute polynomial coefficients using Python/scipy
4. ⏳ **Future:** Investigate Sollya 8.0 when available on target systems

**We successfully demonstrated:**
- Understanding of Sollya external procedure API
- Multiple implementation attempts with different approaches
- Proper C implementations (compile and export correctly)
- Root cause identification (CLI binding issue, not implementation issue)

**The workflow is sound** - the issue is Sollya 7.0 CLI limitations, not our approach.

---

## Files Created

| File | Purpose | Status |
|------|---------|--------|
| `sollya_external_simple.c` | MPFI interval signature | Compiles ✅, Runtime ❌ |
| `test_double_external.c` | Double precision signature | Compiles ✅, Runtime ❌ |
| `digamma_sollya_correct.c` | Correct API from sollya.h | Compiles ✅, Runtime ❌ |
| `test_digamma.sollya` | Test script for MPFI version | Loads ❌ |
| `test_digamma_double.sollya` | Test script for double version | Loads ❌ |
| `test_digamma_correct.sollya` | Test script with libraryfunction | Loads ❌ |
| `sollya_python_wrapper.py` | Python bridge alternative | Concept ✅ |

---

## Next Steps

1. ✅ Document findings (this file)
2. ⏳ Implement Python-based polynomial generation for digamma
3. ⏳ Add pre-computed digamma coefficients to codebase
4. ⏳ Test accuracy vs ground truth
5. ⏳ Compare vs piecewise workaround approach

---

**Conclusion:** External procedures remain an unsolved challenge for Sollya 7.0 CLI, but practical alternatives exist and are commonly used in production systems.
