# Legend Organization - Theoretical Error Plots

## New Legend Layout: Organized by Polynomial Degree

The legend has been reorganized to make it much more intuitive and easier to read. Each column now represents one polynomial degree, with different depths stacked vertically within that column.

## Visual Layout

### Before (Random Order, 6 columns max)
```
Linear (depth=4)  │ Quadratic Remez (depth=4)  │ Cubic Remez (depth=8)    │ ...
Linear (depth=8)  │ Quadratic Remez (depth=8)  │ Cubic Remez (depth=16)   │ ...
```
*Legend entries were added in iteration order, making it hard to compare across degrees*

### After (Organized by Degree)
```
┌─────────────────┬──────────────────┬──────────────────┬──────────────────┬──────────────────┐
│    Linear       │  Quadratic Remez │   Cubic Remez    │   Hexic Remez    │   Octic Remez    │
├─────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┤
│  depth=4        │    depth=4       │    depth=4       │    depth=4       │    depth=4       │
│  depth=8        │    depth=8       │    depth=8       │    depth=8       │    depth=8       │
│  depth=16       │    depth=16      │    depth=16      │    depth=16      │    depth=16      │
│  depth=32       │    depth=32      │    depth=32      │    depth=32      │    depth=32      │
└─────────────────┴──────────────────┴──────────────────┴──────────────────┴──────────────────┘
```
*Each column = one polynomial degree, depths increase from top to bottom*

## Key Changes

### 1. Sorted by Method (Degree) First
Legend entries are now sorted by:
1. **Primary**: Polynomial degree (Constant → Linear → Quadratic → Cubic → Hexic → Octic)
2. **Secondary**: Depth (4 → 8 → 16 → 32)

### 2. One Column Per Degree
```python
ncol=unique_methods  # One column per polynomial degree
```

Instead of arbitrary 6 columns, the legend now has exactly as many columns as there are polynomial degrees in the plot (typically 5-7).

### 3. Increased Column Spacing
```python
columnspacing=2.0  # More space between degree columns
```

Makes it easier to visually separate different polynomial degrees.

### 4. Wider Legend Area
```python
bbox_to_anchor=(0.5, 1.05)  # Slightly higher
plt.tight_layout(rect=[0, 0, 1, 0.88])  # More top space
```

The legend now spans the full width of the figure with proper spacing.

## Benefits

### ✅ Easier to Compare Degrees
You can immediately see all variants (depths) of each polynomial degree grouped together.

**Example**: Want to compare Linear vs Cubic at depth=8?
- Look at the "Linear" column, find depth=8
- Look at the "Cubic Remez" column, find depth=8
- Colors are consistent across the plot

### ✅ Logical Organization
Matches how you think about the data:
1. "What polynomial degree should I use?" → Look across columns
2. "What depth should I use?" → Look down rows within a column

### ✅ Consistent Color Mapping
Each polynomial degree has its own color:
- Linear: Blue
- Quadratic: Green
- Quadratic Remez: Orange
- Cubic Remez: Red
- Hexic Remez: Cyan
- Octic Remez: Purple

Different depths of the same degree vary in opacity/alpha.

### ✅ Scalable Layout
- If you have 3 degrees → 3 columns
- If you have 7 degrees → 7 columns
- Legend width automatically adapts

## Example Reading the Legend

**Question**: "How does cubic depth=16 compare to linear depth=32?"

**Answer**:
1. Find "Cubic Remez" column, look for "depth=16" (3rd row)
2. Find "Linear" column, look for "depth=32" (4th row)
3. Match the colors/lines in the plot

## Implementation Details

### Sort Order
```python
method_order = [
    'Constant',
    'Linear',
    'Quadratic',
    'Quadratic Remez',
    'Cubic Remez',
    'Hexic Remez',
    'Octic Remez'
]
legend_entries.sort(key=lambda x: (
    method_order.index(x[0]) if x[0] in method_order else 999,  # Primary: method
    x[1]  # Secondary: depth
))
```

This ensures:
- Methods appear left-to-right in logical order
- Within each method, depths appear top-to-bottom in ascending order

### Dynamic Column Count
```python
unique_methods = len(set(entry[0] for entry in legend_entries))
ncol=unique_methods
```

Legend automatically adjusts to the number of polynomial degrees present in the data.

### Legend Styling
```python
fig.legend(
    legend_handles, legend_labels,
    loc='upper center',         # Centered at top
    bbox_to_anchor=(0.5, 1.05), # Slightly above plots
    ncol=unique_methods,        # One column per degree
    fontsize=8,                 # Readable but compact
    frameon=True,               # White background
    framealpha=0.9,             # Slightly transparent
    edgecolor='gray',           # Gray border
    columnspacing=2.0,          # Wide column spacing
    handlelength=1.5            # Longer legend markers
)
```

## Typical Legend Dimensions

For a plot with 5 polynomial degrees and 4 depths each:
- **Columns**: 5 (one per degree)
- **Rows**: 4 (one per depth)
- **Total entries**: 20
- **Width**: ~18 inches (spans most of 20" plot width)
- **Height**: ~1 inch (4 rows of text at fontsize=8)

## Usage

No changes needed! All existing commands work:

```bash
# Generate plots with organized legend
python3 plot_theoretical_error.py --activation sigmoid --simulate-bf16

# Sweep
./sweep_theoretical_error.sh --degree cubic --simulate-bf16

# All plots automatically get the new organized legend
```

## Testing

Verified with:
- ✅ sigmoid (multiple degrees, multiple depths)
- ✅ gelu (multiple degrees, multiple depths)
- ✅ tanh (multiple degrees, multiple depths)
- ✅ exp (large error range)

All legends display correctly with proper degree-based column organization.

## File Modified

- `plot_theoretical_error.py` - Updated legend collection and sorting logic in `plot_activation_error_analysis()` function

## Visual Improvement Summary

**Before**: Hard to find specific degree/depth combinations
**After**: Intuitive grid layout - scan columns for degree, scan rows for depth

This makes it much faster to:
- Compare different polynomial degrees
- Find specific depth configurations
- Understand the trade-offs between complexity and accuracy

🎯 **Result**: More professional, more intuitive, more useful legend layout!
