#!/bin/bash
# Display 2D grid results from profiler or host timing sweep

csv_file="${1:-depth_degree_profiler_results/depth_degree_profiler_summary.csv}"

if [[ ! -f "$csv_file" ]]; then
    echo "Error: CSV file not found: $csv_file"
    exit 1
fi

# Detect if this is profiler or host timing data
if grep -q "profiler_compute_us" "$csv_file"; then
    time_col=6
    time_unit="µs"
    time_label="SFPU Compute Time"
else
    time_col=6
    time_unit="ms"
    time_label="Host Kernel Exec Time"
fi

echo "================================================================================"
echo "2D GRID: $time_label ($time_unit) @ 256 tiles"
echo "================================================================================"
echo ""
printf "%-10s" "Degree"
printf "%15s" "FP32"
printf "%15s" "BF16"
printf "%15s" "Ratio"
echo ""
printf "%-10s" "----------"
printf "%15s" "---------------"
printf "%15s" "---------------"
printf "%15s" "---------------"
echo ""

for degree in 1 2 4 6 8; do
    fp32_time=$(grep "^${degree},16,fp32,256_tiles" "$csv_file" | cut -d',' -f${time_col})
    bf16_time=$(grep "^${degree},16,bf16,256_tiles" "$csv_file" | cut -d',' -f${time_col})

    if [[ -n "$fp32_time" && -n "$bf16_time" ]]; then
        ratio=$(python3 -c "bf16=$bf16_time; fp32=$fp32_time; print(f'{bf16/fp32:.3f}')")
        printf "%-10s" "p${degree}_s16"
        printf "%15s" "${fp32_time}${time_unit}"
        printf "%15s" "${bf16_time}${time_unit}"
        printf "%15s" "${ratio}×"
        echo ""
    fi
done

echo ""
echo "================================================================================"
echo "SCALING ACROSS TILE SHAPES (Degree 6, FP32)"
echo "================================================================================"
echo ""
printf "%-20s %10s %15s\n" "Shape" "Tiles" "Time"
printf "%-20s %10s %15s\n" "--------------------" "----------" "---------------"

for shape in single_tile 8_tiles 256_tiles height_sharded yolov4; do
    line=$(grep "^6,16,fp32,${shape}," "$csv_file")
    if [[ -n "$line" ]]; then
        tiles=$(echo "$line" | cut -d',' -f5)
        time=$(echo "$line" | cut -d',' -f${time_col})
        printf "%-20s %10s %15s%s\n" "$shape" "$tiles" "$time" "$time_unit"
    fi
done

echo ""
