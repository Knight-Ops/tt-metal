#!/bin/bash
# Helper functions for extracting timing from device profiler CSVs

# Extract SFPU compute time (TRISC_1 math kernel) from profile_log_device.csv
# Args:
#   $1 - Path to profile_log_device.csv
# Returns:
#   Compute time in microseconds (or "0" on error)
extract_profiler_compute_time() {
    local profiler_csv="$1"
    local work_dir="${2:-$WORK_DIR}"

    if [[ ! -f "$profiler_csv" ]]; then
        echo "0"
        return 1
    fi

    python3 -c "
import sys
sys.path.insert(0, '$work_dir')
from extract_profiler_timing import extract_compute_time
time = extract_compute_time('$profiler_csv')
print(f'{time:.2f}' if time else '0')
" 2>&1
}

# Extract per-shape SFPU compute times from a batch profiler CSV
# In batch mode, multiple shapes run sequentially; this clusters them by time gaps
# Args:
#   $1 - Path to profile_log_device.csv (single CSV with all shapes' data)
#   $2 - Number of shapes in the batch
# Returns:
#   Comma-separated compute times in microseconds, one per shape (in order)
#   e.g. "5.51,5.55,11.84,139.42,2217.34"
extract_batch_profiler_times() {
    local profiler_csv="$1"
    local num_shapes="$2"
    local work_dir="${3:-$WORK_DIR}"

    if [[ ! -f "$profiler_csv" ]]; then
        # Return num_shapes zeros
        python3 -c "print(','.join(['0']*$num_shapes))"
        return 1
    fi

    python3 -c "
import sys
sys.path.insert(0, '$work_dir')
from extract_profiler_timing import extract_batch_compute_times
times = extract_batch_compute_times('$profiler_csv', $num_shapes)
print(','.join(f'{t:.2f}' for t in times))
" 2>&1
}

# Extract full timing breakdown from profile_log_device.csv
# Args:
#   $1 - Path to profile_log_device.csv
# Returns:
#   Space-separated: reader_us unpack_us math_us pack_us writer_us total_us
extract_profiler_full_timing() {
    local profiler_csv="$1"
    local work_dir="${2:-$WORK_DIR}"

    if [[ ! -f "$profiler_csv" ]]; then
        echo "0 0 0 0 0 0"
        return 1
    fi

    python3 -c "
import sys
sys.path.insert(0, '$work_dir')
from extract_profiler_timing import parse_profiler_csv, summarize_timing

try:
    profiler_data = parse_profiler_csv('$profiler_csv')
    summary = summarize_timing(profiler_data)
    print(f\"{summary['reader_us']:.2f} {summary['unpack_us']:.2f} {summary['math_us']:.2f} {summary['pack_us']:.2f} {summary['writer_us']:.2f} {summary['total_us']:.2f}\")
except Exception as e:
    print('0 0 0 0 0 0', file=sys.stderr)
    sys.exit(1)
" 2>&1
}

# Wait for profiler CSV to be written (with timeout)
# Args:
#   $1 - Path to profiler CSV
#   $2 - Timeout in seconds (default: 10)
# Returns:
#   0 if file exists, 1 if timeout
wait_for_profiler_csv() {
    local profiler_csv="$1"
    local timeout="${2:-10}"

    for i in $(seq 1 $timeout); do
        if [[ -f "$profiler_csv" ]]; then
            return 0
        fi
        sleep 1
    done

    return 1
}
