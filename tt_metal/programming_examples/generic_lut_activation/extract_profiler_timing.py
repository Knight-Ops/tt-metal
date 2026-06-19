#!/usr/bin/env python3
"""
Extract kernel timing from TT-Metal device profiler CSV output.

Usage:
    python3 extract_profiler_timing.py <profiler_csv>
    python3 extract_profiler_timing.py generated/profiler/.logs/profile_log_device.csv
"""

import sys
import csv
from pathlib import Path
from collections import defaultdict


def parse_profiler_csv(csv_path):
    """
    Parse TT-Metal device profiler CSV and extract kernel timing.

    Returns:
        dict: {
            'chip_freq_mhz': int,
            'cores': {
                (x, y): {
                    'BRISC': {'start': cycles, 'end': cycles, 'duration_us': float},
                    'NCRISC': {...},
                    'TRISC_0': {...},
                    'TRISC_1': {...},
                    'TRISC_2': {...},
                }
            }
        }
    """
    with open(csv_path, 'r') as f:
        # First line contains chip info
        first_line = f.readline()
        # Parse: "ARCH: blackhole, CHIP_FREQ[MHz]: 1350, Max Compute Cores: 120"
        chip_freq = None
        for part in first_line.split(','):
            if 'CHIP_FREQ[MHz]' in part:
                chip_freq = int(part.split(':')[1].strip())
                break

        if chip_freq is None:
            raise ValueError(f"Could not parse chip frequency from: {first_line}")

        # Read CSV data
        reader = csv.DictReader(f)

        # Organize data by core and RISC processor
        core_data = defaultdict(lambda: defaultdict(dict))

        for row in reader:
            # Strip whitespace from keys (CSV has spaces after commas)
            row = {k.strip(): v.strip() for k, v in row.items()}

            core_x = int(row['core_x'])
            core_y = int(row['core_y'])
            risc_type = row['RISC processor type']
            zone_name = row['zone name']
            zone_type = row['type']
            cycles = int(row['time[cycles since reset]'])

            # Only track kernel zones (not firmware)
            if 'KERNEL' not in zone_name:
                continue

            core_key = (core_x, core_y)

            if zone_type == 'ZONE_START':
                core_data[core_key][risc_type]['start'] = cycles
            elif zone_type == 'ZONE_END':
                core_data[core_key][risc_type]['end'] = cycles

    # Calculate durations
    result = {
        'chip_freq_mhz': chip_freq,
        'cores': {}
    }

    for core_key, risc_data in core_data.items():
        result['cores'][core_key] = {}
        for risc_type, timing in risc_data.items():
            if 'start' in timing and 'end' in timing:
                duration_cycles = timing['end'] - timing['start']
                duration_us = (duration_cycles / chip_freq)  # MHz = cycles/µs, so cycles / MHz = µs
                result['cores'][core_key][risc_type] = {
                    'start': timing['start'],
                    'end': timing['end'],
                    'duration_cycles': duration_cycles,
                    'duration_us': duration_us
                }

    return result


def summarize_timing(profiler_data):
    """
    Summarize timing across all cores.

    Returns:
        dict: {
            'reader_us': float (BRISC average),
            'compute_us': float (TRISC_1 average - math kernel),
            'writer_us': float (NCRISC average),
            'total_us': float (max end - min start),
        }
    """
    brisc_times = []
    ncrisc_times = []
    trisc0_times = []
    trisc1_times = []
    trisc2_times = []

    all_starts = []
    all_ends = []

    for core_key, risc_data in profiler_data['cores'].items():
        for risc_type, timing in risc_data.items():
            all_starts.append(timing['start'])
            all_ends.append(timing['end'])

            if risc_type == 'BRISC':
                brisc_times.append(timing['duration_us'])
            elif risc_type == 'NCRISC':
                ncrisc_times.append(timing['duration_us'])
            elif risc_type == 'TRISC_0':
                trisc0_times.append(timing['duration_us'])
            elif risc_type == 'TRISC_1':
                trisc1_times.append(timing['duration_us'])
            elif risc_type == 'TRISC_2':
                trisc2_times.append(timing['duration_us'])

    # Calculate total execution time (wall clock on device)
    if all_starts and all_ends:
        min_start = min(all_starts)
        max_end = max(all_ends)
        duration_cycles = max_end - min_start
        total_us = duration_cycles / profiler_data['chip_freq_mhz']  # MHz = cycles/µs
    else:
        total_us = 0

    return {
        'reader_us': sum(brisc_times) / len(brisc_times) if brisc_times else 0,
        'compute_us': sum(trisc1_times) / len(trisc1_times) if trisc1_times else 0,
        'writer_us': sum(ncrisc_times) / len(ncrisc_times) if ncrisc_times else 0,
        'unpack_us': sum(trisc0_times) / len(trisc0_times) if trisc0_times else 0,
        'math_us': sum(trisc1_times) / len(trisc1_times) if trisc1_times else 0,
        'pack_us': sum(trisc2_times) / len(trisc2_times) if trisc2_times else 0,
        'total_us': total_us,
        'num_cores': len(profiler_data['cores'])
    }


def print_detailed_report(profiler_data):
    """Print detailed per-core timing breakdown."""
    print(f"\nChip Frequency: {profiler_data['chip_freq_mhz']} MHz")
    print(f"Number of cores: {len(profiler_data['cores'])}")
    print("\n" + "=" * 100)
    print(f"{'Core':^10} {'RISC':^10} {'Duration (cycles)':>20} {'Duration (µs)':>15}")
    print("=" * 100)

    for core_key in sorted(profiler_data['cores'].keys()):
        risc_data = profiler_data['cores'][core_key]
        core_str = f"({core_key[0]},{core_key[1]})"

        for risc_type in ['BRISC', 'NCRISC', 'TRISC_0', 'TRISC_1', 'TRISC_2']:
            if risc_type in risc_data:
                timing = risc_data[risc_type]
                print(f"{core_str:^10} {risc_type:^10} {timing['duration_cycles']:>20,} {timing['duration_us']:>15.2f}")


def print_summary_report(summary):
    """Print summary timing report."""
    print("\n" + "=" * 60)
    print("KERNEL TIMING SUMMARY")
    print("=" * 60)
    print(f"{'Kernel Type':30} {'Time (µs)':>15} {'Time (ms)':>12}")
    print("-" * 60)
    print(f"{'Reader (BRISC)':30} {summary['reader_us']:>15.2f} {summary['reader_us']/1000:>12.3f}")
    print(f"{'Unpack (TRISC_0)':30} {summary['unpack_us']:>15.2f} {summary['unpack_us']/1000:>12.3f}")
    print(f"{'Math/Compute (TRISC_1)':30} {summary['math_us']:>15.2f} {summary['math_us']/1000:>12.3f}")
    print(f"{'Pack (TRISC_2)':30} {summary['pack_us']:>15.2f} {summary['pack_us']/1000:>12.3f}")
    print(f"{'Writer (NCRISC)':30} {summary['writer_us']:>15.2f} {summary['writer_us']/1000:>12.3f}")
    print("-" * 60)
    print(f"{'Total (wall clock on device)':30} {summary['total_us']:>15.2f} {summary['total_us']/1000:>12.3f}")
    print("=" * 60)
    print(f"\nNote: Kernels run in parallel. TRISC_1 (Math) is the SFPU compute kernel.")
    print(f"Number of cores: {summary['num_cores']}")


def extract_compute_time(csv_path):
    """
    Extract just the SFPU compute time from profiler CSV.

    Returns:
        float: Compute time in microseconds (TRISC_1 math kernel)
    """
    try:
        profiler_data = parse_profiler_csv(csv_path)
        summary = summarize_timing(profiler_data)
        return summary['math_us']
    except Exception:
        return None


def extract_batch_compute_times(csv_path, num_shapes):
    """
    Extract per-shape SFPU compute times from a batch profiler CSV.

    In batch mode, multiple programs run sequentially in one device session.
    Each program's TRISC_1 kernel zones form a distinct time cluster separated
    by large gaps (program creation + buffer alloc between shapes).
    We detect these clusters by finding the (num_shapes-1) largest time gaps
    among consecutive TRISC_1 start/end events.

    Returns:
        list[float]: Per-shape compute times in microseconds (TRISC_1 avg),
                     one per shape in execution order.
    """
    try:
        with open(csv_path, 'r') as f:
            first_line = f.readline()
            chip_freq = None
            for part in first_line.split(','):
                if 'CHIP_FREQ[MHz]' in part:
                    chip_freq = int(part.split(':')[1].strip())
                    break
            if chip_freq is None:
                return [0.0] * num_shapes

            reader = csv.DictReader(f)

            # Collect all TRISC_1 KERNEL (start, end) pairs
            pending = {}  # (core_x, core_y) -> start_time
            executions = []  # list of (start, end)

            for row in reader:
                row = {k.strip(): v.strip() for k, v in row.items()}
                if 'KERNEL' not in row.get('zone name', ''):
                    continue
                if row.get('RISC processor type', '') != 'TRISC_1':
                    continue

                core_key = (int(row['core_x']), int(row['core_y']))
                cycles = int(row['time[cycles since reset]'])

                if row['type'] == 'ZONE_START':
                    pending[core_key] = cycles
                elif row['type'] == 'ZONE_END' and core_key in pending:
                    executions.append((pending[core_key], cycles))
                    del pending[core_key]

        if not executions or num_shapes <= 0:
            return [0.0] * max(num_shapes, 1)

        executions.sort(key=lambda e: e[0])

        if num_shapes == 1:
            durations = [(end - start) / chip_freq for start, end in executions]
            return [sum(durations) / len(durations)] if durations else [0.0]

        # Find the (num_shapes-1) largest gaps between consecutive start times
        starts = [e[0] for e in executions]
        gaps = [(starts[i+1] - starts[i], i+1) for i in range(len(starts)-1)]
        gaps.sort(reverse=True)
        split_indices = sorted([g[1] for g in gaps[:num_shapes-1]])

        # Split executions into clusters
        clusters = []
        prev = 0
        for idx in split_indices:
            clusters.append(executions[prev:idx])
            prev = idx
        clusters.append(executions[prev:])

        # Average TRISC_1 duration per cluster
        times = []
        for cluster in clusters:
            if cluster:
                durations = [(end - start) / chip_freq for start, end in cluster]
                times.append(sum(durations) / len(durations))
            else:
                times.append(0.0)

        return times

    except Exception:
        return [0.0] * num_shapes


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1

    csv_path = Path(sys.argv[1])

    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_path}")
        return 1

    try:
        profiler_data = parse_profiler_csv(csv_path)
        summary = summarize_timing(profiler_data)

        # Print detailed per-core breakdown
        print_detailed_report(profiler_data)

        # Print summary
        print_summary_report(summary)

        # Print machine-readable output for scripting
        print(f"\n# Machine-readable output:")
        print(f"COMPUTE_TIME_US={summary['math_us']:.2f}")
        print(f"TOTAL_TIME_US={summary['total_us']:.2f}")

        return 0

    except Exception as e:
        print(f"Error parsing CSV: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
