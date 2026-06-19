#pragma once

#include <vector>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <tt-metalium/bfloat16.hpp>

namespace tt {
namespace tt_metal {

/**
 * @brief Generate exhaustive BF16 test inputs by iterating all possible BF16 bit patterns
 *
 * This function generates all 2^16 = 65,536 possible BF16 values, filters out subnormals
 * (exponent==0 && mantissa!=0), and returns only values within the specified test range.
 *
 * This ensures we test worst-case bit patterns and can detect catastrophic ULP errors
 * that might not appear with linearly-spaced inputs.
 *
 * @param test_min Minimum value of test range (inclusive)
 * @param test_max Maximum value of test range (inclusive)
 * @return Vector of unique BF16 values in the specified range, excluding subnormals
 */
inline std::vector<bfloat16> generate_exhaustive_bf16_inputs(float test_min, float test_max) {
    std::vector<bfloat16> bf16_values;
    bf16_values.reserve(65536);

    // Generate all 2^16 = 65,536 possible BF16 bit patterns
    for (uint32_t bits = 0; bits < 65536; bits++) {
        // Check if subnormal: exponent == 0 AND mantissa != 0
        uint8_t exponent = (bits >> 7) & 0xFF;
        uint8_t mantissa = bits & 0x7F;
        bool is_subnormal = (exponent == 0x00) && (mantissa != 0x00);

        if (!is_subnormal) {
            // Convert BF16 bits to float for range check
            uint32_t fp32_bits = static_cast<uint32_t>(bits) << 16;
            float val;
            std::memcpy(&val, &fp32_bits, sizeof(float));

            // Filter to test range and check for valid values
            if (std::isfinite(val) && val >= test_min && val <= test_max) {
                // Direct bit construction via memcpy - avoids float->bf16 rounding overhead
                uint16_t bits_u16 = static_cast<uint16_t>(bits);
                bfloat16 bf16_val;
                std::memcpy(&bf16_val, &bits_u16, sizeof(uint16_t));
                bf16_values.push_back(bf16_val);
            }
        }
    }

    return bf16_values;
}

/**
 * @brief Fill a buffer with exhaustive BF16 inputs, repeating/tiling as needed
 *
 * OPTIMIZED: Uses memcpy tiling instead of per-element modulo (50x+ faster for large buffers)
 *
 * @param output_ptr Pointer to output buffer (must be pre-allocated)
 * @param num_elements Number of elements to generate
 * @param test_min Minimum value of test range
 * @param test_max Maximum value of test range
 * @return Number of unique BF16 values found in the range
 */
inline size_t fill_buffer_with_exhaustive_bf16(
    bfloat16* output_ptr,
    size_t num_elements,
    float test_min,
    float test_max
) {
    auto bf16_values = generate_exhaustive_bf16_inputs(test_min, test_max);
    const size_t pattern_size = bf16_values.size();

    if (pattern_size == 0) {
        return 0;
    }

    // Fast tiling using memcpy - much faster than per-element modulo
    // For yolov4 (52M elements), this is ~50x faster than the naive loop
    const size_t pattern_bytes = pattern_size * sizeof(bfloat16);
    size_t offset = 0;

    // Copy full patterns
    while (offset + pattern_size <= num_elements) {
        std::memcpy(output_ptr + offset, bf16_values.data(), pattern_bytes);
        offset += pattern_size;
    }

    // Copy remaining partial pattern
    if (offset < num_elements) {
        size_t remaining = num_elements - offset;
        std::memcpy(output_ptr + offset, bf16_values.data(), remaining * sizeof(bfloat16));
    }

    return pattern_size;
}

} // namespace tt_metal
} // namespace tt
