// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <iostream>
#include <cctype>
#include <algorithm>
#include <cmath>
#include <limits>

namespace {
/**
 * Clamp values outside float32 representable range.
 *
 * Float32 limits:
 * - Maximum: 3.4028235e+38
 * - Minimum denormalized: 1.401298e-45
 *
 * Values outside these bounds will be clamped to prevent inf values in LUTs.
 * Note: Clipping large coefficients may produce incorrect results, but allows
 * testing of otherwise-invalid approximations.
 */
inline float clamp_float32(double value) {
    constexpr double MIN_FLOAT32_DENORM = 1.4e-45;
    constexpr double MAX_FLOAT32 = 3.4028234663852886e38;

    // Handle overflow (clip to max float32)
    if (std::abs(value) > MAX_FLOAT32) {
        return value > 0 ? MAX_FLOAT32 : -MAX_FLOAT32;
    }

    // Handle underflow (clip to zero)
    if (std::abs(value) < MIN_FLOAT32_DENORM) {
        return 0.0f;
    }

    return static_cast<float>(value);
}
}  // anonymous namespace

/**
 * RationalLUTLoader - Utility class for loading rational approximation LUTs from CSV files
 *
 * CSV format (from polynomial fitter with rational approximation):
 *   segment_id,lo,hi,approximation_type,num_degree,den_degree,n0,n1,n2,...,d0,d1,d2,...,error,method,segmentation
 *
 * Output format:
 *   - Boundaries: [b0, b1, b2, ..., bn] where b0=first segment's lo, bi=segment[i-1]'s hi
 *   - Coefficients: [n0, n1, ..., nm, d0, d1, ..., dk] for each segment (interleaved)
 *
 * Rational approximation: y = (n0 + n1*x + n2*x² + ... + nm*x^m) / (d0 + d1*x + d2*x² + ... + dk*x^k)
 */
class RationalLUTLoader {
public:
    /**
     * Rational LUT information structure
     */
    struct RationalLUTInfo {
        std::vector<float> lut_data;    // Flat array: [boundaries...][num_coeffs...][den_coeffs...]
        uint32_t num_segments;          // Number of piecewise segments
        uint32_t num_degree;            // Numerator polynomial degree
        uint32_t den_degree;            // Denominator polynomial degree
        uint32_t lut_size;              // Total LUT size = (num_segments + 1) + num_segments * (num_degree + den_degree + 2)
    };

    /**
     * Load rational LUT from coefficient CSV file
     *
     * @param filename Path to rational coefficient CSV file
     * @return RationalLUTInfo structure with parsed data
     * @throws std::runtime_error if file cannot be opened or parsed
     */
    static RationalLUTInfo load(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open rational coefficient CSV file: " + filename);
        }

        // Read header line
        std::string header_line;
        if (!std::getline(file, header_line)) {
            throw std::runtime_error("Failed to read CSV header from: " + filename);
        }
        // Strip trailing \r if present (handle Windows line endings)
        if (!header_line.empty() && header_line.back() == '\r') {
            header_line.pop_back();
        }

        // Parse header to find coefficient columns (n0, n1, ..., d0, d1, ...)
        std::vector<std::string> headers;
        std::stringstream header_ss(header_line);
        std::string header;
        while (std::getline(header_ss, header, ',')) {
            headers.push_back(header);
        }

        // Find indices of required columns
        int lo_idx = -1, hi_idx = -1, num_degree_idx = -1, den_degree_idx = -1;
        std::vector<int> num_coeff_indices;  // n0, n1, n2, ...
        std::vector<int> den_coeff_indices;  // d0, d1, d2, ...

        for (size_t i = 0; i < headers.size(); i++) {
            if (headers[i] == "lo") {
                lo_idx = i;
            } else if (headers[i] == "hi") {
                hi_idx = i;
            } else if (headers[i] == "num_degree") {
                num_degree_idx = i;
            } else if (headers[i] == "den_degree") {
                den_degree_idx = i;
            } else if (headers[i].size() >= 2 && headers[i][0] == 'n' && std::isdigit(headers[i][1])) {
                // Numerator coefficient column (n0, n1, n2, ...)
                int coeff_num = std::stoi(headers[i].substr(1));
                while (num_coeff_indices.size() <= static_cast<size_t>(coeff_num)) {
                    num_coeff_indices.push_back(-1);
                }
                num_coeff_indices[coeff_num] = i;
            } else if (headers[i].size() >= 2 && headers[i][0] == 'd' && std::isdigit(headers[i][1])) {
                // Denominator coefficient column (d0, d1, d2, ...)
                int coeff_num = std::stoi(headers[i].substr(1));
                while (den_coeff_indices.size() <= static_cast<size_t>(coeff_num)) {
                    den_coeff_indices.push_back(-1);
                }
                den_coeff_indices[coeff_num] = i;
            }
        }

        if (lo_idx == -1 || hi_idx == -1) {
            throw std::runtime_error("CSV missing required 'lo' or 'hi' columns: " + filename);
        }
        if (num_coeff_indices.empty() || den_coeff_indices.empty()) {
            throw std::runtime_error("CSV has no numerator (n0, n1, ...) or denominator (d0, d1, ...) coefficient columns: " + filename);
        }

        // Read all segments
        struct Segment {
            float lo;
            float hi;
            uint32_t num_degree;
            uint32_t den_degree;
            std::vector<float> num_coeffs;
            std::vector<float> den_coeffs;
        };
        std::vector<Segment> segments;

        // Track max degrees across all segments (for validation)
        uint32_t max_num_degree = 0;
        uint32_t max_den_degree = 0;

        std::string line;
        while (std::getline(file, line)) {
            // Strip trailing \r if present (handle Windows line endings)
            if (!line.empty() && line.back() == '\r') {
                line.pop_back();
            }
            if (line.empty()) continue;

            // Skip METADATA rows from polynomial fitter
            if (line.find("METADATA") == 0) {
                continue;
            }

            std::vector<std::string> values;
            std::stringstream line_ss(line);
            std::string value;
            while (std::getline(line_ss, value, ',')) {
                values.push_back(value);
            }

            if (values.size() < headers.size()) {
                throw std::runtime_error("CSV row has fewer columns than header: " + filename);
            }

            Segment seg;
            try {
                seg.lo = static_cast<float>(std::stod(values[lo_idx]));
            } catch (const std::exception& e) {
                throw std::runtime_error("Failed to parse 'lo' value: '" + values[lo_idx] + "' in file: " + filename);
            }
            try {
                seg.hi = static_cast<float>(std::stod(values[hi_idx]));
            } catch (const std::exception& e) {
                throw std::runtime_error("Failed to parse 'hi' value: '" + values[hi_idx] + "' in file: " + filename);
            }

            // Parse degrees (if present in CSV)
            if (num_degree_idx != -1) {
                try {
                    seg.num_degree = static_cast<uint32_t>(std::stoi(values[num_degree_idx]));
                } catch (const std::exception& e) {
                    throw std::runtime_error("Failed to parse 'num_degree' value: '" + values[num_degree_idx] + "' in file: " + filename);
                }
            } else {
                // Infer from number of non-empty coefficient columns
                seg.num_degree = static_cast<uint32_t>(num_coeff_indices.size()) - 1;
            }

            if (den_degree_idx != -1) {
                try {
                    seg.den_degree = static_cast<uint32_t>(std::stoi(values[den_degree_idx]));
                } catch (const std::exception& e) {
                    throw std::runtime_error("Failed to parse 'den_degree' value: '" + values[den_degree_idx] + "' in file: " + filename);
                }
            } else {
                // Infer from number of non-empty coefficient columns
                seg.den_degree = static_cast<uint32_t>(den_coeff_indices.size()) - 1;
            }

            // Parse numerator coefficients
            for (size_t i = 0; i < num_coeff_indices.size(); i++) {
                int coeff_idx = num_coeff_indices[i];
                if (coeff_idx != -1 && static_cast<size_t>(coeff_idx) < values.size()) {
                    try {
                        double coeff_value = std::stod(values[coeff_idx]);
                        seg.num_coeffs.push_back(clamp_float32(coeff_value));
                    } catch (const std::exception& e) {
                        throw std::runtime_error("Failed to parse numerator coefficient n" + std::to_string(i) +
                                               " value: '" + values[coeff_idx] + "' in file: " + filename);
                    }
                }
            }

            // Parse denominator coefficients
            for (size_t i = 0; i < den_coeff_indices.size(); i++) {
                int coeff_idx = den_coeff_indices[i];
                if (coeff_idx != -1 && static_cast<size_t>(coeff_idx) < values.size()) {
                    try {
                        double coeff_value = std::stod(values[coeff_idx]);
                        seg.den_coeffs.push_back(clamp_float32(coeff_value));
                    } catch (const std::exception& e) {
                        throw std::runtime_error("Failed to parse denominator coefficient d" + std::to_string(i) +
                                               " value: '" + values[coeff_idx] + "' in file: " + filename);
                    }
                }
            }

            max_num_degree = std::max(max_num_degree, seg.num_degree);
            max_den_degree = std::max(max_den_degree, seg.den_degree);

            segments.push_back(seg);
        }

        file.close();

        if (segments.empty()) {
            throw std::runtime_error("No segments found in CSV: " + filename);
        }

        // Validate all segments have same degrees
        for (const auto& seg : segments) {
            if (seg.num_degree != max_num_degree || seg.den_degree != max_den_degree) {
                throw std::runtime_error("Inconsistent degrees across segments in: " + filename);
            }
        }

        // Build LUT: [boundaries...] [num_coeffs_seg0, den_coeffs_seg0, num_coeffs_seg1, den_coeffs_seg1, ...]
        std::vector<float> lut_data;

        // Add boundaries: [b0, b1, b2, ..., bn]
        // b0 = first segment's lo (lower clamp), bn = last segment's hi (upper clamp)
        // bi (i=1..n-1) = boundaries between segments
        lut_data.push_back(segments[0].lo);  // b0: lower bound
        for (const auto& seg : segments) {
            lut_data.push_back(seg.hi);  // b1..bn: segment transitions + upper bound
        }

        // Add coefficients interleaved per segment: [num_coeffs, den_coeffs] for each segment
        for (const auto& seg : segments) {
            // Add numerator coefficients
            for (float coeff : seg.num_coeffs) {
                lut_data.push_back(coeff);
            }
            // Add denominator coefficients
            for (float coeff : seg.den_coeffs) {
                lut_data.push_back(coeff);
            }
        }

        // Calculate expected LUT size
        uint32_t num_segments = static_cast<uint32_t>(segments.size());
        uint32_t lut_size = (num_segments + 1) + num_segments * (max_num_degree + max_den_degree + 2);

        if (lut_data.size() != lut_size) {
            throw std::runtime_error("LUT size mismatch: expected " + std::to_string(lut_size) +
                                   ", got " + std::to_string(lut_data.size()) + " in file: " + filename);
        }

        RationalLUTInfo info;
        info.lut_data = lut_data;
        info.num_segments = num_segments;
        info.num_degree = max_num_degree;
        info.den_degree = max_den_degree;
        info.lut_size = lut_size;

        return info;
    }

    /**
     * Print summary statistics about rational LUT
     */
    static void print_stats(const RationalLUTInfo& info) {
        if (info.lut_data.empty()) {
            std::cout << "Rational LUT is empty" << std::endl;
            return;
        }

        float min_val = info.lut_data[0];
        float max_val = info.lut_data[0];
        double sum = 0.0;

        for (float val : info.lut_data) {
            if (val < min_val) min_val = val;
            if (val > max_val) max_val = val;
            sum += val;
        }

        double mean = sum / info.lut_data.size();

        std::cout << "Rational LUT Statistics:" << std::endl;
        std::cout << "  Segments: " << info.num_segments << std::endl;
        std::cout << "  Numerator degree: " << info.num_degree << std::endl;
        std::cout << "  Denominator degree: " << info.den_degree << std::endl;
        std::cout << "  LUT Size: " << info.lut_size << std::endl;
        std::cout << "  Min coefficient: " << min_val << std::endl;
        std::cout << "  Max coefficient: " << max_val << std::endl;
        std::cout << "  Mean coefficient: " << mean << std::endl;
    }

    /**
     * Extract range reduction method from CSV METADATA rows
     *
     * @param filename Path to coefficient CSV file
     * @return Range reduction method string (e.g., "exp", "trig"), or "" if none
     */
    static std::string extract_range_reduction_method(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            return "";
        }

        std::string line;
        std::getline(file, line);  // skip header

        while (std::getline(file, line)) {
            if (!line.empty() && line.back() == '\r') {
                line.pop_back();
            }
            if (line.find("METADATA") != 0) continue;

            std::vector<std::string> values;
            std::stringstream ss(line);
            std::string token;
            while (std::getline(ss, token, ',')) {
                values.push_back(token);
            }

            if (values.size() >= 3 && values[1] == "range_reduction_method") {
                return values[2];
            }
        }
        return "";
    }

};
