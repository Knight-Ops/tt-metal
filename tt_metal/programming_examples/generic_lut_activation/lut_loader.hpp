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
 * LUTLoader - Utility class for loading lookup tables from coefficient CSV files
 *
 * CSV format (from polynomial fitter):
 *   segment_id,lo,hi,c0,c1,c2,...,error,method,segmentation
 *
 * Output format:
 *   - Boundaries: [b0, b1, b2, ..., bn] where b0=first segment's lo, bi=segment[i-1]'s hi
 *   - Coefficients: [seg0_c0, seg0_c1, ..., seg0_cN, seg1_c0, seg1_c1, ..., seg1_cN, ...]
 */
class LUTLoader {
public:
    /**
     * Load LUT from coefficient CSV file
     *
     * @param filename Path to coefficient CSV file
     * @return Vector of float values (boundaries + coefficients)
     * @throws std::runtime_error if file cannot be opened or parsed
     */
    static std::vector<float> load(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open coefficient CSV file: " + filename);
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

        // Parse header to find coefficient columns (c0, c1, c2, ...)
        std::vector<std::string> headers;
        std::stringstream header_ss(header_line);
        std::string header;
        while (std::getline(header_ss, header, ',')) {
            headers.push_back(header);
        }

        // Find indices of required columns
        int lo_idx = -1, hi_idx = -1;
        std::vector<int> coeff_indices;

        for (size_t i = 0; i < headers.size(); i++) {
            if (headers[i] == "lo") lo_idx = i;
            else if (headers[i] == "hi") hi_idx = i;
            else if (headers[i].size() >= 2 && headers[i][0] == 'c' &&
                     std::isdigit(headers[i][1])) {
                // Coefficient column (c0, c1, c2, ...)
                int coeff_num = std::stoi(headers[i].substr(1));
                while (coeff_indices.size() <= coeff_num) {
                    coeff_indices.push_back(-1);
                }
                coeff_indices[coeff_num] = i;
            }
        }

        if (lo_idx == -1 || hi_idx == -1) {
            throw std::runtime_error("CSV missing required 'lo' or 'hi' columns: " + filename);
        }
        if (coeff_indices.empty()) {
            throw std::runtime_error("CSV has no coefficient columns (c0, c1, ...): " + filename);
        }

        // Read all segments
        struct Segment {
            float lo;
            float hi;
            std::vector<float> coeffs;
        };
        std::vector<Segment> segments;

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

            for (size_t i = 0; i < coeff_indices.size(); i++) {
                int coeff_idx = coeff_indices[i];
                if (coeff_idx != -1 && coeff_idx < values.size()) {
                    try {
                        double coeff_value = std::stod(values[coeff_idx]);
                        seg.coeffs.push_back(clamp_float32(coeff_value));
                    } catch (const std::exception& e) {
                        throw std::runtime_error("Failed to parse coefficient c" + std::to_string(i) +
                                               " value: '" + values[coeff_idx] + "' in file: " + filename);
                    }
                }
            }

            segments.push_back(seg);
        }

        file.close();

        if (segments.empty()) {
            throw std::runtime_error("No segments found in CSV: " + filename);
        }

        // Build LUT: [boundaries...] [coefficients...]
        std::vector<float> lut_data;

        // Add boundaries: [b0, b1, b2, ..., bn]
        // b0 = first segment's lo (lower clamp), bn = last segment's hi (upper clamp)
        // bi (i=1..n-1) = boundaries between segments
        lut_data.push_back(segments[0].lo);  // b0: lower bound
        for (const auto& seg : segments) {
            lut_data.push_back(seg.hi);  // b1..bn: segment transitions + upper bound
        }

        // Add coefficients interleaved per segment
        for (const auto& seg : segments) {
            for (float coeff : seg.coeffs) {
                lut_data.push_back(coeff);
            }
        }

        return lut_data;
    }


    /**
     * Split double-precision coefficient into float32 hi/lo pair for double-float
     *
     * Double-float representation: x = x_hi + x_lo where both are float32.
     * This provides ~48-bit precision (~14-15 decimal digits) vs ~24-bit for single float32.
     *
     * Algorithm:
     *   x_hi = float32(coeff)                    (round to float32)
     *   x_lo = float32(coeff - double(x_hi))     (residual error)
     *
     * @param coeff Double-precision coefficient value
     * @return Pair of (hi, lo) float32 values
     */
    struct DDCoeff {
        float hi;
        float lo;
    };

    static DDCoeff split_coefficient(double coeff) {
        DDCoeff result;
        result.hi = static_cast<float>(coeff);  // Round to float32
        // Compute residual in double precision, then cast to float32
        result.lo = static_cast<float>(coeff - static_cast<double>(result.hi));
        return result;
    }

    /**
     * Load LUT with double-float coefficient storage
     *
     * Same as load() but splits each coefficient into (hi, lo) pair and interleaves them.
     *
     * CSV format (from polynomial fitter):
     *   segment_id,lo,hi,c0,c1,c2,...,error,method,segmentation
     *
     * Output format:
     *   - Boundaries: [b0, b1, b2, ..., bn] (unchanged)
     *   - Coefficients: [seg0_c0_hi, seg0_c0_lo, seg0_c1_hi, seg0_c1_lo, ..., seg1_c0_hi, seg1_c0_lo, ...]
     *
     * LUT size: (NUM_SEGMENTS + 1) + NUM_SEGMENTS * 2 * (POLY_DEGREE + 1)
     *           (boundaries)       + (double-float coefficient pairs)
     *
     * @param filename Path to coefficient CSV file
     * @return Vector of float values (boundaries + DD coefficient pairs)
     * @throws std::runtime_error if file cannot be opened or parsed
     */
    static std::vector<float> load_dd(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open coefficient CSV file: " + filename);
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

        // Parse header to find coefficient columns (c0, c1, c2, ...)
        std::vector<std::string> headers;
        std::stringstream header_ss(header_line);
        std::string header;
        while (std::getline(header_ss, header, ',')) {
            headers.push_back(header);
        }

        // Find indices of required columns
        int lo_idx = -1, hi_idx = -1;
        std::vector<int> coeff_indices;

        for (size_t i = 0; i < headers.size(); i++) {
            if (headers[i] == "lo") lo_idx = i;
            else if (headers[i] == "hi") hi_idx = i;
            else if (headers[i].size() >= 2 && headers[i][0] == 'c' &&
                     std::isdigit(headers[i][1])) {
                // Coefficient column (c0, c1, c2, ...)
                int coeff_num = std::stoi(headers[i].substr(1));
                while (coeff_indices.size() <= coeff_num) {
                    coeff_indices.push_back(-1);
                }
                coeff_indices[coeff_num] = i;
            }
        }

        if (lo_idx == -1 || hi_idx == -1) {
            throw std::runtime_error("CSV missing required 'lo' or 'hi' columns: " + filename);
        }
        if (coeff_indices.empty()) {
            throw std::runtime_error("CSV has no coefficient columns (c0, c1, ...): " + filename);
        }

        // Read all segments
        struct Segment {
            float lo;
            float hi;
            std::vector<DDCoeff> coeffs;  // Store as DD pairs
        };
        std::vector<Segment> segments;

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

            for (size_t i = 0; i < coeff_indices.size(); i++) {
                int coeff_idx = coeff_indices[i];
                if (coeff_idx != -1 && coeff_idx < values.size()) {
                    try {
                        double coeff_value = std::stod(values[coeff_idx]);
                        // Split coefficient into DD pair (no clamping for DD)
                        seg.coeffs.push_back(split_coefficient(coeff_value));
                    } catch (const std::exception& e) {
                        throw std::runtime_error("Failed to parse coefficient c" + std::to_string(i) +
                                               " value: '" + values[coeff_idx] + "' in file: " + filename);
                    }
                }
            }

            segments.push_back(seg);
        }

        file.close();

        if (segments.empty()) {
            throw std::runtime_error("No segments found in CSV: " + filename);
        }

        // Build LUT: [boundaries...] [DD coefficient pairs...]
        std::vector<float> lut_data;

        // Add boundaries: [b0, b1, b2, ..., bn] (unchanged from single precision)
        lut_data.push_back(segments[0].lo);  // b0: lower bound
        for (const auto& seg : segments) {
            lut_data.push_back(seg.hi);  // b1..bn: segment transitions + upper bound
        }

        // Add DD coefficients interleaved per segment: [c0_hi, c0_lo, c1_hi, c1_lo, ...]
        for (const auto& seg : segments) {
            for (const DDCoeff& coeff : seg.coeffs) {
                lut_data.push_back(coeff.hi);
                lut_data.push_back(coeff.lo);
            }
        }

        return lut_data;
    }

    /**
     * Print summary statistics about LUT
     */
    static void print_stats(const std::vector<float>& lut_data) {
        if (lut_data.empty()) {
            std::cout << "LUT is empty" << std::endl;
            return;
        }

        float min_val = lut_data[0];
        float max_val = lut_data[0];
        double sum = 0.0;

        for (float val : lut_data) {
            if (val < min_val) min_val = val;
            if (val > max_val) max_val = val;
            sum += val;
        }

        double mean = sum / lut_data.size();

        std::cout << "LUT Statistics:" << std::endl;
        std::cout << "  Size: " << lut_data.size() << std::endl;
        std::cout << "  Min: " << min_val << std::endl;
        std::cout << "  Max: " << max_val << std::endl;
        std::cout << "  Mean: " << mean << std::endl;
    }

    /**
     * Extract range reduction method from CSV METADATA rows
     *
     * Scans for rows starting with "METADATA" and looks for key "range_reduction_method".
     * Returns the method string (e.g., "exp", "trig") or empty string if not found.
     *
     * @param filename Path to coefficient CSV file
     * @return Range reduction method string, or "" if none
     */
    static std::string extract_range_reduction_method(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            return "";
        }

        // Skip header
        std::string line;
        std::getline(file, line);

        // Scan for METADATA rows
        while (std::getline(file, line)) {
            if (!line.empty() && line.back() == '\r') {
                line.pop_back();
            }
            if (line.find("METADATA") != 0) continue;

            // Parse: METADATA,key,value,...
            std::vector<std::string> values;
            std::stringstream ss(line);
            std::string token;
            while (std::getline(ss, token, ',')) {
                values.push_back(token);
            }

            // values[0]="METADATA", values[1]=key, values[2]=value
            if (values.size() >= 3 && values[1] == "range_reduction_method") {
                return values[2];
            }
        }
        return "";
    }

};
