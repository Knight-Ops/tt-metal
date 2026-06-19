# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.22)
project(programming_examples_generic_lut_activation)

if(NOT TARGET TT::Metalium)
    find_package(TT-Metalium REQUIRED)
endif()

# =============================================================================
# ADHOC TARGETS - Dynamic kernel generation at sweep time
# =============================================================================
# Instead of 200+ pre-compiled binaries, use adhoc targets that are
# rebuilt on-demand with the specific POLY_DEGREE/NUM_SEGMENTS needed.
#
# Workflow:
#   1. Sweep script sets compile defs in adhoc kernel header
#   2. ninja rebuilds just the adhoc target (fast incremental)
#   3. Binary loads LUT from CSV at runtime
#
# This eliminates hours of upfront compilation.

# Primary polynomial adhoc target
add_executable(programming_examples_generic_lut_activation_adhoc)
target_sources(programming_examples_generic_lut_activation_adhoc PRIVATE generic_lut_activation.cpp)
target_compile_definitions(programming_examples_generic_lut_activation_adhoc PRIVATE
    KERNEL_VARIANT="piecewise_generic"
    ADHOC_MODE=1)
target_link_libraries(programming_examples_generic_lut_activation_adhoc PUBLIC TT::Metalium)
target_include_directories(programming_examples_generic_lut_activation_adhoc PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Rational adhoc target
add_executable(programming_examples_generic_lut_activation_rational_adhoc)
target_sources(programming_examples_generic_lut_activation_rational_adhoc PRIVATE generic_lut_activation_rational.cpp)
target_compile_definitions(programming_examples_generic_lut_activation_rational_adhoc PRIVATE
    ADHOC_MODE=1)
target_link_libraries(programming_examples_generic_lut_activation_rational_adhoc PUBLIC TT::Metalium)
target_include_directories(programming_examples_generic_lut_activation_rational_adhoc PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# =============================================================================
# Native SFPU (hardware operations, no LUT)
# =============================================================================
add_executable(programming_examples_generic_lut_activation_native_sfpu)
target_sources(programming_examples_generic_lut_activation_native_sfpu PRIVATE generic_lut_activation_native.cpp)
target_compile_definitions(programming_examples_generic_lut_activation_native_sfpu PRIVATE
    KERNEL_VARIANT="native_sfpu")
target_link_libraries(programming_examples_generic_lut_activation_native_sfpu PUBLIC TT::Metalium)
target_include_directories(programming_examples_generic_lut_activation_native_sfpu PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# =============================================================================
# Default target (backward compatibility)
# =============================================================================
add_executable(programming_examples_generic_lut_activation)
target_sources(programming_examples_generic_lut_activation PRIVATE generic_lut_activation.cpp)
target_link_libraries(programming_examples_generic_lut_activation PUBLIC TT::Metalium)
target_include_directories(programming_examples_generic_lut_activation PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
