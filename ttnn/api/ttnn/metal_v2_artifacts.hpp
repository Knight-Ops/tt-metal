// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>

namespace ttnn::device_operation {

// Build product of a Metal 2.0 op-porting factory: the immutable ProgramSpec and the mutable
// ProgramRunArgs. Returned by a MetalV2FactoryConcept factory's create_program_artifacts method; the
// framework adapter stamps a Program out of this artifact onto each mesh coordinate range of the
// workload.
//
// Op-owned tensors are deliberately NOT here: keeping them out of ProgramArtifacts (and the
// ProgramSpec) is what keeps them out of the program-cache key. Ops that genuinely need op-owned
// scratch opt into MetalV2OwnedTensorsFactoryConcept (operation_concepts.hpp) and supply them via
// get_owned_tensors; ttnn parks them and hands them to create_program_artifacts.
//
// A future MeshWorkloadSpecFactoryConcept will return a different (multi-program) artifact type for
// ops whose programs vary across the mesh.
struct ProgramArtifacts {
    tt::tt_metal::experimental::ProgramSpec spec;
    tt::tt_metal::experimental::ProgramRunArgs run_params;
};

}  // namespace ttnn::device_operation
