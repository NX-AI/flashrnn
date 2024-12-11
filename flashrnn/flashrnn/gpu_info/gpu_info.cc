// Copyright 2024 NXAI GmbH, All Rights Reserved
// Author: Korbinian Poeppel
// Adapted from the haste library
//
// See:
// Copyright 2020 LMNT, Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================

#include <ATen/cuda/CUDAContext.h>

#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <driver_types.h>
#include <iostream>
#include <memory>
#include <pybind11/embed.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include <tuple>
#include <vector>

#include "../util/support.h"
#include "gpu_info.h"

namespace {

class GPUInfo {
public:
  GPUInfo() {}

  pybind11::dict gpu_info(int device) {
    pybind11::dict dict;

    cudaDeviceProp prop;
    xlstm::get_gpu_info(prop, device);

    dict["name"] = prop.name; //*< ASCII string identifying device */
    dict["luid"] =
        prop.luid; //*< 8-byte locally unique identifier. Value is undefined on
                   // TCC                 and non-Windows platforms */
    dict["luidDeviceNodeMask"] =
        prop.luidDeviceNodeMask; //*< LUID device node mask. Value is undefined
                                 // on TCC and non-Windows platforms */
    dict["totalGlobalMem"] =
        prop.totalGlobalMem; //*< Global memory available on device in bytes */
    dict["sharedMemPerBlock"] =
        prop.sharedMemPerBlock; //*< Shared memory available per block in bytes
                                //*/
    dict["regsPerBlock"] =
        prop.regsPerBlock; //*< 32-bit registers available per block */
    dict["warpSize"] = prop.warpSize; //*< Warp size in threads */
    dict["memPitch"] =
        prop.memPitch; //*< Maximum pitch in bytes allowed by memory copies */
    dict["maxThreadsPerBlock"] =
        prop.maxThreadsPerBlock; //*< Maximum number of threads per block */
    dict["maxThreadsDim"] =
        prop.maxThreadsDim; //*< Maximum size of each dimension of a block */
    dict["maxGridSize"] =
        prop.maxGridSize; //*< Maximum size of each dimension of a grid */
    dict["clockRate"] = prop.clockRate; //*< Clock frequency in kilohertz */
    dict["totalConstMem"] =
        prop.totalConstMem; //*< Constant memory available on device in bytes */
    dict["major"] = prop.major; //*< Major compute capability */
    dict["minor"] = prop.minor; //*< Minor compute capability */
    dict["textureAlignment"] =
        prop.textureAlignment; //*< Alignment requirement for textures */
    dict["texturePitchAlignment"] =
        prop.texturePitchAlignment; //*< Pitch alignment requirement for texture
                                    // references bound to pitched memory */
    dict["deviceOverlap"] =
        prop.deviceOverlap; //*< Device can concurrently copy memory and execute
                            // a                            kernel. Deprecated.
                            // Use instead asyncEngineCount. */
    dict["multiProcessorCount"] =
        prop.multiProcessorCount; //*< Number of multiprocessors on device */
    dict["kernelExecTimeoutEnabled"] =
        prop.kernelExecTimeoutEnabled; //*< Specified whether there is a run
                                       // time limit on kernels */
    dict["integrated"] =
        prop.integrated; //*< Device is integrated as opposed to discrete */
    dict["canMapHostMemory"] =
        prop.canMapHostMemory; //*< Device can map host memory with
                               // cudaHostAlloc/cudaHostGetDevicePointer */
    dict["computeMode"] =
        prop.computeMode; //*< Compute mode (See ::cudaComputeMode) */
    dict["maxTexture1D"] = prop.maxTexture1D; //*< Maximum 1D texture size */
    dict["maxTexture1DMipmap"] =
        prop.maxTexture1DMipmap; //*< Maximum 1D mipmapped texture size */
    dict["maxTexture1DLinear"] =
        prop.maxTexture1DLinear; //*< Deprecated, do not use. Use
                                 // cudaDeviceGetTexture1DLinearMaxWidth() or
                                 // cuDeviceGetTexture1DLinearMaxWidth()
                                 // instead.
                                 //*/
    dict["maxTexture2D"] =
        prop.maxTexture2D; //*< Maximum 2D texture dimensions */
    dict["maxTexture2DMipmap"] =
        prop.maxTexture2DMipmap; //*< Maximum 2D mipmapped texture dimensions */
    dict["maxTexture2DLinear"] =
        prop.maxTexture2DLinear; //*< Maximum dimensions (width, height, pitch)
                                 // for 2D                              textures
                                 // bound to pitched memory */
    dict["maxTexture2DGather"] =
        prop.maxTexture2DGather; //*< Maximum 2D texture dimensions if texture
                                 // gather operations have to be performed */
    dict["maxTexture3D"] =
        prop.maxTexture3D; //*< Maximum 3D texture dimensions */
    dict["maxTexture3DAlt"] =
        prop.maxTexture3DAlt; //*< Maximum alternate 3D texture dimensions */
    dict["maxTextureCubemap"] =
        prop.maxTextureCubemap; //*< Maximum Cubemap texture dimensions */
    dict["maxTexture1DLayered"] =
        prop.maxTexture1DLayered; //*< Maximum 1D layered texture dimensions */
    dict["maxTexture2DLayered"] =
        prop.maxTexture2DLayered; //*< Maximum 2D layered texture dimensions */
    dict["maxTextureCubemapLayered"] =
        prop.maxTextureCubemapLayered; //*< Maximum Cubemap layered texture
                                       // dimensions */
    dict["maxSurface1D"] = prop.maxSurface1D; //*< Maximum 1D surface size */
    dict["maxSurface2D"] =
        prop.maxSurface2D; //*< Maximum 2D surface dimensions */
    dict["maxSurface3D"] =
        prop.maxSurface3D; //*< Maximum 3D surface dimensions */
    dict["maxSurface1DLayered"] =
        prop.maxSurface1DLayered; //*< Maximum 1D layered surface dimensions */
    dict["maxSurface2DLayered"] =
        prop.maxSurface2DLayered; //*< Maximum 2D layered surface dimensions */
    dict["maxSurfaceCubemap"] =
        prop.maxSurfaceCubemap; //*< Maximum Cubemap surface dimensions */
    dict["maxSurfaceCubemapLayered"] =
        prop.maxSurfaceCubemapLayered; //*< Maximum Cubemap layered surface
                                       // dimensions */
    dict["surfaceAlignment"] =
        prop.surfaceAlignment; //*< Alignment requirements for surfaces */
    dict["concurrentKernels"] =
        prop.concurrentKernels; //*< Device can possibly execute multiple
                                // kernels                          concurrently
                                //*/
    dict["ECCEnabled"] = prop.ECCEnabled; //*< Device has ECC support enabled */
    dict["pciBusID"] = prop.pciBusID;     //*< PCI bus ID of the device */
    dict["pciDeviceID"] = prop.pciDeviceID; //*< PCI device ID of the device */
    dict["pciDomainID"] = prop.pciDomainID; //*< PCI domain ID of the device */
    dict["tccDriver"] = prop.tccDriver; //*< 1 if device is a Tesla device using
                                        // TCC driver, 0 otherwise */
    dict["asyncEngineCount"] =
        prop.asyncEngineCount; //*< Number of asynchronous engines */
    dict["unifiedAddressing"] =
        prop.unifiedAddressing; //*< Device shares a unified address space with
                                // the host                        */
    dict["memoryClockRate"] =
        prop.memoryClockRate; //*< Peak memory clock frequency in kilohertz */
    dict["memoryBusWidth"] =
        prop.memoryBusWidth; //*< Global memory bus width in bits */
    dict["l2CacheSize"] = prop.l2CacheSize; //*< Size of L2 cache in bytes */
    dict["persistingL2CacheMaxSize"] =
        prop.persistingL2CacheMaxSize; //*< Device's maximum l2 persisting lines
                                       // capacity setting in bytes */
    dict["maxThreadsPerMultiProcessor"] =
        prop.maxThreadsPerMultiProcessor; //*< Maximum resident threads per
                                          // multiprocessor */
    dict["streamPrioritiesSupported"] =
        prop.streamPrioritiesSupported; //*< Device supports stream priorities
                                        //*/
    dict["globalL1CacheSupported"] =
        prop.globalL1CacheSupported; //*< Device supports caching globals in L1
                                     //*/
    dict["localL1CacheSupported"] =
        prop.localL1CacheSupported; //*< Device supports caching locals in L1 */
    dict["sharedMemPerMultiprocessor"] =
        prop.sharedMemPerMultiprocessor; //*< Shared memory available per
                                         // multiprocessor in bytes */
    dict["regsPerMultiprocessor"] =
        prop.regsPerMultiprocessor; //*< 32-bit registers available per
                                    // multiprocessor */
    dict["managedMemory"] =
        prop.managedMemory; //*< Device supports allocating managed memory on
                            // this system                    */
    dict["isMultiGpuBoard"] =
        prop.isMultiGpuBoard; //*< Device is on a multi-GPU board */
    dict["multiGpuBoardGroupID"] =
        prop.multiGpuBoardGroupID; //*< Unique identifier for a group of devices
                                   // on the                             same
                                   // multi-GPU board */
    dict["hostNativeAtomicSupported"] =
        prop.hostNativeAtomicSupported; //*< Link between the device and the
                                        // host supports native atomic
                                        // operations
                                        //*/
    dict["singleToDoublePrecisionPerfRatio"] =
        prop.singleToDoublePrecisionPerfRatio; //*< Ratio of single precision
                                               // performance (in floating-point
                                               // operations per second) to
                                               // double precision performance
                                               // */
    dict["pageableMemoryAccess"] =
        prop.pageableMemoryAccess; //*< Device supports coherently accessing
                                   // pageable memory without calling
                                   // cudaHostRegister on it */
    dict["concurrentManagedAccess"] =
        prop.concurrentManagedAccess; //*< Device can coherently access managed
                                      // memory concurrently with the CPU */
    dict["computePreemptionSupported"] =
        prop.computePreemptionSupported; //*< Device supports Compute Preemption
                                         //*/
    dict["canUseHostPointerForRegisteredMem"] =
        prop.canUseHostPointerForRegisteredMem; //*< Device can access host
                                                // registered memory at the same
                                                // virtual address as the CPU */
    dict["cooperativeLaunch"] =
        prop.cooperativeLaunch; //*< Device supports launching cooperative
                                // kernels via ::cudaLaunchCooperativeKernel */
    dict["cooperativeMultiDeviceLaunch"] =
        prop.cooperativeMultiDeviceLaunch; //*< Device can participate in
                                           // cooperative kernels launched via
                                           //::cudaLaunchCooperativeKernelMultiDevice
                                           //*/
    dict["sharedMemPerBlockOptin"] =
        prop.sharedMemPerBlockOptin; //*< Per device maximum shared memory per
                                     // block usable by special opt in */
    dict["pageableMemoryAccessUsesHostPageTables"] =
        prop.pageableMemoryAccessUsesHostPageTables; //*< Device accesses
                                                     // pageable memory via the
                                                     // host's page tables */
    dict["directManagedMemAccessFromHost"] =
        prop.directManagedMemAccessFromHost; //*< Host can directly access
                                             // managed memory on the device
                                             // without migration. */
    dict["maxBlocksPerMultiProcessor"] =
        prop.maxBlocksPerMultiProcessor; //*< Maximum number of resident blocks
                                         // per multiprocessor */
    dict["accessPolicyMaxWindowSize"] =
        prop.accessPolicyMaxWindowSize; //*< The maximum value of
                                        //::cudaAccessPolicyWindow::num_bytes.
                                        //*/
    dict["reservedSharedMemPerBlock"] =
        prop.reservedSharedMemPerBlock; //*< Shared memory reserved by CUDA
                                        // driver per block in bytes */

    return dict;
  }
};

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  pybind11::class_<GPUInfo>(m, "GPUInfo")
      .def(pybind11::init<>())
      .def("gpu_info", &GPUInfo::gpu_info);
}
