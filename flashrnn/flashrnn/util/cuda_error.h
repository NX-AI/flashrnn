#include <cuda.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <stdio.h>

void cudaOccupancyMaxActiveBlocksPerMultiprocessor2(dim3 blockSize,
                                                    size_t dynamicSMemSize,
                                                    const void *func);
