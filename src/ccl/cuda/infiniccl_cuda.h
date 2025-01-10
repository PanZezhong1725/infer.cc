#ifndef INFINICCL_CUDA_H_
#define INFINICCL_CUDA_H_
#include "infiniccl.h"

#if defined(ENABLE_NV_GPU) && defined(ENABLE_CCL)
#define IMPL_WITH_CUDA ;
#else
#define IMPL_WITH_CUDA { return INFINICCL_STATUS_DEVICE_NOT_SUPPORTED; }
#endif

infinicclStatus_t
infinicclCudaCommInitAll(infinicclComm_t *comms, unsigned int numDevices, unsigned int const *deviceIDs) IMPL_WITH_CUDA infinicclStatus_t
infinicclCudaCommDestroy(infinicclComm_t comm) IMPL_WITH_CUDA 
infinicclStatus_t infinicclCudaAllReduceSum(infinicclComm_t comm, void *sendbuf,
                              void *recvbuf, size_t count,
                              InfiniDataType_t datatype,
                              infinirtStream_t stream) IMPL_WITH_CUDA

#endif /* INFINICCL_CUDA_H_ */
