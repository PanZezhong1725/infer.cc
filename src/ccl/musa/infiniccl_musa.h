#ifndef INFINICCL_MUSA_H_
#define INFINICCL_MUSA_H_
#include "infiniccl.h"

#if defined(ENABLE_MTHREADS_GPU) && defined(ENABLE_CCL)
#define IMPL_WITH_MUSA ;
#else
#define IMPL_WITH_MUSA { return INFINICCL_STATUS_DEVICE_NOT_SUPPORTED; }
#endif

infinicclStatus_t
infinicclMusaCommInitAll(infinicclComm_t *comms, unsigned int numDevices, unsigned int const *deviceIDs) IMPL_WITH_MUSA infinicclStatus_t
infinicclMusaCommDestroy(infinicclComm_t comm) IMPL_WITH_MUSA 
infinicclStatus_t infinicclMusaAllReduceSum(infinicclComm_t comm, void *sendbuf,
                              void *recvbuf, size_t count,
                              InfiniDataType_t datatype,
                              infinirtStream_t stream) IMPL_WITH_MUSA

#endif /* INFINICCL_CUDA_H_ */
