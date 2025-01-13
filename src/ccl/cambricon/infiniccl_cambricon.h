#ifndef INFINICCL_CAMBRICON_H_
#define INFINICCL_CAMBRICON_H_
#include "infiniccl.h"

#if defined(ENABLE_CAMBRICON_MLU) && defined(ENABLE_CCL)
#define IMPL_WITH_CAMBRICON ;
#else
#define IMPL_WITH_CAMBRICON { return INFINICCL_STATUS_DEVICE_NOT_SUPPORTED; }
#endif

infinicclStatus_t
infinicclCambriconCommInitAll(infinicclComm_t *comms, unsigned int numDevices, unsigned int const *deviceIDs) IMPL_WITH_CAMBRICON infinicclStatus_t
infinicclCambriconCommDestroy(infinicclComm_t comm) IMPL_WITH_CAMBRICON 
infinicclStatus_t infinicclCambriconAllReduceSum(infinicclComm_t comm, void *sendbuf,
                              void *recvbuf, size_t count,
                              InfiniDataType_t datatype,
                              infinirtStream_t stream) IMPL_WITH_CAMBRICON

#endif /* INFINICCL_CAMBRICON_H_ */
