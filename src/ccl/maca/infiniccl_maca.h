#ifndef INFINICCL_MACA_H_
#define INFINICCL_MACA_H_
#include "infiniccl.h"

#ifdef ENABLE_METAX_GPU
#define IMPL_WITH_MACA ;
#else
#define IMPL_WITH_MACA { return INFINICCL_STATUS_DEVICE_NOT_SUPPORTED; }
#endif

infinicclStatus_t
infinicclMacaCommInitAll(infinicclComm_t *comms, unsigned int numDevices, unsigned int const *deviceIDs) IMPL_WITH_MACA 
infinicclStatus_t
infinicclMacaCommDestroy(infinicclComm_t comm) IMPL_WITH_MACA 
infinicclStatus_t infinicclMacaAllReduceSum(infinicclComm_t comm, void *sendbuf,
                              void *recvbuf, size_t count,
                              InfiniDataType_t datatype,
                              infinirtStream_t stream) IMPL_WITH_MACA

#endif /* INFINICCL_MACA_H_ */
