#ifndef INFINICCL_ASCEND_H_
#define INFINICCL_ASCEND_H_
#include "infiniccl.h"

#if defined(ENABLE_ASCEND_NPU) && defined (ENABLE_CCL)
#define IMPL_WITH_ASCEND ;
#else
#define IMPL_WITH_ASCEND { return INFINICCL_STATUS_DEVICE_NOT_SUPPORTED; }
#endif

infinicclStatus_t infinicclAscendCommInitAll(
    infinicclComm_t *comms, 
    unsigned int numDevices, 
    unsigned int const *deviceIDs
) IMPL_WITH_ASCEND 

infinicclStatus_t infinicclAscendCommDestroy(
    infinicclComm_t comm
) IMPL_WITH_ASCEND 

infinicclStatus_t infinicclAscendAllReduceSum(
    infinicclComm_t comm, 
    void *sendbuf,
    void *recvbuf, size_t count,
    InfiniDataType_t datatype,
    infinirtStream_t stream
) IMPL_WITH_ASCEND

#endif /* INFINICCL_ASCEND_H_ */
