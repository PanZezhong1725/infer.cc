#ifndef __INFINICCL_KUNLUN_H__
#define __INFINICCL_KUNLUN_H__
#include "infiniccl.h"

#if defined(ENABLE_KUNLUN_XPU) && defined(ENABLE_CCL)
#define IMPL_WITH_KUNLUN ;
#else
#define IMPL_WITH_KUNLUN {return INFINICCL_STATUS_DEVICE_NOT_SUPPORTED; }
#endif

infinicclStatus_t infinicclKunlunCommInitAll(infinicclComm_t *comms,
                                             uint32_t numDevices, 
                                             uint32_t const *deviceIDs) IMPL_WITH_KUNLUN
infinicclStatus_t infinicclKunlunCommDestroy(infinicclComm_t comm) IMPL_WITH_KUNLUN
infinicclStatus_t infinicclKunlunAllReduceSum(infinicclComm_t comm,
                                              void *sendbuf, 
                                              void *recvbuf,
                                              size_t count, 
                                              InfiniDataType_t datatype,
                                              infinirtStream_t stream) IMPL_WITH_KUNLUN

#endif
