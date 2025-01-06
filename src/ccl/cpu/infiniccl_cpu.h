
#ifndef INFINICCL_CPU_H_
#define INFINICCL_CPU_H_
#include <mpi.h>
#include "infiniccl.h"

infinicclStatus_t infinicclCpuCommInitAll(infinicclComm_t *comms);

infinicclStatus_t infinicclCpuCommDestroy(infinicclComm_t comm);

infinicclStatus_t infinicclCpuAllReduceSum(infinicclComm_t comm, void *sendbuf,
                              void *recvbuf, size_t count,
                              InfiniDataType_t datatype);

#endif /* INFINICCL_CPU_H_ */