#include "infiniccl.h"
#include "../runtime/runtime.h"
#include "./ascend/infiniccl_ascend.h"
#include "./cambricon/infiniccl_cambricon.h"
#include "./cuda/infiniccl_cuda.h"
#include "./kunlun/infiniccl_kunlun.h"
#include "./maca/infiniccl_maca.h"
#include "./musa/infiniccl_musa.h"

__C infinicclStatus_t infinicclCommInitAll(DeviceType deviceType,
                                           infinicclComm_t *comms,
                                           unsigned numDevices,
                                           unsigned const *deviceIDs) {
    switch (deviceType) {
    case DEVICE_NVIDIA:
        return infinicclCudaCommInitAll(comms, numDevices, deviceIDs);
    case DEVICE_CAMBRICON:
        return infinicclCambriconCommInitAll(comms, numDevices, deviceIDs);
    case DEVICE_ASCEND:
        return infinicclAscendCommInitAll(comms, numDevices, deviceIDs);
    case DEVICE_KUNLUN:
        return infinicclKunlunCommInitAll(comms, numDevices, deviceIDs);
    case DEVICE_METAX:
        return infinicclMacaCommInitAll(comms, numDevices, deviceIDs);
    case DEVICE_MTHREADS:
        return infinicclMusaCommInitAll(comms, numDevices, deviceIDs);
    default:
        return INFINICCL_STATUS_DEVICE_NOT_SUPPORTED;
    }
}

__C infinicclStatus_t infinicclCommDestroy(infinicclComm_t comm) {
    if (comm == nullptr) {
        return INFINICCL_STATUS_SUCCESS;
    }
    switch (comm->deviceType) {
    case DEVICE_NVIDIA:
        return infinicclCudaCommDestroy(comm);
    case DEVICE_CAMBRICON:
        return infinicclCambriconCommDestroy(comm);
    case DEVICE_ASCEND:
        return infinicclAscendCommDestroy(comm);
    case DEVICE_KUNLUN:
        return infinicclKunlunCommDestroy(comm);
    case DEVICE_METAX:
        return infinicclMacaCommDestroy(comm);
    case DEVICE_MTHREADS:
        return infinicclMusaCommDestroy(comm);
    default:
        return INFINICCL_STATUS_DEVICE_NOT_SUPPORTED;
    }
}

__C infinicclStatus_t infinicclAllReduceSum(infinicclComm_t comm, void *sendbuf,
                                            void *recvbuf, size_t count,
                                            InfiniDataType_t datatype,
                                            infinirtStream_t stream) {
    if (comm == nullptr) {
        return INFINICCL_STATUS_COMMUNICATOR_UNINITIALIZED;
    }
    if (stream != nullptr && comm->deviceType != stream->device) {
        return INFINICCL_STATUS_DEVICE_MISMATCH;
    }
    switch (comm->deviceType) {
    case DEVICE_NVIDIA:
        return infinicclCudaAllReduceSum(comm, sendbuf, recvbuf, count,
                                         datatype, stream);
    case DEVICE_CAMBRICON:
        return infinicclCambriconAllReduceSum(comm, sendbuf, recvbuf, count,
                                              datatype, stream);
    case DEVICE_ASCEND:
        return infinicclAscendAllReduceSum(comm, sendbuf, recvbuf, count,
                                           datatype, stream);
    case DEVICE_KUNLUN:
        return infinicclKunlunAllReduceSum(comm, sendbuf, recvbuf, count,
                                           datatype, stream);
    case DEVICE_METAX:
        return infinicclMacaAllReduceSum(comm, sendbuf, recvbuf, count,
                                         datatype, stream);
    case DEVICE_MTHREADS:
        return infinicclMusaAllReduceSum(comm, sendbuf, recvbuf, count,
                                         datatype, stream);
    default:
        return INFINICCL_STATUS_DEVICE_NOT_SUPPORTED;
    }
}
