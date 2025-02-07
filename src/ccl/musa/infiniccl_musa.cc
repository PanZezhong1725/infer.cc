#include "infiniccl_musa.h"
#include "../../runtime/runtime.h"
#include "musa_runtime.h"
#include "musa_runtime_api.h"
#include <iostream>
#include "mccl.h"
#include <vector>

#define MCCL_CALL(x)                                                           \
    do {                                                                       \
        mcclResult_t mcclErr = (x);                                            \
        if (mcclErr != mcclSuccess) {                                          \
            std::cerr << "MCCL error: " << mcclErr << " in function "          \
                      << __func__ << std::endl;                                \
            return INFINICCL_STATUS_EXECUTION_FAILED;                          \
        }                                                                      \
    } while (0)

#define SWITCH_DEVICE(deviceId)                                                \
    do {                                                                       \
        musaError_t err = musaSetDevice(deviceId);                             \
        if (err != musaSuccess) {                                              \
            std::cerr << "Musa set device " << deviceId << "error: " << err    \
                      << " in function " << __func__ << std::endl;             \
            return INFINICCL_STATUS_BAD_DEVICE;                                \
        }                                                                      \
    } while (0)

inline musaStream_t getMusaStream(infinirtStream_t stream) {
    if (stream == nullptr) {
        return 0;
    }
    return static_cast<musaStream_t>(stream->stream);
}

inline mcclDataType_t getMusaDtype(InfiniDataType_t datatype) {
    switch (datatype) {
    case INFINI_F32:
        return mcclFloat;
    case INFINI_F16:
        return mcclHalf;
    default:
        return mcclHalf;
    }
}

inline mcclComm_t getMcclComm(infinicclComm_t comm) {
    return static_cast<mcclComm_t>(comm->comm);
}

infinicclStatus_t infinicclMusaCommInitAll(infinicclComm_t *comms,
                                           unsigned int numDevices, unsigned int const *deviceIDs) {
    std::vector<mcclComm_t> mcclComms(numDevices);
    MCCL_CALL(mcclCommInitAll(mcclComms.data(), numDevices, (int const *)deviceIDs));

    for (unsigned int i = 0; i < numDevices; i++) {
        comms[i] = new InfiniComm{DEVICE_MTHREADS, deviceIDs[i], (void *)(mcclComms[i])};
    }
    return INFINICCL_STATUS_SUCCESS;
}

infinicclStatus_t infinicclMusaCommDestroy(infinicclComm_t comm) {
    MCCL_CALL(mcclCommDestroy(getMcclComm(comm)));
    delete comm;
    return INFINICCL_STATUS_SUCCESS;
}

infinicclStatus_t infinicclMusaAllReduceSum(infinicclComm_t comm, void *sendbuf,
                                            void *recvbuf, size_t count,
                                            InfiniDataType_t datatype,
                                            infinirtStream_t stream) {
    if (datatype != INFINI_F32 && datatype != INFINI_F16) {
        return INFINICCL_STATUS_BAD_DATATYPE;
    }
    SWITCH_DEVICE(comm->deviceID);
    MCCL_CALL(mcclAllReduce(sendbuf, recvbuf, count, getMusaDtype(datatype),
                            mcclSum, getMcclComm(comm), getMusaStream(stream)));
    return INFINICCL_STATUS_SUCCESS;
}
