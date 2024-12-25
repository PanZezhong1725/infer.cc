#include "infiniccl_maca.h"
#include "../../runtime/runtime.h"
#include <hcr/hc_runtime_api.h>
#include <iostream>
#include <hccl.h>
#include <vector>

#define HCCL_CALL(x)                                                           \
    do {                                                                       \
        hcclResult_t hcclErr = (x);                                            \
        if (hcclErr != hcclSuccess) {                                          \
            std::cerr << "HCCL error: " << hcclErr << " in function "          \
                      << __func__ << std::endl;                                \
            return INFINICCL_STATUS_EXECUTION_FAILED;                          \
        }                                                                      \
    } while (0)

#define SWITCH_DEVICE(deviceId)                                                \
    do {                                                                       \
        hcError_t err = hcSetDevice(deviceId);                             \
        if (err != hcSuccess) {                                              \
            std::cerr << "Maca set device " << deviceId << "error: " << err    \
                      << " in function " << __func__ << std::endl;             \
            return INFINICCL_STATUS_BAD_DEVICE;                                \
        }                                                                      \
    } while (0)

inline hcStream_t getMacaStream(infinirtStream_t stream) {
    if (stream == nullptr) {
        return 0;
    }
    return static_cast<hcStream_t>(stream->stream);
}

inline hcclDataType_t getMacaDtype(InfiniDataType_t datatype) {
    switch (datatype) {
    case INFINI_F32:
        return hcclFloat;
    case INFINI_F16:
        return hcclHalf;
    default:
        return hcclHalf;
    }
}

inline hcclComm_t getHcclComm(infinicclComm_t comm) {
    return static_cast<hcclComm_t>(comm->comm);
}

infinicclStatus_t infinicclMacaCommInitAll(infinicclComm_t *comms,
                                           unsigned int numDevices, unsigned int const *deviceIDs) {
    std::vector<hcclComm_t> mcclComms(numDevices);
    HCCL_CALL(hcclCommInitAll(mcclComms.data(), numDevices, (int const *)deviceIDs));

    for (unsigned int i = 0; i < numDevices; i++) {
        comms[i] = new InfiniComm{DEVICE_METAX, deviceIDs[i], (void *)(mcclComms[i])};
    }
    return INFINICCL_STATUS_SUCCESS;
}

infinicclStatus_t infinicclMacaCommDestroy(infinicclComm_t comm) {
    HCCL_CALL(hcclCommDestroy(getHcclComm(comm)));
    delete comm;
    return INFINICCL_STATUS_SUCCESS;
}

infinicclStatus_t infinicclMacaAllReduceSum(infinicclComm_t comm, void *sendbuf,
                                            void *recvbuf, size_t count,
                                            InfiniDataType_t datatype,
                                            infinirtStream_t stream) {
    if (datatype != INFINI_F32 && datatype != INFINI_F16) {
        return INFINICCL_STATUS_BAD_DATATYPE;
    }
    SWITCH_DEVICE(comm->deviceID);
    HCCL_CALL(hcclAllReduce(sendbuf, recvbuf, count, getMacaDtype(datatype),
                            hcclSum, getHcclComm(comm), getMacaStream(stream)));
    return INFINICCL_STATUS_SUCCESS;
}
