#include "infiniccl_cambricon.h"
#include "../../runtime/runtime.h"
#include "cncl.h"
#include "cnrt.h"
#include <iostream>
#include <vector>

#define CNCL_CALL(x)                                                           \
    do {                                                                       \
        cnclResult_t err = (x);                                                \
        if (err != CNCL_RET_SUCCESS) {                                         \
            std::cerr << "CNCL Error: " << err << " in function " << __func__  \
                      << std::endl;                                            \
            return INFINICCL_STATUS_EXECUTION_FAILED;                          \
        }                                                                      \
    } while (0)

#define SWITCH_DEVICE(deviceId)                                                \
    do {                                                                       \
        cnrtRet_t err = cnrtSetDevice(deviceId);                               \
        if (err != cnrtSuccess) {                                              \
            std::cerr << "CNRT set device " << deviceId << " Error: " << err   \
                      << " in function " << __func__ << std::endl;             \
            return INFINICCL_STATUS_BAD_DEVICE;                                \
        }                                                                      \
    } while (0)

inline cnrtQueue_t getCnrtQueue(infinirtStream_t stream) {
    if (stream == nullptr) {
        return (cnrtQueue_t)(0);
    }
    return (cnrtQueue_t)(stream->stream);
}

inline cnclDataType_t getCnclDtype(InfiniDataType_t datatype) {
    switch (datatype) {
    case INFINI_F32:
        return cnclFloat32;
    case INFINI_F16:
        return cnclFloat16;
    default:
        return cnclInvalid;
    }
}

inline cnclComm_t getCnclComm(infinicclComm_t comm) {
    return static_cast<cnclComm_t>(comm->comm);
}

infinicclStatus_t infinicclCambriconCommInitAll(infinicclComm_t *comms,
                                                unsigned int numDevices,
                                                unsigned int const *deviceIDs) {
    std::vector<cnclComm_t> cnclComms(numDevices);

    std::vector<int> rank_list(numDevices);
    for (int i = 0; i < (int)numDevices; i++) {
        rank_list[i] = i;
    }
    CNCL_CALL(cnclInitComms(cnclComms.data(), numDevices,
                            (int const *)deviceIDs, rank_list.data(),
                            numDevices, NULL));
    for (int i = 0; i < (int)numDevices; i++) {
        comms[i] =
            new InfiniComm{DEVICE_CAMBRICON, deviceIDs[i], (void *)(cnclComms[i])};
    }
    return INFINICCL_STATUS_SUCCESS;
}

infinicclStatus_t infinicclCambriconCommDestroy(infinicclComm_t comm) {
    CNCL_CALL(cnclFreeComm(getCnclComm(comm)));
    delete comm;
    return INFINICCL_STATUS_SUCCESS;
}

infinicclStatus_t infinicclCambriconAllReduceSum(infinicclComm_t comm, void *sendbuf,
                                            void *recvbuf, size_t count,
                                            InfiniDataType_t datatype,
                                            infinirtStream_t stream) {
    if (datatype != INFINI_F32 && datatype != INFINI_F16) {
        return INFINICCL_STATUS_BAD_DATATYPE;
    }
    SWITCH_DEVICE(comm->deviceID);
    CNCL_CALL(cnclAllReduce(sendbuf, recvbuf, count, getCnclDtype(datatype),
                            cnclSum, getCnclComm(comm), getCnrtQueue(stream)));
    return INFINICCL_STATUS_SUCCESS;
}
