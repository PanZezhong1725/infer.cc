#include "infiniccl_kunlun.h"
#include "../../runtime/runtime.h"
#include <iostream>
#include <pthread.h>
#include <vector>
#include <xpu/bkcl.h>
#include <xpu/runtime.h>
#include <xpu/runtime_ex.h>

#define XCCL_CALL(x)                                                           \
    do {                                                                       \
        auto err = (x);                                                        \
        if (BKCL_SUCCESS != err) {                                             \
            std::cerr << "KUNLUN XCCL error in " << __FILE__ << ":"            \
                      << __LINE__ << ".\n";                                    \
            return INFINICCL_STATUS_EXECUTION_FAILED;                          \
        }                                                                      \
    } while (0)

#define SWITCH_DEVICE(deviceId)                                                \
    do {                                                                       \
        auto err = xpu_set_device(deviceId);                                   \
        if (err != XPU_SUCCESS) {                                              \
            std::cerr << "KUNLUN Runtime error in " << __FILE__ << ":"         \
                      << __LINE__ << " : " << xpu_strerror(err) << ".\n";      \
            return INFINICCL_STATUS_BAD_DEVICE;                                \
        }                                                                      \
    } while (0)

inline XPUStream getKunlunStream(infinirtStream_t stream) {
    if (stream == nullptr) {
        return 0;
    }
    return static_cast<XPUStream>(stream->stream);
}

inline BKCLDataType getKunlunDtype(InfiniDataType_t datatype) {
    switch (datatype) {
    case INFINI_F32:
        return BKCL_FLOAT;
    case INFINI_F16:
        return BKCL_FLOAT16;
    default:
        return BKCL_FLOAT16;
    }
}

inline BKCLContext_t getXcclComm(infinicclComm_t comm) {
    return static_cast<BKCLContext_t>(comm->comm);
}

// Parameters for thread initialization
struct InitRankParam {
    int rank;
    int nranks;
    BKCLUniqueId id;
    BKCLContext_t *ctx;
    int *errorFlag; // Shared error flag
};

// Thread function for initializing BKCL context
void *init_bkcl_context_func(void *args) {
    auto *param = static_cast<InitRankParam *>(args);
    xpu_set_device(param->rank);
    int result =
        bkcl_init_rank(param->ctx, param->rank, param->nranks, &param->id);

    if (result != BKCL_SUCCESS) {
        std::cerr << "rank " << param->rank
                  << " init failed with error code: " << result << "\n";
        *(param->errorFlag) = result;
    }
    return nullptr;
}

// Interface function for BKCL communication initialization
infinicclStatus_t infinicclKunlunCommInitAll(infinicclComm_t *comms,
                                             unsigned int numDevices,
                                             unsigned int const *deviceIDs) {
    // Allocate memory for BKCL contexts
    std::vector<BKCLContext_t> ctxs(numDevices);

    // Generate BKCL unique ID
    BKCLUniqueId id;
    BKCLResult_t uniqueIdResult = bkcl_get_unique_id(&id);
    if (uniqueIdResult != BKCL_SUCCESS) {
        std::cerr << "Failed to get BKCL unique ID. Error code: "
                  << uniqueIdResult << "\n";
        return INFINICCL_STATUS_INVALID_ARGUMENT;
    }

    // Allocate threads and parameters
    std::vector<pthread_t> tids(numDevices);
    std::vector<InitRankParam> params(numDevices);
    int globalError = BKCL_SUCCESS;

    for (unsigned int i = 0; i < numDevices; i++) {
        params[i].rank = i;
        params[i].nranks = numDevices;
        params[i].id = id;
        params[i].ctx = &ctxs[i];
        params[i].errorFlag = &globalError;

        pthread_create(&tids[i], nullptr, init_bkcl_context_func,
                       static_cast<void *>(&params[i]));
    }

    // Wait for all threads to complete
    for (unsigned int i = 0; i < numDevices; i++) {
        pthread_join(tids[i], nullptr);
    }

    // Check for errors during initialization
    if (globalError != BKCL_SUCCESS) {
        std::cerr << "BKCL context initialization failed.\n";
        return INFINICCL_STATUS_COMMUNICATOR_UNINITIALIZED;
    }

    // Create InfiniComm instances for each device
    for (unsigned int i = 0; i < numDevices; i++) {
        comms[i] = new InfiniComm{DEVICE_KUNLUN, deviceIDs[i],
                                  static_cast<void *>(ctxs[i])};
    }

    return INFINICCL_STATUS_SUCCESS;
}

infinicclStatus_t infinicclKunlunCommDestroy(infinicclComm_t comm) {
    XCCL_CALL(bkcl_destroy_context(getXcclComm(comm)));
    delete comm;
    return INFINICCL_STATUS_SUCCESS;
}

infinicclStatus_t infinicclKunlunAllReduceSum(infinicclComm_t comm,
                                              void *sendbuf, void *recvbuf,
                                              size_t count,
                                              InfiniDataType_t datatype,
                                              infinirtStream_t stream) {
    if (datatype != INFINI_F32 && datatype != INFINI_F16) {
        return INFINICCL_STATUS_BAD_DATATYPE;
    }
    SWITCH_DEVICE(comm->deviceID);
    bkcl_all_reduce(getXcclComm(comm), sendbuf, recvbuf, count,
                    getKunlunDtype(datatype), BKCLOp::BKCL_ADD,
                    getKunlunStream(stream));
    std::vector<float> out(count);
    xpu_memcpy(out.data(), recvbuf, count * 4,
               XPUMemcpyKind::XPU_DEVICE_TO_HOST);
    return INFINICCL_STATUS_SUCCESS;
}
