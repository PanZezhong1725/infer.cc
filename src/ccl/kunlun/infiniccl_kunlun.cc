#include "infiniccl_kunlun.h"
#include "../../runtime/runtime.h"
#include <cstdio>
#include <omp.h>
#include <vector>
#include <xpu/bkcl.h>
#include <xpu/runtime.h>
#include <xpu/runtime_ex.h>

#define XCCL_CALL(x)                                                           \
    do {                                                                       \
        auto err = (x);                                                        \
        if (BKCL_SUCCESS != err) {                                             \
            fprintf(stderr, "KUNLUN XCCL error in %s:%i.\n", __FILE__,         \
                    __LINE__);                                                 \
            return INFINICCL_STATUS_EXECUTION_FAILED;                          \
        }                                                                      \
    } while (0)

#define SWITCH_DEVICE(deviceId)                                                \
    do {                                                                       \
        auto err = xpu_set_device(deviceId);                                   \
        if (err != XPU_SUCCESS) {                                              \
            fprintf(stderr, "KUNLUN Runtime error in %s:%i : %s.\n", __FILE__, \
                    __LINE__, xpu_strerror(err));                              \
            return INFINICCL_STATUS_BAD_DEVICE;                                 \
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

BKCLResult_t bkclCommInitAllOmp(BKCLContext_t *ctx, int rank, int nranks,
                                BKCLUniqueId &unique_id) {
    if (nranks <= 0) {
        return BKCL_INVALID_ARGUMENT;
    }
    // rank 0 gen unique_id
    if (rank == 0) {
        BKCLResult_t result = bkcl_get_unique_id(&unique_id);
        if (result != BKCL_SUCCESS) {
            return result;
        }
#pragma omp flush(unique_id)
    }
    // wait all
#pragma omp barrier
    // every rank init ctx
    return bkcl_init_rank(ctx, rank, nranks, &unique_id);
}

infinicclStatus_t infinicclKunlunCommInitAll(infinicclComm_t *comms,
                                             uint32_t numDevices,
                                             uint32_t const *deviceIDs) {
    BKCLUniqueId unique_id;
    std::vector<BKCLContext_t> xcclComms(numDevices);
    BKCLResult_t thread_result = BKCL_SUCCESS;

#pragma omp parallel num_threads(numDevices)
    {
        int rank = omp_get_thread_num();
        BKCLResult_t result =
            bkclCommInitAllOmp(&xcclComms[rank], rank, numDevices, unique_id);
        if (result != BKCL_SUCCESS) {
#pragma omp critical
            thread_result = result;
        }
    }
    if (thread_result != BKCL_SUCCESS) {
        return INFINICCL_STATUS_COMMUNICATOR_UNINITIALIZED;
    }
    for (uint32_t i = 0; i < numDevices; i++) {
        comms[i] =
            new InfiniComm{DEVICE_KUNLUN, deviceIDs[i], (void *)(xcclComms[i])};
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
    XCCL_CALL(bkcl_all_reduce(getXcclComm(comm), sendbuf, recvbuf, count,
                              getKunlunDtype(datatype), BKCLOp::BKCL_ADD,
                              getKunlunStream(stream)));
    return INFINICCL_STATUS_SUCCESS;
}
