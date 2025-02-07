#include "infinirt_musa.h"
#include "musa_runtime.h"
#include "musa_runtime_api.h"
#include <iostream>

#define MUSA_CALL(x)                                                           \
    do {                                                                       \
        musaError_t err = (x);                                                 \
        if (err != musaSuccess) {                                              \
            std::cerr << "Musa error: " << err << " in function " << __func__  \
                      << std::endl;                                            \
            return INFINIRT_STATUS_EXECUTION_FAILED;                           \
        }                                                                      \
    } while (0)

#define SWITCH_DEVICE(deviceId)                                                \
    do {                                                                       \
        musaError_t err = musaSetDevice(deviceId);                             \
        if (err != musaSuccess) {                                              \
            std::cerr << "Musa set device " << deviceId << "error: " << err    \
                      << " in function " << __func__ << std::endl;             \
            return INFINIRT_STATUS_BAD_DEVICE;                                 \
        }                                                                      \
    } while (0)

inline musaStream_t getMusaStream(infinirtStream_t stream) {
    if (stream == nullptr) {
        return 0;
    }
    return static_cast<musaStream_t>(stream->stream);
}

infinirtStatus_t synchronizeMusaDevice(uint32_t deviceId){
    SWITCH_DEVICE(deviceId);
    MUSA_CALL(musaDeviceSynchronize());
    return INFINIRT_STATUS_SUCCESS;
}

infinirtStatus_t createMusaStream(infinirtStream_t *pStream,
                                  uint32_t deviceId) {
    SWITCH_DEVICE(deviceId);
    musaStream_t musa_stream;
    MUSA_CALL(musaStreamCreate(&musa_stream));
    infinirtStream_t stream = new infinirtStream();
    stream->device = DEVICE_MTHREADS;
    stream->device_id = deviceId;
    stream->stream = musa_stream;
    *pStream = stream;
    return INFINIRT_STATUS_SUCCESS;
}
infinirtStatus_t destoryMusaStream(infinirtStream_t stream) {
    SWITCH_DEVICE(stream->device_id);
    MUSA_CALL(musaStreamDestroy(getMusaStream(stream)));
    return INFINIRT_STATUS_SUCCESS;
}

infinirtStatus_t synchronizeMusaStream(infinirtStream_t stream){
    SWITCH_DEVICE(stream->device_id);
    MUSA_CALL(musaStreamSynchronize(getMusaStream(stream)));
    return INFINIRT_STATUS_SUCCESS;
}

infinirtStatus_t createMusaEvent(infinirtEvent_t *pEvent, uint32_t deviceId) {
    SWITCH_DEVICE(deviceId);
    musaEvent_t musa_event;
    MUSA_CALL(musaEventCreate(&musa_event));
    infinirtEvent_t event = new infinirtEvent();
    event->device = DEVICE_MTHREADS;
    event->device_id = deviceId;
    event->event = musa_event;
    *pEvent = event;
    return INFINIRT_STATUS_SUCCESS;
}

infinirtStatus_t destoryMusaEvent(infinirtEvent_t event) {
    SWITCH_DEVICE(event->device_id);
    MUSA_CALL(musaEventDestroy(static_cast<musaEvent_t>(event->event)));
    delete event;
    return INFINIRT_STATUS_SUCCESS;
}

infinirtStatus_t waitMusaEvent(infinirtEvent_t event, infinirtStream_t stream) {
    SWITCH_DEVICE(event->device_id);
    MUSA_CALL(musaStreamWaitEvent(getMusaStream(stream),
                                  static_cast<musaEvent_t>(event->event)));
    return INFINIRT_STATUS_SUCCESS;
}
infinirtStatus_t recordMusaEvent(infinirtEvent_t event,
                                 infinirtStream_t stream) {
    SWITCH_DEVICE(event->device_id);
    MUSA_CALL(musaEventRecord(static_cast<musaEvent_t>(event->event),
                              getMusaStream(stream)));
    return INFINIRT_STATUS_SUCCESS;
}

infinirtStatus_t queryMusaEvent(infinirtEvent_t event) {
    SWITCH_DEVICE(event->device_id);
    musaError_t err = musaEventQuery(static_cast<musaEvent_t>(event->event));
    if (err != musaSuccess){
        return INFINIRT_STATUS_NOT_READY;
    }
    return INFINIRT_STATUS_SUCCESS;
}

infinirtStatus_t synchronizeMusaEvent(infinirtEvent_t event) {
    SWITCH_DEVICE(event->device_id);
    MUSA_CALL(musaEventSynchronize(static_cast<musaEvent_t>(event->event)));
    return INFINIRT_STATUS_SUCCESS;
}

infinirtStatus_t mallocMusa(void **pMemory, uint32_t deviceId, size_t size) {
    SWITCH_DEVICE(deviceId);
    void *musa_ptr;
    MUSA_CALL(musaMalloc(&musa_ptr, size));
    *pMemory = musa_ptr;
    return INFINIRT_STATUS_SUCCESS;
}

infinirtStatus_t mallocMusaAsync(void **pMemory, uint32_t deviceId, size_t size,
                                 infinirtStream_t stream) {
    /*
     * Not implemented by Musa Runtime. Use mallocMusa.
     */
    return mallocMusa(pMemory, deviceId, size);
}

infinirtStatus_t mallocHostMusa(void **pMemory, uint32_t deviceId,
                                size_t size) {
    SWITCH_DEVICE(deviceId);
    void *host_ptr;
    MUSA_CALL(musaMallocHost(&host_ptr, size));
    *pMemory = host_ptr;
    return INFINIRT_STATUS_SUCCESS;
}

infinirtStatus_t freeMusa(void *ptr, uint32_t deviceId) {
    SWITCH_DEVICE(deviceId);
    MUSA_CALL(musaFree(ptr));
    return INFINIRT_STATUS_SUCCESS;
}

infinirtStatus_t freeMusaAsync(void *ptr, uint32_t deviceId,
                               infinirtStream_t stream) {
    /*
     * Not implemented by Musa Runtime. Use freeMusa.
     */
    return freeMusa(ptr, deviceId);
}

infinirtStatus_t freeHostMusa(void *ptr, uint32_t deviceId) {
    SWITCH_DEVICE(deviceId);
    MUSA_CALL(musaFreeHost(ptr));
    return INFINIRT_STATUS_SUCCESS;
}

infinirtStatus_t memcpyHost2Musa(void *dst, uint32_t deviceId, const void *src,
                                 size_t size) {
    SWITCH_DEVICE(deviceId);
    MUSA_CALL(musaMemcpy(dst, src, size, musaMemcpyHostToDevice));
    return INFINIRT_STATUS_SUCCESS;
}

infinirtStatus_t memcpyHost2MusaAsync(void *dst, uint32_t deviceId,
                                      const void *src, size_t size,
                                      infinirtStream_t stream) {
    SWITCH_DEVICE(deviceId);
    MUSA_CALL(musaMemcpyAsync(dst, src, size, musaMemcpyHostToDevice,
                              getMusaStream(stream)));
    return INFINIRT_STATUS_SUCCESS;
}

infinirtStatus_t memcpyMusa2Host(void *dst, const void *src, uint32_t deviceId,
                                 size_t size) {
    SWITCH_DEVICE(deviceId);
    MUSA_CALL(musaMemcpy(dst, src, size, musaMemcpyDeviceToHost));
    return INFINIRT_STATUS_SUCCESS;
}

infinirtStatus_t memcpyMusa(void *dst, const void *src, uint32_t deviceId,
                            size_t size) {
    SWITCH_DEVICE(deviceId);
    MUSA_CALL(musaMemcpy(dst, src, size, musaMemcpyDeviceToDevice));
    return INFINIRT_STATUS_SUCCESS;
}

infinirtStatus_t memcpyMusaAsync(void *dst, const void *src, uint32_t deviceId,
                                 size_t size, infinirtStream_t stream) {
    SWITCH_DEVICE(deviceId);
    MUSA_CALL(musaMemcpyAsync(dst, src, size, musaMemcpyDeviceToDevice,
                              getMusaStream(stream)));
    return INFINIRT_STATUS_SUCCESS;
}
