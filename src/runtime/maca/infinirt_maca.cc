#include "infinirt_maca.h"
#include "hcr/hc_runtime_api.h"
#include "hcr/hc_runtime.h"
#include <iostream>

#define MACA_CALL(x)                                                           \
    do {                                                                       \
        hcError_t err = (x);                                                 \
        if (err != hcSuccess) {                                              \
            std::cerr << "Maca error: " << err << " in function " << __func__  \
                      << std::endl;                                            \
            return INFINIRT_STATUS_EXECUTION_FAILED;                           \
        }                                                                      \
    } while (0)

#define SWITCH_DEVICE(deviceId)                                                \
    do {                                                                       \
        hcError_t err = hcSetDevice(deviceId);                             \
        if (err != hcSuccess) {                                              \
            std::cerr << "Maca set device " << deviceId << "error: " << err    \
                      << " in function " << __func__ << std::endl;             \
            return INFINIRT_STATUS_BAD_DEVICE;                                 \
        }                                                                      \
    } while (0)

inline hcStream_t getMacaStream(infinirtStream_t stream) {
    if (stream == nullptr) {
        return 0;
    }
    return static_cast<hcStream_t>(stream->stream);
}

infinirtStatus_t synchronizeMacaDevice(uint32_t deviceId){
    SWITCH_DEVICE(deviceId);
    MACA_CALL(hcDeviceSynchronize());
    return INFINIRT_STATUS_SUCCESS;
}

infinirtStatus_t createMacaStream(infinirtStream_t *pStream,
                                  uint32_t deviceId) {
    SWITCH_DEVICE(deviceId);
    hcStream_t maca_stream;
    MACA_CALL(hcStreamCreate(&maca_stream));
    infinirtStream_t stream = new infinirtStream();
    stream->device = DEVICE_METAX;
    stream->device_id = deviceId;
    stream->stream = maca_stream;
    *pStream = stream;
    return INFINIRT_STATUS_SUCCESS;
}
infinirtStatus_t destoryMacaStream(infinirtStream_t stream) {
    SWITCH_DEVICE(stream->device_id);
    MACA_CALL(hcStreamDestroy(getMacaStream(stream)));
    return INFINIRT_STATUS_SUCCESS;
}

infinirtStatus_t synchronizeMacaStream(infinirtStream_t stream){
    SWITCH_DEVICE(stream->device_id);
    MACA_CALL(hcStreamSynchronize(getMacaStream(stream)));
    return INFINIRT_STATUS_SUCCESS;
}

infinirtStatus_t createMacaEvent(infinirtEvent_t *pEvent, uint32_t deviceId) {
    SWITCH_DEVICE(deviceId);
    hcEvent_t maca_event;
    MACA_CALL(hcEventCreate(&maca_event));
    infinirtEvent_t event = new infinirtEvent();
    event->device = DEVICE_METAX;
    event->device_id = deviceId;
    event->event = maca_event;
    *pEvent = event;
    return INFINIRT_STATUS_SUCCESS;
}

infinirtStatus_t destoryMacaEvent(infinirtEvent_t event) {
    SWITCH_DEVICE(event->device_id);
    MACA_CALL(hcEventDestroy(static_cast<hcEvent_t>(event->event)));
    delete event;
    return INFINIRT_STATUS_SUCCESS;
}

infinirtStatus_t waitMacaEvent(infinirtEvent_t event, infinirtStream_t stream) {
    SWITCH_DEVICE(event->device_id);
    MACA_CALL(hcStreamWaitEvent(getMacaStream(stream),
                                  static_cast<hcEvent_t>(event->event)));
    return INFINIRT_STATUS_SUCCESS;
}
infinirtStatus_t recordMacaEvent(infinirtEvent_t event,
                                 infinirtStream_t stream) {
    SWITCH_DEVICE(event->device_id);
    MACA_CALL(hcEventRecord(static_cast<hcEvent_t>(event->event),
                              getMacaStream(stream)));
    return INFINIRT_STATUS_SUCCESS;
}

infinirtStatus_t queryMacaEvent(infinirtEvent_t event) {
    SWITCH_DEVICE(event->device_id);
    hcError_t err = hcEventQuery(static_cast<hcEvent_t>(event->event));
    if (err != hcSuccess){
        return INFINIRT_STATUS_NOT_READY;
    }
    return INFINIRT_STATUS_SUCCESS;
}

infinirtStatus_t synchronizeMacaEvent(infinirtEvent_t event) {
    SWITCH_DEVICE(event->device_id);
    MACA_CALL(hcEventSynchronize(static_cast<hcEvent_t>(event->event)));
    return INFINIRT_STATUS_SUCCESS;
}

infinirtStatus_t mallocMaca(void **pMemory, uint32_t deviceId, size_t size) {
    SWITCH_DEVICE(deviceId);
    void *maca_ptr;
    MACA_CALL(hcMalloc(&maca_ptr, size));
    *pMemory = maca_ptr;
    return INFINIRT_STATUS_SUCCESS;
}

infinirtStatus_t mallocMacaAsync(void **pMemory, uint32_t deviceId, size_t size,
                                 infinirtStream_t stream) {
    SWITCH_DEVICE(deviceId);
    void *maca_ptr;
    MACA_CALL(hcMallocAsync(&maca_ptr, size, getMacaStream(stream)));
    *pMemory = maca_ptr;
    return INFINIRT_STATUS_SUCCESS;
}

infinirtStatus_t mallocHostMaca(void **pMemory, uint32_t deviceId,
                                size_t size) {
    SWITCH_DEVICE(deviceId);
    void *host_ptr;
    MACA_CALL(hcMallocHost(&host_ptr, size));
    *pMemory = host_ptr;
    return INFINIRT_STATUS_SUCCESS;
}

infinirtStatus_t freeMaca(void *ptr, uint32_t deviceId) {
    SWITCH_DEVICE(deviceId);
    MACA_CALL(hcFree(ptr));
    return INFINIRT_STATUS_SUCCESS;
}

infinirtStatus_t freeMacaAsync(void *ptr, uint32_t deviceId,
                               infinirtStream_t stream) {
    SWITCH_DEVICE(deviceId);
    MACA_CALL(hcFreeAsync(ptr, getMacaStream(stream)));
    return INFINIRT_STATUS_SUCCESS;
}

infinirtStatus_t freeHostMaca(void *ptr, uint32_t deviceId) {
    SWITCH_DEVICE(deviceId);
    MACA_CALL(hcFreeHost(ptr));
    return INFINIRT_STATUS_SUCCESS;
}

infinirtStatus_t memcpyHost2Maca(void *dst, uint32_t deviceId, const void *src,
                                 size_t size) {
    SWITCH_DEVICE(deviceId);
    for (size_t i = 0; i < size / 4096; ++i){
        MACA_CALL(hcMemcpy(((uint8_t *) dst) + i * 4096, ((uint8_t *) src) + i * 4096, 4096, hcMemcpyHostToDevice));
    }
    return INFINIRT_STATUS_SUCCESS;
}

infinirtStatus_t memcpyHost2MacaAsync(void *dst, uint32_t deviceId,
                                      const void *src, size_t size,
                                      infinirtStream_t stream) {
    SWITCH_DEVICE(deviceId);
    // printf("memcpyh2d_async ");
    MACA_CALL(hcMemcpyAsync(dst, src, size, hcMemcpyHostToDevice,
                              getMacaStream(stream)));
    return INFINIRT_STATUS_SUCCESS;
}

infinirtStatus_t memcpyMaca2Host(void *dst, const void *src, uint32_t deviceId,
                                 size_t size) {
    SWITCH_DEVICE(deviceId);
    MACA_CALL(hcMemcpy(dst, src, size, hcMemcpyDeviceToHost));
    return INFINIRT_STATUS_SUCCESS;
}

infinirtStatus_t memcpyMaca(void *dst, const void *src, uint32_t deviceId,
                            size_t size) {
    SWITCH_DEVICE(deviceId);
    MACA_CALL(hcMemcpy(dst, src, size, hcMemcpyDeviceToDevice));
    return INFINIRT_STATUS_SUCCESS;
}

infinirtStatus_t memcpyMacaAsync(void *dst, const void *src, uint32_t deviceId,
                                 size_t size, infinirtStream_t stream) {
    SWITCH_DEVICE(deviceId);
    MACA_CALL(hcMemcpyAsync(dst, src, size, hcMemcpyDeviceToDevice,
                              getMacaStream(stream)));
    return INFINIRT_STATUS_SUCCESS;
}
