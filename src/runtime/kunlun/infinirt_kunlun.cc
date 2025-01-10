#include "infinirt_kunlun.h"
#include "xpu/runtime.h"
#include "xpu/runtime_ex.h"
#include <iostream>

#define KUNLUN_CALL(x)                                                         \
    do {                                                                       \
        auto err = (x);                                                        \
        if (err != XPU_SUCCESS) {                                              \
            std::cerr << "KUNLUN Runtime error in " << __FILE__ << ":" << __LINE__ \
                      << " : " << xpu_strerror(err) << ".\n";                  \
            return INFINIRT_STATUS_EXECUTION_FAILED;                           \
        }                                                                      \
    } while (0)

#define SWITCH_DEVICE(deviceId)                                                \
    do {                                                                       \
        auto err = xpu_set_device(deviceId);                                   \
        if (err != XPU_SUCCESS) {                                              \
            std::cerr << "KUNLUN Runtime error in " << __FILE__ << ":" << __LINE__ \
                      << " : " << xpu_strerror(err) << ".\n";                  \
            return INFINIRT_STATUS_BAD_DEVICE;                                 \
        }                                                                      \
    } while (0)

inline XPUStream getKunlunStream(infinirtStream_t stream) {
    if (stream == nullptr) {
        return 0;
    }
    return static_cast<XPUStream>(stream->stream);
}

infinirtStatus_t createKunlunStream(infinirtStream_t *pStream,
                                    uint32_t deviceId) {
    SWITCH_DEVICE(deviceId);
    XPUStream xpu_stream;
    KUNLUN_CALL(xpu_stream_create(&xpu_stream));
    infinirtStream_t stream =
        new infinirtStream{DEVICE_KUNLUN, deviceId, xpu_stream};
    *pStream = stream;
    return INFINIRT_STATUS_SUCCESS;
}

infinirtStatus_t destoryKunlunStream(infinirtStream_t stream) {
    SWITCH_DEVICE(stream->device_id);
    KUNLUN_CALL(xpu_stream_destroy(getKunlunStream(stream)));
    return INFINIRT_STATUS_SUCCESS;
}

infinirtStatus_t synchronizeKunlunStream(infinirtStream_t stream) {
    SWITCH_DEVICE(stream->device_id);
    KUNLUN_CALL(xpu_wait(getKunlunStream(stream)));
    return INFINIRT_STATUS_SUCCESS;
}

infinirtStatus_t createKunlunEvent(infinirtEvent_t *pEvent, uint32_t deviceId) {
    SWITCH_DEVICE(deviceId);
    XPUEvent xpu_event;
    KUNLUN_CALL(xpu_event_create(&xpu_event));
    infinirtEvent_t event =
        new infinirtEvent{DEVICE_KUNLUN, deviceId, xpu_event};
    *pEvent = event;
    return INFINIRT_STATUS_SUCCESS;
}

infinirtStatus_t destoryKunlunEvent(infinirtEvent_t event) {
    SWITCH_DEVICE(event->device_id);
    KUNLUN_CALL(xpu_event_destroy(static_cast<XPUEvent>(event->event)));
    delete event;
    return INFINIRT_STATUS_SUCCESS;
}

infinirtStatus_t waitKunlunEvent(infinirtEvent_t event,
                                 infinirtStream_t stream) {
    SWITCH_DEVICE(event->device_id);
    KUNLUN_CALL(xpu_stream_wait_event(getKunlunStream(stream),
                                      static_cast<XPUEvent>(event->event)));
    return INFINIRT_STATUS_SUCCESS;
}

infinirtStatus_t recordKunlunEvent(infinirtEvent_t event,
                                   infinirtStream_t stream) {
    SWITCH_DEVICE(event->device_id);
    KUNLUN_CALL(xpu_event_record(static_cast<XPUEvent>(event->event),
                                 getKunlunStream(stream)));
    return INFINIRT_STATUS_SUCCESS;
}

infinirtStatus_t queryKunlunEvent(infinirtEvent_t event) {
    SWITCH_DEVICE(event->device_id);
    return INFINIRT_STATUS_SUCCESS;
}

infinirtStatus_t synchronizeKunlunEvent(infinirtEvent_t event) {
    SWITCH_DEVICE(event->device_id);
    KUNLUN_CALL(xpu_event_wait(static_cast<XPUEvent>(event->event)));
    return INFINIRT_STATUS_SUCCESS;
}

infinirtStatus_t mallocKunlun(void **pMemory, uint32_t deviceId, size_t size) {
    SWITCH_DEVICE(deviceId);
    void *kunlun_ptr;
    KUNLUN_CALL(xpu_malloc(&kunlun_ptr, size));
    *pMemory = kunlun_ptr;
    return INFINIRT_STATUS_SUCCESS;
}

infinirtStatus_t freeKunlun(void *ptr, uint32_t deviceId) {
    SWITCH_DEVICE(deviceId);
    KUNLUN_CALL(xpu_free(ptr));
    return INFINIRT_STATUS_SUCCESS;
}

infinirtStatus_t mallocHostKunlun(void **pMemory, uint32_t deviceId,
                                  size_t size) {
    SWITCH_DEVICE(deviceId);
    void *host_ptr;
    KUNLUN_CALL(xpu_host_alloc(&host_ptr, size, 0));
    *pMemory = host_ptr;
    return INFINIRT_STATUS_SUCCESS;
}

infinirtStatus_t freeHostKunlun(void *ptr, uint32_t deviceId) {
    SWITCH_DEVICE(deviceId);
    KUNLUN_CALL(xpu_host_free(ptr));
    return INFINIRT_STATUS_SUCCESS;
}

infinirtStatus_t memcpyHost2Kunlun(void *dst, uint32_t deviceId,
                                   const void *src, size_t size) {
    SWITCH_DEVICE(deviceId);
    KUNLUN_CALL(xpu_memcpy(dst, src, size, XPUMemcpyKind::XPU_HOST_TO_DEVICE));
    return INFINIRT_STATUS_SUCCESS;
}

infinirtStatus_t memcpyKunlun2Host(void *dst, const void *src,
                                   uint32_t deviceId, size_t size) {
    SWITCH_DEVICE(deviceId);
    KUNLUN_CALL(xpu_memcpy(dst, src, size, XPUMemcpyKind::XPU_DEVICE_TO_HOST));
    return INFINIRT_STATUS_SUCCESS;
}

infinirtStatus_t memcpyKunlun(void *dst, const void *src, uint32_t deviceId,
                              size_t size) {
    SWITCH_DEVICE(deviceId);
    KUNLUN_CALL(
        xpu_memcpy(dst, src, size, XPUMemcpyKind::XPU_DEVICE_TO_DEVICE));
    return INFINIRT_STATUS_SUCCESS;
}
