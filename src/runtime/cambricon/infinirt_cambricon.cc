#include "infinirt_cambricon.h"
#include "cnrt.h"
#include <iostream>

#define CN_CALL(x)                                                             \
    do {                                                                       \
        cnrtRet_t err = (x);                                                   \
        if (err != cnrtSuccess) {                                              \
            std::cerr << "CNRT Error: " << err << " in function " << __func__  \
                      << std::endl;                                            \
            return INFINIRT_STATUS_EXECUTION_FAILED;                           \
        }                                                                      \
    } while (0)

#define SWITCH_DEVICE(deviceId)                                                \
    do {                                                                       \
        cnrtRet_t err = cnrtSetDevice(deviceId);                               \
        if (err != cnrtSuccess) {                                              \
            std::cerr << "CNRT set device " << deviceId << " Error: " << err   \
                      << " in function " << __func__ << std::endl;             \
            return INFINIRT_STATUS_BAD_DEVICE;                                 \
        }                                                                      \
    } while (0)

inline cnrtQueue_t getCnrtQueue(infinirtStream_t stream){
    return (cnrtQueue_t)(stream->stream);
}

inline cnrtNotifier_t getCnrtNotifier(infinirtEvent_t event){
    return (cnrtNotifier_t)(event->event);
}

infinirtStatus_t synchronizeCambriconDevice(uint32_t deviceId) {
    SWITCH_DEVICE(deviceId);
    CN_CALL(cnrtSyncDevice());
    return INFINIRT_STATUS_SUCCESS;
}

infinirtStatus_t createCambriconStream(infinirtStream_t *pStream,
                                       uint32_t deviceId) {
    SWITCH_DEVICE(deviceId);
    cnrtQueue_t queue;
    CN_CALL(cnrtQueueCreate(&queue));
    infinirtStream_t stream = new infinirtStream();
    stream->device = DEVICE_CAMBRICON;
    stream->device_id = deviceId;
    stream->stream = queue;
    *pStream = stream;
    return INFINIRT_STATUS_SUCCESS;
}

infinirtStatus_t destoryCambriconStream(infinirtStream_t stream) {
    SWITCH_DEVICE(stream->device_id);
    CN_CALL(cnrtQueueDestroy(getCnrtQueue(stream)));
    delete stream;
    return INFINIRT_STATUS_SUCCESS;
}

infinirtStatus_t synchronizeCambriconStream(infinirtStream_t stream) {
    SWITCH_DEVICE(stream->device_id);
    CN_CALL(cnrtQueueSync(getCnrtQueue(stream)));
    return INFINIRT_STATUS_SUCCESS;
}

infinirtStatus_t createCambriconEvent(infinirtEvent_t *pEvent,
                                      uint32_t deviceId) {
    SWITCH_DEVICE(deviceId);
    cnrtNotifier_t notifier;
    CN_CALL(cnrtNotifierCreate(&notifier));
    infinirtEvent_t event = new infinirtEvent();
    event->device = DEVICE_CAMBRICON;
    event->device_id = deviceId;
    event->event = notifier;
    *pEvent = event;
    return INFINIRT_STATUS_SUCCESS;
}

infinirtStatus_t destoryCambriconEvent(infinirtEvent_t event) {
    SWITCH_DEVICE(event->device_id);
    CN_CALL(cnrtNotifierDestroy(getCnrtNotifier(event)));
    delete event;
    return INFINIRT_STATUS_SUCCESS;
}

infinirtStatus_t waitCambriconEvent(infinirtEvent_t event,
                                    infinirtStream_t stream) {
    SWITCH_DEVICE(event->device_id);
    CN_CALL(cnrtQueueWaitNotifier(getCnrtNotifier(event), getCnrtQueue(stream), 0));
    return INFINIRT_STATUS_SUCCESS;
}

infinirtStatus_t recordCambriconEvent(infinirtEvent_t event,
                                      infinirtStream_t stream) {
    SWITCH_DEVICE(event->device_id);
    CN_CALL(cnrtPlaceNotifier(getCnrtNotifier(event), getCnrtQueue(stream)));
    return INFINIRT_STATUS_SUCCESS;
}

infinirtStatus_t queryCambriconEvent(infinirtEvent_t event) {
    SWITCH_DEVICE(event->device_id);
    cnrtRet_t err = cnrtQueryNotifier(getCnrtNotifier(event));
    if (err == cnrtSuccess) {
        return INFINIRT_STATUS_SUCCESS;
    }
    return INFINIRT_STATUS_NOT_READY;
}

infinirtStatus_t synchronizeCambriconEvent(infinirtEvent_t event) {
    SWITCH_DEVICE(event->device_id);
    CN_CALL(cnrtWaitNotifier(getCnrtNotifier(event)));
    return INFINIRT_STATUS_SUCCESS;
}

infinirtStatus_t mallocCambricon(void **pMemory, uint32_t deviceId,
                                 size_t size) {
    SWITCH_DEVICE(deviceId);
    CN_CALL(cnrtMalloc(pMemory, size));
    return INFINIRT_STATUS_SUCCESS;
}

infinirtStatus_t mallocCambriconAsync(void **pMemory, uint32_t deviceId,
                                      size_t size, infinirtStream_t stream) {
    /// @todo Cambricon does not support async malloc yet
    return mallocCambricon(pMemory, deviceId, size);
}

infinirtStatus_t mallocHostCambricon(void **pMemory, uint32_t deviceId,
                                     size_t size) {
    SWITCH_DEVICE(deviceId);
    CN_CALL(cnrtHostMalloc(pMemory, size));
    return INFINIRT_STATUS_SUCCESS;
}

infinirtStatus_t freeCambricon(void *ptr, uint32_t deviceId) {
    SWITCH_DEVICE(deviceId);
    CN_CALL(cnrtFree(ptr));
    return INFINIRT_STATUS_SUCCESS;
}

infinirtStatus_t freeCambriconAsync(void *ptr, uint32_t deviceId,
                                    infinirtStream_t stream) {
    /// @todo Cambricon does not support async free yet
    return freeCambricon(ptr, deviceId);
}

infinirtStatus_t freeHostCambricon(void *ptr, uint32_t deviceId) {
    SWITCH_DEVICE(deviceId);
    CN_CALL(cnrtFreeHost(ptr));
    return INFINIRT_STATUS_SUCCESS;
}

infinirtStatus_t memcpyHost2Cambricon(void *dst, uint32_t deviceId,
                                      const void *src, size_t size) {
    SWITCH_DEVICE(deviceId);
    CN_CALL(cnrtMemcpy(dst, (void *)src, size, cnrtMemcpyHostToDev));
    return INFINIRT_STATUS_SUCCESS;
}

infinirtStatus_t memcpyHost2CambriconAsync(void *dst, uint32_t deviceId,
                                           const void *src, size_t size,
                                           infinirtStream_t stream) {
    SWITCH_DEVICE(deviceId);
    CN_CALL(cnrtMemcpyAsync(dst, (void *)src, size, getCnrtQueue(stream), cnrtMemcpyHostToDev));
    return INFINIRT_STATUS_SUCCESS;
}
infinirtStatus_t memcpyCambricon2Host(void *dst, const void *src,
                                      uint32_t deviceId, size_t size) {
    SWITCH_DEVICE(deviceId);
    CN_CALL(cnrtMemcpy(dst, (void *)src, size, cnrtMemcpyDevToHost));
    return INFINIRT_STATUS_SUCCESS;
}

infinirtStatus_t memcpyCambricon(void *dst, const void *src, uint32_t deviceId,
                                 size_t size) {
    SWITCH_DEVICE(deviceId);
    CN_CALL(cnrtMemcpy(dst, (void *)src, size, cnrtMemcpyDevToDev));
    return INFINIRT_STATUS_SUCCESS;
}

infinirtStatus_t memcpyCambriconAsync(void *dst, const void *src,
                                      uint32_t deviceId, size_t size,
                                      infinirtStream_t stream) {
    SWITCH_DEVICE(deviceId);
    CN_CALL(cnrtMemcpyAsync(dst, (void *)src, size, getCnrtQueue(stream), cnrtMemcpyDevToDev));
    return INFINIRT_STATUS_SUCCESS;
}
