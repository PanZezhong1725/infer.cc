#include "runtime.h"
#include "ascend/infinirt_ascend.h"
#include "cuda/infinirt_cuda.h"
#include "kunlun/infinirt_kunlun.h"
#include "cambricon/infinirt_cambricon.h"
#include "maca/infinirt_maca.h"
#include <cstdlib>
#include <string.h>

__C __export infinirtStatus_t infinirtInit(DeviceType device){
    switch (device){
        case DEVICE_CPU:
            return INFINIRT_STATUS_SUCCESS;
        case DEVICE_NVIDIA:
            return INFINIRT_STATUS_SUCCESS;
        case DEVICE_CAMBRICON:
            return INFINIRT_STATUS_SUCCESS;
        case DEVICE_ASCEND:
            return initAscend();
        case DEVICE_KUNLUN:
            return INFINIRT_STATUS_SUCCESS;
        case DEVICE_METAX:
            return INFINIRT_STATUS_SUCCESS;
        default:
            return INFINIRT_STATUS_DEVICE_NOT_SUPPORTED;
    }
}

// Device
__C infinirtStatus_t infinirtDeviceSynchronize(DeviceType device, uint32_t deviceId){
    switch (device)
    {
    case DEVICE_CPU:
        return INFINIRT_STATUS_SUCCESS;
    case DEVICE_NVIDIA:
        return synchronizeCudaDevice(deviceId);
    case DEVICE_CAMBRICON:
        return synchronizeCambriconDevice(deviceId);
    case DEVICE_ASCEND:
        return synchronizeAscendDevice(deviceId);
    case DEVICE_KUNLUN:
        return INFINIRT_STATUS_SUCCESS;
    case DEVICE_METAX:
        return synchronizeMacaDevice(deviceId);
    default:
        return INFINIRT_STATUS_DEVICE_NOT_SUPPORTED;
    }
}

// Stream
__C infinirtStatus_t infinirtStreamCreate(infinirtStream_t *pStream, DeviceType device, uint32_t deviceId)
{
    switch (device)
    {
    case DEVICE_CPU:
    {
        auto stream = new infinirtStream();
        stream->device = DEVICE_CPU;
        stream->device_id = 0;
        stream->stream = nullptr;
        (*pStream) = stream;
        return INFINIRT_STATUS_SUCCESS;
    }
        
    case DEVICE_NVIDIA:
        return createCudaStream(pStream, deviceId);
    case DEVICE_CAMBRICON:
        return createCambriconStream(pStream, deviceId);
    case DEVICE_ASCEND:
        return createAscendStream(pStream, deviceId);
    case DEVICE_KUNLUN:
        return createKunlunStream(pStream, deviceId);
    case DEVICE_METAX:
        return createMacaStream(pStream, deviceId);
    default:
        return INFINIRT_STATUS_DEVICE_NOT_SUPPORTED;
    }
}
__C infinirtStatus_t infinirtStreamDestroy(infinirtStream_t stream)
{
    if (stream == nullptr)
        return INFINIRT_STATUS_SUCCESS;
    switch (stream->device)
    {
    case DEVICE_CPU:
        delete stream;
        return INFINIRT_STATUS_SUCCESS;
    case DEVICE_NVIDIA:
        return destoryCudaStream(stream);
    case DEVICE_CAMBRICON:
        return destoryCambriconStream(stream);
    case DEVICE_ASCEND:
        return destoryAscendStream(stream);
    case DEVICE_KUNLUN:
        return destoryKunlunStream(stream);
    case DEVICE_METAX:
        return destoryMacaStream(stream);
    default:
        return INFINIRT_STATUS_DEVICE_NOT_SUPPORTED;
    }
}

__C infinirtStatus_t infinirtStreamSynchronize(infinirtStream_t stream){
    if (stream == nullptr)
        return INFINIRT_STATUS_SUCCESS;
    switch (stream->device)
    {
    case DEVICE_CPU:
        return INFINIRT_STATUS_SUCCESS;
    case DEVICE_NVIDIA:
        return synchronizeCudaStream(stream);
    case DEVICE_CAMBRICON:
        return synchronizeCambriconStream(stream);
    case DEVICE_ASCEND:
        return synchronizeAscendStream(stream);
    case DEVICE_KUNLUN:
        return synchronizeKunlunStream(stream);
    case DEVICE_METAX:
        return synchronizeMacaStream(stream);
    default:
        return INFINIRT_STATUS_DEVICE_NOT_SUPPORTED;
    }
}

__C infinirtStatus_t infinirtGetRawStream(void **ptr, infinirtStream_t stream) {
    if (stream == nullptr)
        return INFINIRT_STATUS_INVALID_ARGUMENT;
    *ptr = stream->stream;
    return INFINIRT_STATUS_SUCCESS;
}

__C infinirtStatus_t infinirtGetStreamDeviceInfo(DeviceType *deviceType,
                                                 uint32_t *deviceId,
                                                 infinirtStream_t stream) {
    if (stream == nullptr)
        return INFINIRT_STATUS_INVALID_ARGUMENT;
    if (deviceType != nullptr) {
        *deviceType = stream->device;
    }
    if (deviceId != nullptr) {
        *deviceId = stream->device_id;
    }
    return INFINIRT_STATUS_SUCCESS;
}

// Event
__C infinirtStatus_t infinirtEventCreate(infinirtEvent_t *pEvent, DeviceType device, uint32_t deviceId)
{
    switch (device)
    {
    case DEVICE_CPU:
    {
        auto event = new infinirtEvent();
        event->device = DEVICE_CPU;
        event->device_id = 0;
        event->event = nullptr;
        (*pEvent) = event;
        return INFINIRT_STATUS_SUCCESS;
    }
        
    case DEVICE_NVIDIA:
        return createCudaEvent(pEvent, deviceId);
    case DEVICE_CAMBRICON:
        return createCambriconEvent(pEvent, deviceId);
    case DEVICE_ASCEND:
        return createAscendEvent(pEvent, deviceId);
    case DEVICE_KUNLUN:
        return createKunlunEvent(pEvent, deviceId);
    case DEVICE_METAX:
        return createMacaEvent(pEvent, deviceId);
    default:
        return INFINIRT_STATUS_DEVICE_NOT_SUPPORTED;
    }
}
__C infinirtStatus_t infinirtEventRecord(infinirtEvent_t event,
                                         infinirtStream_t stream) {
    if (event == nullptr)
        return INFINIRT_STATUS_INVALID_ARGUMENT;
    if (stream != nullptr && event->device != stream->device)
        return INFINIRT_STATUS_DEVICE_MISMATCH;
    switch (event->device) {
    case DEVICE_CPU:
        return INFINIRT_STATUS_SUCCESS;
    case DEVICE_NVIDIA:
        return recordCudaEvent(event, stream);
    case DEVICE_CAMBRICON:
        return recordCambriconEvent(event, stream);
    case DEVICE_ASCEND:
        return recordAscendEvent(event, stream);
    case DEVICE_KUNLUN:
        return recordKunlunEvent(event, stream);
    case DEVICE_METAX:
        return recordMacaEvent(event, stream);
    default:
        return INFINIRT_STATUS_DEVICE_NOT_SUPPORTED;
    }
}
__C infinirtStatus_t infinirtEventQuery(infinirtEvent_t event) {
    if (event == nullptr)
        return INFINIRT_STATUS_INVALID_ARGUMENT;
    switch (event->device) {
    case DEVICE_CPU:
        return INFINIRT_STATUS_SUCCESS;
    case DEVICE_NVIDIA:
        return queryCudaEvent(event);
    case DEVICE_CAMBRICON:
        return queryCambriconEvent(event);
    case DEVICE_ASCEND:
        return queryAscendEvent(event);
    case DEVICE_KUNLUN:
        return INFINIRT_STATUS_SUCCESS;
    case DEVICE_METAX:
        return queryMacaEvent(event);
    default:
        return INFINIRT_STATUS_DEVICE_NOT_SUPPORTED;
    }
}
__C infinirtStatus_t infinirtEventSynchronize(infinirtEvent_t event) {
    if (event == nullptr)
        return INFINIRT_STATUS_INVALID_ARGUMENT;
    switch (event->device) {
    case DEVICE_CPU:
        return INFINIRT_STATUS_SUCCESS;
    case DEVICE_NVIDIA:
        return synchronizeCudaEvent(event);
    case DEVICE_CAMBRICON:
        return synchronizeCambriconEvent(event);
    case DEVICE_ASCEND:
        return synchronizeAscendEvent(event);
    case DEVICE_KUNLUN:
        return synchronizeKunlunEvent(event);
    case DEVICE_METAX:
        return synchronizeMacaEvent(event);
    default:
        return INFINIRT_STATUS_DEVICE_NOT_SUPPORTED;
    }
}
__C infinirtStatus_t infinirtEventDestroy(infinirtEvent_t event)
{
    if (event == nullptr)
        return INFINIRT_STATUS_SUCCESS;
    switch (event->device)
    {
    case DEVICE_CPU:
        delete event;
        return INFINIRT_STATUS_SUCCESS;
    case DEVICE_NVIDIA:
        return destoryCudaEvent(event);
    case DEVICE_CAMBRICON:
        return destoryCambriconEvent(event);
    case DEVICE_ASCEND:
        return destoryAscendEvent(event);
    case DEVICE_KUNLUN:
        return destoryKunlunEvent(event);
    case DEVICE_METAX:
        return destoryMacaEvent(event);
    default:
        return INFINIRT_STATUS_DEVICE_NOT_SUPPORTED;
    }
}
__C infinirtStatus_t infinirtStreamWaitEvent(infinirtEvent_t event, infinirtStream_t stream)
{
    if (event == nullptr)
        return INFINIRT_STATUS_INVALID_ARGUMENT;
    if (stream != nullptr && (event->device != stream->device ||
                              stream->device_id != event->device_id))
        return INFINIRT_STATUS_DEVICE_MISMATCH;
    switch (event->device)
    {
    case DEVICE_CPU:
        return INFINIRT_STATUS_SUCCESS;
    case DEVICE_NVIDIA:
        return waitCudaEvent(event, stream);
    case DEVICE_CAMBRICON:
        return waitCambriconEvent(event, stream);
    case DEVICE_ASCEND:
        return waitAscendEvent(event, stream);
    case DEVICE_KUNLUN:
        return waitKunlunEvent(event, stream);
    case DEVICE_METAX:
        return waitMacaEvent(event, stream);
    default:
        return INFINIRT_STATUS_DEVICE_NOT_SUPPORTED;
    }
}

// Memory
__C infinirtStatus_t infinirtMalloc(void **pMemory, DeviceType device,
                                    uint32_t deviceId, size_t size) {
    switch (device)
    {
    case DEVICE_CPU:
        *pMemory = std::malloc(size);
        return INFINIRT_STATUS_SUCCESS;
    case DEVICE_NVIDIA:
        return mallocCuda(pMemory, deviceId, size);
    case DEVICE_CAMBRICON:
        return mallocCambricon(pMemory, deviceId, size);
    case DEVICE_ASCEND:
        return mallocAscend(pMemory, deviceId, size);
    case DEVICE_KUNLUN:
        return mallocKunlun(pMemory, deviceId, size);
    case DEVICE_METAX:
        return mallocMaca(pMemory, deviceId, size);
    default:
        return INFINIRT_STATUS_DEVICE_NOT_SUPPORTED;
    }
}

__C infinirtStatus_t infinirtMallocAsync(void **pMemory, DeviceType device,
                                         uint32_t deviceId, size_t size,
                                         infinirtStream_t stream) {
    if (stream != nullptr &&
        (device != stream->device || deviceId != stream->device_id))
        return INFINIRT_STATUS_DEVICE_MISMATCH;
    switch (device)
    {
    case DEVICE_CPU:
        return infinirtMalloc(pMemory, device, deviceId, size);
    case DEVICE_NVIDIA:
        return mallocCudaAsync(pMemory, deviceId, size, stream);
    case DEVICE_CAMBRICON:
        return mallocCambriconAsync(pMemory, deviceId, size, stream);
    case DEVICE_ASCEND:
        return mallocAscendAsync(pMemory, deviceId, size, stream);
    case DEVICE_KUNLUN:
        return mallocKunlunAsync(pMemory, deviceId, size, stream);
    case DEVICE_METAX:
        return mallocMacaAsync(pMemory, deviceId, size, stream);
    default:
        return INFINIRT_STATUS_DEVICE_NOT_SUPPORTED;
    }
}

__C __export infinirtStatus_t infinirtMallocHost(void **pMemory,
                                                 DeviceType device,
                                                 uint32_t deviceId,
                                                 size_t size) {
    switch (device) {
    case DEVICE_CPU:
        return infinirtMalloc(pMemory, device, deviceId, size);
    case DEVICE_NVIDIA:
        return mallocHostCuda(pMemory, deviceId, size);
    case DEVICE_CAMBRICON:
        return mallocHostCambricon(pMemory, deviceId, size);
    case DEVICE_ASCEND:
        return mallocHostAscend(pMemory, deviceId, size);
    case DEVICE_KUNLUN:
        return mallocHostKunlun(pMemory, deviceId, size);
    case DEVICE_METAX:
        return mallocHostMaca(pMemory, deviceId, size);
    default:
        return INFINIRT_STATUS_DEVICE_NOT_SUPPORTED;
    }
}

__C infinirtStatus_t infinirtFree(void *ptr, DeviceType device,
                                  uint32_t deviceId) {
    if (ptr == nullptr)
        return INFINIRT_STATUS_SUCCESS;
    switch (device) {
    case DEVICE_CPU:
        std::free(ptr);
        return INFINIRT_STATUS_SUCCESS;
    case DEVICE_NVIDIA:
        return freeCuda(ptr, deviceId);
    case DEVICE_CAMBRICON:
        return freeCambricon(ptr, deviceId);
    case DEVICE_ASCEND:
        return freeAscend(ptr, deviceId);
    case DEVICE_KUNLUN:
        return freeKunlun(ptr, deviceId);
    case DEVICE_METAX:
        return freeMaca(ptr, deviceId);
    default:
        return INFINIRT_STATUS_DEVICE_NOT_SUPPORTED;
    }
}

__C infinirtStatus_t infinirtFreeAsync(void *ptr, DeviceType device,
                                       uint32_t deviceId,
                                       infinirtStream_t stream) {
    if (ptr == nullptr)
        return INFINIRT_STATUS_SUCCESS;
    if (stream == nullptr) {
        return infinirtFree(ptr, device, deviceId);
    } else if (device != stream->device || deviceId != stream->device_id) {
        return INFINIRT_STATUS_DEVICE_MISMATCH;
    }
    switch (device) {
    case DEVICE_CPU:
        return infinirtFree(ptr, device, deviceId);
    case DEVICE_NVIDIA:
        return freeCudaAsync(ptr, deviceId, stream);
    case DEVICE_CAMBRICON:
        return freeCambriconAsync(ptr, deviceId, stream);
    case DEVICE_ASCEND:
        return freeAscendAsync(ptr, deviceId, stream);
    case DEVICE_KUNLUN:
        return freeKunlunAsync(ptr, deviceId, stream);
    case DEVICE_METAX:
        return freeMacaAsync(ptr, deviceId, stream);
    default:
        return INFINIRT_STATUS_DEVICE_NOT_SUPPORTED;
    }
}

__C __export infinirtStatus_t infinirtFreeHost(void *ptr, DeviceType device,
                                               uint32_t deviceId) {
    if (ptr == nullptr)
        return INFINIRT_STATUS_SUCCESS;
    switch (device) {
    case DEVICE_CPU:
        return infinirtFree(ptr, device, deviceId);
    case DEVICE_NVIDIA:
        return freeHostCuda(ptr, deviceId);
    case DEVICE_CAMBRICON:
        return freeHostCambricon(ptr, deviceId);
    case DEVICE_ASCEND:
        return freeHostAscend(ptr, deviceId);
    case DEVICE_KUNLUN:
        return freeHostKunlun(ptr, deviceId);
    case DEVICE_METAX:
        return freeHostMaca(ptr, deviceId);
    default:
        return INFINIRT_STATUS_DEVICE_NOT_SUPPORTED;
    }
}

__C infinirtStatus_t infinirtMemcpyH2D(void *dst, DeviceType device,
                                            uint32_t deviceId, const void *src,
                                            size_t size) {
    if (dst == nullptr || src == nullptr)
        return INFINIRT_STATUS_INVALID_ARGUMENT;

    switch (device) {
    case DEVICE_CPU:
        memcpy(dst, src, size);
        return INFINIRT_STATUS_SUCCESS;
    case DEVICE_NVIDIA:
        return memcpyHost2Cuda(dst, deviceId, src, size);
    case DEVICE_CAMBRICON:
        return memcpyHost2Cambricon(dst, deviceId, src, size);
    case DEVICE_ASCEND:
        return memcpyHost2Ascend(dst, deviceId, src, size);
    case DEVICE_KUNLUN:
        return memcpyHost2Kunlun(dst, deviceId, src, size);
    case DEVICE_METAX:
        return memcpyHost2Maca(dst, deviceId, src, size);
    default:
        return INFINIRT_STATUS_DEVICE_NOT_SUPPORTED;
    }
}

__C infinirtStatus_t infinirtMemcpyH2DAsync(void *dst, DeviceType device,
                                            uint32_t deviceId, const void *src,
                                            size_t size,
                                            infinirtStream_t stream) {
    if (dst == nullptr || src == nullptr)
        return INFINIRT_STATUS_INVALID_ARGUMENT;
    if (stream != nullptr &&
        (device != stream->device || deviceId != stream->device_id))
        return INFINIRT_STATUS_DEVICE_MISMATCH;

    switch (device) {
    case DEVICE_CPU:
        return infinirtMemcpyH2D(dst, device, deviceId, src, size);
    case DEVICE_NVIDIA:
        return memcpyHost2CudaAsync(dst, deviceId, src, size, stream);
    case DEVICE_CAMBRICON:
        return memcpyHost2CambriconAsync(dst, deviceId, src, size, stream);
    case DEVICE_ASCEND:
        return memcpyHost2AscendAsync(dst, deviceId, src, size, stream);
    case DEVICE_KUNLUN:
        return memcpyHost2KunlunAsync(dst, deviceId, src, size, stream);
    case DEVICE_METAX:
        return memcpyHost2MacaAsync(dst, deviceId, src, size, stream);
    default:
        return INFINIRT_STATUS_DEVICE_NOT_SUPPORTED;
    }
}

__C infinirtStatus_t infinirtMemcpyD2H(void *dst, const void *src,
                                       DeviceType device, uint32_t deviceId,
                                       size_t size) {
    if (src == nullptr || dst == nullptr)
        return INFINIRT_STATUS_INVALID_ARGUMENT;

    switch (device) {
    case DEVICE_CPU:
        memcpy(dst, src, size);
        return INFINIRT_STATUS_SUCCESS;
    case DEVICE_NVIDIA:
        return memcpyCuda2Host(dst, src, deviceId, size);
    case DEVICE_CAMBRICON:
        return memcpyCambricon2Host(dst, src, deviceId, size);
    case DEVICE_ASCEND:
        return memcpyAscend2Host(dst, src, deviceId, size);
    case DEVICE_KUNLUN:
        return memcpyKunlun2Host(dst, src, deviceId, size);
    case DEVICE_METAX:
        return memcpyMaca2Host(dst, src, deviceId, size);
    default:
        return INFINIRT_STATUS_DEVICE_NOT_SUPPORTED;
    }
}

__C __export infinirtStatus_t infinirtMemcpy(void *dst, const void *src,
                                             DeviceType device,
                                             uint32_t deviceId, size_t size) {
    if (size == 0)
        return INFINIRT_STATUS_SUCCESS;
    if (dst == nullptr || src == nullptr)
        return INFINIRT_STATUS_INVALID_ARGUMENT;
    switch (device) {
    case DEVICE_CPU:
        memcpy(dst, src, size);
        return INFINIRT_STATUS_SUCCESS;
    case DEVICE_NVIDIA:
        return memcpyCuda(dst, src, deviceId, size);
    case DEVICE_CAMBRICON:
        return memcpyCambricon(dst, src, deviceId, size);
    case DEVICE_ASCEND:
        return memcpyAscend(dst, src, deviceId, size);
    case DEVICE_KUNLUN:
        return memcpyKunlun(dst, src, deviceId, size);
    case DEVICE_METAX:
        return memcpyMaca(dst, src, deviceId, size);
    default:
        return INFINIRT_STATUS_DEVICE_NOT_SUPPORTED;
    }
}

__C __export infinirtStatus_t infinirtMemcpyAsync(void *dst, const void *src,
                                                  DeviceType device,
                                                  uint32_t deviceId,
                                                  size_t size,
                                                  infinirtStream_t stream) {
    if (size == 0)
        return INFINIRT_STATUS_SUCCESS;
    if (dst == nullptr || src == nullptr)
        return INFINIRT_STATUS_INVALID_ARGUMENT;
    if (stream != nullptr &&
        (device != stream->device || deviceId != stream->device_id))
        return INFINIRT_STATUS_DEVICE_MISMATCH;

    switch (device) {
    case DEVICE_CPU:
        return infinirtMemcpy(dst, src, device, deviceId, size);
    case DEVICE_NVIDIA:
        return memcpyCudaAsync(dst, src, deviceId, size, stream);
    case DEVICE_CAMBRICON:
        return memcpyCambriconAsync(dst, src, deviceId, size, stream);
    case DEVICE_ASCEND:
        return memcpyAscendAsync(dst, src, deviceId, size, stream);
    case DEVICE_KUNLUN:
        return memcpyKunlunAsync(dst, src, deviceId, size, stream);
    case DEVICE_METAX:
        return memcpyMacaAsync(dst, src, deviceId, size, stream);
    default:
        return INFINIRT_STATUS_DEVICE_NOT_SUPPORTED;
    }
}
