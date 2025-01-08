#ifndef INFINIRT_CAMBRICON_H
#define INFINIRT_CAMBRICON_H
#include "../runtime.h"

#ifdef ENABLE_CAMBRICON_MLU
#define IMPL_WITH_CAMBRICON ;
#else
#define IMPL_WITH_CAMBRICON {  return INFINIRT_STATUS_DEVICE_NOT_SUPPORTED; }
#endif

infinirtStatus_t initCambricon() IMPL_WITH_CAMBRICON

infinirtStatus_t synchronizeCambriconDevice(uint32_t deviceId) IMPL_WITH_CAMBRICON

infinirtStatus_t createCambriconStream(infinirtStream_t *pStream, uint32_t deviceId) IMPL_WITH_CAMBRICON
infinirtStatus_t destoryCambriconStream(infinirtStream_t stream) IMPL_WITH_CAMBRICON
infinirtStatus_t synchronizeCambriconStream(infinirtStream_t stream) IMPL_WITH_CAMBRICON

infinirtStatus_t createCambriconEvent(infinirtEvent_t *pEvent, uint32_t deviceId) IMPL_WITH_CAMBRICON
infinirtStatus_t destoryCambriconEvent(infinirtEvent_t event) IMPL_WITH_CAMBRICON
infinirtStatus_t waitCambriconEvent(infinirtEvent_t event, infinirtStream_t stream) IMPL_WITH_CAMBRICON
infinirtStatus_t recordCambriconEvent(infinirtEvent_t event, infinirtStream_t stream) IMPL_WITH_CAMBRICON
infinirtStatus_t queryCambriconEvent(infinirtEvent_t event) IMPL_WITH_CAMBRICON
infinirtStatus_t synchronizeCambriconEvent(infinirtEvent_t event) IMPL_WITH_CAMBRICON

infinirtStatus_t mallocCambricon(void **pMemory, uint32_t deviceId, size_t size) IMPL_WITH_CAMBRICON
infinirtStatus_t mallocCambriconAsync(void **pMemory, uint32_t deviceId, size_t size, infinirtStream_t stream) IMPL_WITH_CAMBRICON
infinirtStatus_t mallocHostCambricon(void **pMemory, uint32_t deviceId, size_t size) IMPL_WITH_CAMBRICON
infinirtStatus_t freeCambricon(void *ptr, uint32_t deviceId) IMPL_WITH_CAMBRICON
infinirtStatus_t freeCambriconAsync(void *ptr, uint32_t deviceId, infinirtStream_t stream) IMPL_WITH_CAMBRICON
infinirtStatus_t freeHostCambricon(void *ptr, uint32_t deviceId) IMPL_WITH_CAMBRICON
infinirtStatus_t memcpyHost2Cambricon(void *dst, uint32_t deviceId, const void *src, size_t size) IMPL_WITH_CAMBRICON
infinirtStatus_t memcpyHost2CambriconAsync(void *dst, uint32_t deviceId, const void *src, size_t size, infinirtStream_t stream) IMPL_WITH_CAMBRICON
infinirtStatus_t memcpyCambricon2Host(void *dst, const void *src, uint32_t deviceId, size_t size) IMPL_WITH_CAMBRICON
infinirtStatus_t memcpyCambricon(void *dst, const void *src, uint32_t deviceId, size_t size) IMPL_WITH_CAMBRICON
infinirtStatus_t memcpyCambriconAsync(void *dst, const void *src, uint32_t deviceId, size_t size, infinirtStream_t stream) IMPL_WITH_CAMBRICON

#endif
