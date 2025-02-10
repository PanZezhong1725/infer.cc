#ifndef INFINIRT_MUSA_H
#define INFINIRT_MUSA_H
#include "../runtime.h"

#ifdef ENABLE_MTHREADS_GPU
#define IMPL_WITH_MUSA ;
#else
#define IMPL_WITH_MUSA {  return INFINIRT_STATUS_DEVICE_NOT_SUPPORTED; }
#endif

infinirtStatus_t synchronizeMusaDevice(uint32_t deviceId) IMPL_WITH_MUSA

infinirtStatus_t createMusaStream(infinirtStream_t *pStream, uint32_t deviceId) IMPL_WITH_MUSA
infinirtStatus_t destoryMusaStream(infinirtStream_t stream) IMPL_WITH_MUSA
infinirtStatus_t synchronizeMusaStream(infinirtStream_t stream) IMPL_WITH_MUSA

infinirtStatus_t createMusaEvent(infinirtEvent_t *pEvent, uint32_t deviceId) IMPL_WITH_MUSA
infinirtStatus_t destoryMusaEvent(infinirtEvent_t event) IMPL_WITH_MUSA
infinirtStatus_t waitMusaEvent(infinirtEvent_t event, infinirtStream_t stream) IMPL_WITH_MUSA
infinirtStatus_t recordMusaEvent(infinirtEvent_t event, infinirtStream_t stream) IMPL_WITH_MUSA
infinirtStatus_t queryMusaEvent(infinirtEvent_t event) IMPL_WITH_MUSA
infinirtStatus_t synchronizeMusaEvent(infinirtEvent_t event) IMPL_WITH_MUSA

infinirtStatus_t mallocMusa(void **pMemory, uint32_t deviceId, size_t size) IMPL_WITH_MUSA
infinirtStatus_t mallocMusaAsync(void **pMemory, uint32_t deviceId, size_t size, infinirtStream_t stream) IMPL_WITH_MUSA
infinirtStatus_t mallocHostMusa(void **pMemory, uint32_t deviceId, size_t size) IMPL_WITH_MUSA
infinirtStatus_t freeMusa(void *ptr, uint32_t deviceId) IMPL_WITH_MUSA
infinirtStatus_t freeMusaAsync(void *ptr, uint32_t deviceId, infinirtStream_t stream) IMPL_WITH_MUSA
infinirtStatus_t freeHostMusa(void *ptr, uint32_t deviceId) IMPL_WITH_MUSA
infinirtStatus_t memcpyHost2Musa(void *dst, uint32_t deviceId, const void *src, size_t size) IMPL_WITH_MUSA
infinirtStatus_t memcpyHost2MusaAsync(void *dst, uint32_t deviceId, const void *src, size_t size, infinirtStream_t stream) IMPL_WITH_MUSA
infinirtStatus_t memcpyMusa2Host(void *dst, const void *src, uint32_t deviceId, size_t size) IMPL_WITH_MUSA
infinirtStatus_t memcpyMusa(void *dst, const void *src, uint32_t deviceId, size_t size) IMPL_WITH_MUSA
infinirtStatus_t memcpyMusaAsync(void *dst, const void *src, uint32_t deviceId, size_t size, infinirtStream_t stream) IMPL_WITH_MUSA
#endif
