#ifndef INFINIRT_MACA_H
#define INFINIRT_MACA_H
#include "../runtime.h"

#ifdef ENABLE_METAX_GPU
#define IMPL_WITH_MACA ;
#else
#define IMPL_WITH_MACA {  return INFINIRT_STATUS_DEVICE_NOT_SUPPORTED; }
#endif

infinirtStatus_t synchronizeMacaDevice(uint32_t deviceId) IMPL_WITH_MACA

infinirtStatus_t createMacaStream(infinirtStream_t *pStream, uint32_t deviceId) IMPL_WITH_MACA
infinirtStatus_t destoryMacaStream(infinirtStream_t stream) IMPL_WITH_MACA
infinirtStatus_t synchronizeMacaStream(infinirtStream_t stream) IMPL_WITH_MACA

infinirtStatus_t createMacaEvent(infinirtEvent_t *pEvent, uint32_t deviceId) IMPL_WITH_MACA
infinirtStatus_t destoryMacaEvent(infinirtEvent_t event) IMPL_WITH_MACA
infinirtStatus_t waitMacaEvent(infinirtEvent_t event, infinirtStream_t stream) IMPL_WITH_MACA
infinirtStatus_t recordMacaEvent(infinirtEvent_t event, infinirtStream_t stream) IMPL_WITH_MACA
infinirtStatus_t queryMacaEvent(infinirtEvent_t event) IMPL_WITH_MACA
infinirtStatus_t synchronizeMacaEvent(infinirtEvent_t event) IMPL_WITH_MACA

infinirtStatus_t mallocMaca(void **pMemory, uint32_t deviceId, size_t size) IMPL_WITH_MACA
infinirtStatus_t mallocMacaAsync(void **pMemory, uint32_t deviceId, size_t size, infinirtStream_t stream) IMPL_WITH_MACA
infinirtStatus_t mallocHostMaca(void **pMemory, uint32_t deviceId, size_t size) IMPL_WITH_MACA
infinirtStatus_t freeMaca(void *ptr, uint32_t deviceId) IMPL_WITH_MACA
infinirtStatus_t freeMacaAsync(void *ptr, uint32_t deviceId, infinirtStream_t stream) IMPL_WITH_MACA
infinirtStatus_t freeHostMaca(void *ptr, uint32_t deviceId) IMPL_WITH_MACA
infinirtStatus_t memcpyHost2Maca(void *dst, uint32_t deviceId, const void *src, size_t size) IMPL_WITH_MACA
infinirtStatus_t memcpyHost2MacaAsync(void *dst, uint32_t deviceId, const void *src, size_t size, infinirtStream_t stream) IMPL_WITH_MACA
infinirtStatus_t memcpyMaca2Host(void *dst, const void *src, uint32_t deviceId, size_t size) IMPL_WITH_MACA
infinirtStatus_t memcpyMaca(void *dst, const void *src, uint32_t deviceId, size_t size) IMPL_WITH_MACA
infinirtStatus_t memcpyMacaAsync(void *dst, const void *src, uint32_t deviceId, size_t size, infinirtStream_t stream) IMPL_WITH_MACA
#endif
