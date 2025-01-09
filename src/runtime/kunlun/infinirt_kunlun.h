#ifndef INFINIRT_KUNLUN_H
#define INFINIRT_KUNLUN_H
#include "../runtime.h"

#ifdef ENABLE_KUNLUN_XPU
#define IMPL_WITH_KUNLUN ;
#else
#define IMPL_WITH_KUNLUN { return INFINIRT_STATUS_DEVICE_NOT_SUPPORTED; }
#endif

infinirtStatus_t createKunlunStream(infinirtStream_t *pStream, uint32_t deviceId) IMPL_WITH_KUNLUN
infinirtStatus_t destoryKunlunStream(infinirtStream_t stream) IMPL_WITH_KUNLUN
infinirtStatus_t synchronizeKunlunStream(infinirtStream_t stream) IMPL_WITH_KUNLUN

infinirtStatus_t createKunlunEvent(infinirtEvent_t *pEvent, uint32_t deviceId) IMPL_WITH_KUNLUN
infinirtStatus_t destoryKunlunEvent(infinirtEvent_t event) IMPL_WITH_KUNLUN
infinirtStatus_t waitKunlunEvent(infinirtEvent_t event, infinirtStream_t stream) IMPL_WITH_KUNLUN
infinirtStatus_t recordKunlunEvent(infinirtEvent_t event, infinirtStream_t stream) IMPL_WITH_KUNLUN
infinirtStatus_t synchronizeKunlunEvent(infinirtEvent_t event) IMPL_WITH_KUNLUN

infinirtStatus_t mallocKunlun(void **pMemory, uint32_t deviceId, size_t size) IMPL_WITH_KUNLUN
infinirtStatus_t mallocHostKunlun(void **pMemory, uint32_t deviceId, size_t size) IMPL_WITH_KUNLUN
infinirtStatus_t freeKunlun(void *ptr, uint32_t deviceId) IMPL_WITH_KUNLUN
infinirtStatus_t freeHostKunlun(void *ptr, uint32_t deviceId) IMPL_WITH_KUNLUN
infinirtStatus_t memcpyHost2Kunlun(void *dst, uint32_t deviceId, const void *src, size_t size) IMPL_WITH_KUNLUN
infinirtStatus_t memcpyKunlun2Host(void *dst, const void *src, uint32_t deviceId, size_t size) IMPL_WITH_KUNLUN
infinirtStatus_t memcpyKunlun(void *dst, const void *src, uint32_t deviceId, size_t size) IMPL_WITH_KUNLUN

#endif
