#ifndef INFINI_RUNTIME_H
#define INFINI_RUNTIME_H

#if defined(_WIN32)
#define __export __declspec(dllexport)
#elif defined(__GNUC__) && ((__GNUC__ >= 4) || (__GNUC__ == 3 && __GNUC_MINOR__ >= 3))
#define __export __attribute__((visibility("default")))
#else
#define __export
#endif

#ifdef __cplusplus
#define __C extern "C"
#else
#define __C
#endif
#include <stddef.h>
#include <stdint.h>

typedef enum
{
    DEVICE_CPU = 0,
    DEVICE_NVIDIA = 1,
    DEVICE_CAMBRICON = 2,
    DEVICE_ASCEND = 3,
    DEVICE_METAX = 4,
    DEVICE_MTHREADS = 5,
    DEVICE_KUNLUN = 7
} DeviceType;

typedef enum
{
    INFINIRT_STATUS_SUCCESS = 0,
    INFINIRT_STATUS_EXECUTION_FAILED = 1,
    INFINIRT_STATUS_BAD_DEVICE = 2,
    INFINIRT_STATUS_DEVICE_NOT_SUPPORTED = 3,
    INFINIRT_STATUS_DEVICE_MISMATCH = 4,
    INFINIRT_STATUS_INVALID_ARGUMENT = 5,
    INFINIRT_STATUS_ILLEGAL_MEMORY_ACCESS = 6,
    INFINIRT_STATUS_NOT_READY = 7,
} infinirtStatus_t;

__C __export infinirtStatus_t infinirtInit(DeviceType device);

// Device
__C __export infinirtStatus_t infinirtDeviceSynchronize(DeviceType device, uint32_t deviceId);

// Stream
struct infinirtStream;
typedef struct infinirtStream *infinirtStream_t;
#define INFINIRT_NULL_STREAM nullptr
__C __export infinirtStatus_t infinirtStreamCreate(infinirtStream_t *pStream, DeviceType device, uint32_t deviceId);
__C __export infinirtStatus_t infinirtStreamDestroy(infinirtStream_t stream);
__C __export infinirtStatus_t infinirtStreamSynchronize(infinirtStream_t stream);
__C __export infinirtStatus_t infinirtGetRawStream(void** ptr, infinirtStream_t stream);
__C __export infinirtStatus_t infinirtGetStreamDeviceInfo(DeviceType* deviceType, uint32_t *deviceId, infinirtStream_t stream);

// Event
struct infinirtEvent;
typedef struct infinirtEvent *infinirtEvent_t;
__C __export infinirtStatus_t infinirtEventCreate(infinirtEvent_t *pEvent, DeviceType device, uint32_t deviceId);
__C __export infinirtStatus_t infinirtEventRecord(infinirtEvent_t event, infinirtStream_t stream);
__C __export infinirtStatus_t infinirtEventQuery(infinirtEvent_t event);
__C __export infinirtStatus_t infinirtEventSynchronize(infinirtEvent_t event);
__C __export infinirtStatus_t infinirtEventDestroy(infinirtEvent_t event);
__C __export infinirtStatus_t infinirtStreamWaitEvent(infinirtEvent_t event, infinirtStream_t stream);

// Memory
__C __export infinirtStatus_t infinirtMalloc(void **pMemory, DeviceType device, uint32_t deviceId, size_t size);
__C __export infinirtStatus_t infinirtMallocAsync(void **pMemory, DeviceType device, uint32_t deviceId, size_t size, infinirtStream_t stream);
__C __export infinirtStatus_t infinirtMallocHost(void **pMemory, DeviceType device, uint32_t deviceId, size_t size);
__C __export infinirtStatus_t infinirtFree(void *ptr, DeviceType device, uint32_t deviceId);
__C __export infinirtStatus_t infinirtFreeAsync(void *ptr, DeviceType device, uint32_t deviceId, infinirtStream_t stream);
__C __export infinirtStatus_t infinirtFreeHost(void *ptr, DeviceType device, uint32_t deviceId);
__C __export infinirtStatus_t infinirtMemcpyH2D(void *dst, DeviceType device, uint32_t deviceId, const void *src, size_t size);
__C __export infinirtStatus_t infinirtMemcpyH2DAsync(void *dst, DeviceType device, uint32_t deviceId, const void *src, size_t size, infinirtStream_t stream);
__C __export infinirtStatus_t infinirtMemcpyD2H(void *dst, const void* src, DeviceType device, uint32_t deviceId, size_t size);
__C __export infinirtStatus_t infinirtMemcpy(void *dst, const void* src, DeviceType device, uint32_t deviceId, size_t size);
__C __export infinirtStatus_t infinirtMemcpyAsync(void *dst, const void* src, DeviceType device, uint32_t deviceId, size_t size, infinirtStream_t stream);
#endif
