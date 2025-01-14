#include "test.h"
#include <iostream>

int main() {
#ifdef ENABLE_NV_GPU
    printf("Test tensor functions: Nvidia\n");
    test_tensor(DEVICE_NVIDIA);
#ifdef ENABLE_CCL
    printf("Test CCL functions: Nvidia\n");
    test_ccl(DEVICE_NVIDIA);
#endif
#endif

#ifdef ENABLE_ASCEND_NPU
    printf("Test tensor functions: Ascend\n");
    test_tensor(DEVICE_ASCEND);
#ifdef ENABLE_CCL
    printf("Test CCL functions: Ascend\n");
    test_ccl(DEVICE_ASCEND);
#endif
#endif

#ifdef ENABLE_KUNLUN_XPU
    printf("Test tensor fuctions: Kunlun\n");
    test_tensor(DEVICE_KUNLUN);
#ifdef ENABLE_CCL
    printf("Test CCL functions: Kunlun\n");
    test_ccl(DEVICE_KUNLUN);
#endif
#endif

#ifdef ENABLE_CAMBRICON_MLU
    printf("Test tensor functions: Cambricon\n");
    test_tensor(DEVICE_CAMBRICON);
#ifdef ENABLE_CCL
    printf("Test CCL functions: Cambricon\n");
    test_ccl(DEVICE_CAMBRICON);
#endif
#endif
}
