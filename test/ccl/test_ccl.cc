#include "../../include/infinirt.h"
#include "../../include/infiniccl.h"
#include "../../src/tensor.h"
#include "../test.h"
#include <thread>
#include <vector>

#define TEST_GROUP_SIZE 2

#define CHECK_RUN(EXPR)                                                        \
    do {                                                                       \
        int code = static_cast<int>(EXPR);                                     \
        if (code != 0) {                                                       \
            printf("Error at %s:%d with code %d\n", __FILE__, __LINE__, code); \
            return TEST_FAILED;                                                \
        }                                                                      \
    } while (0)

int allreduce_sum(DeviceType deviceType, uint32_t deviceID, infinicclComm_t comm,
                   std::vector<float> &data,
                   std::vector<std::vector<float>> &output) {
    auto result = std::vector<float>(data.size(), 0.0);
    auto send_buf =
        Tensor::weight(data.data(), INFINI_F32,
                       std::vector<index_t>({data.size()}), deviceType, deviceID);
    auto recv_buf =
        Tensor::weight(result.data(), INFINI_F32,
                       std::vector<index_t>({result.size()}), deviceType, deviceID);
    infinirtStream_t stream;
    CHECK_RUN(infinirtStreamCreate(&stream, deviceType, deviceID));
    CHECK_RUN(infinicclAllReduceSum(comm, send_buf->data(), recv_buf->data(),
                                    data.size(), INFINI_F32,
                                    stream));
    infinirtStreamSynchronize(stream);
    CHECK_RUN(infinirtMemcpyD2H(result.data(), recv_buf->data(), deviceType,
                                deviceID, recv_buf->byte_size()));
    output[deviceID] = std::move(result);
    return TEST_PASSED;
}

int test_allreduce_sum(DeviceType deviceType) {
    auto data = std::vector<float>{1.0, 2.0};
    auto len = data.size();
    auto ans = std::vector<float>(len);
    for (int i = 0; i < len; i++) {
        ans[i] = data[i] * TEST_GROUP_SIZE;
    }

    auto output = std::vector<std::vector<float>>(len);
    auto threads = std::vector<std::thread>(TEST_GROUP_SIZE);

    infinicclComm_t comm[TEST_GROUP_SIZE];
    auto deviceIds = std::vector<uint32_t>(TEST_GROUP_SIZE);
    for (int i = 0; i < TEST_GROUP_SIZE; i++) {
        deviceIds[i] = (uint32_t)i;
    }


    CHECK_RUN(infinicclCommInitAll(deviceType, comm, TEST_GROUP_SIZE, deviceIds.data()));

    for (int i = 0; i < TEST_GROUP_SIZE; i++) {
        threads[i] = std::thread(allreduce_sum, deviceType, deviceIds[i], comm[i], std::ref(data),
                                 std::ref(output));
    }
    for (int i = 0; i < TEST_GROUP_SIZE; i++) {
        threads[i].join();
    }

    for (int i = 0; i < TEST_GROUP_SIZE; i++) {
        TEST_EQUAL(output[i], ans);
    }

    return TEST_PASSED;
}

void test_ccl(DeviceType deviceType) {
    RUN_TEST(test_allreduce_sum(deviceType));
}
