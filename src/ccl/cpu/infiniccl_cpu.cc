#include "infiniccl_cpu.h"

inline MPI_Datatype getCpuDtype(InfiniDataType_t datatype) {
    MPI_Datatype MPI_HALF;
    MPI_Type_contiguous(2, MPI_BYTE, &MPI_HALF);
    MPI_Type_commit(&MPI_HALF);
    switch (datatype) {
    case INFINI_F32:
        return MPI_FLOAT;
    case INFINI_F16:
        return MPI_HALF;
    default:
        return MPI_HALF;
    }
}

infinicclStatus_t infinicclCpuCommInitAll(infinicclComm_t *comms) {
    MPI_Init(NULL, NULL);
    comms[0] = new InfiniComm{DEVICE_CPU, (unsigned int)0, (void*) MPI_COMM_WORLD};
    return INFINICCL_STATUS_SUCCESS;
}

infinicclStatus_t infinicclCpuCommDestroy(infinicclComm_t comm) {
    delete comm;
    return INFINICCL_STATUS_SUCCESS;
}

infinicclStatus_t infinicclCpuAllReduceSum(infinicclComm_t comms, void *sendbuf,
                              void *recvbuf, size_t count,
                              InfiniDataType_t datatype) {
    if (datatype != INFINI_F32 && datatype != INFINI_F16) {
        return INFINICCL_STATUS_BAD_DATATYPE;
    }

    MPI_Datatype MPI_HALF;
    MPI_Type_contiguous(2, MPI_BYTE, &MPI_HALF);
    MPI_Type_commit(&MPI_HALF);
    MPI_Datatype datatype_cpu = getCpuDtype(datatype);
    MPI_Comm* comm = (MPI_Comm*)comms[0].comm;
    MPI_Allreduce(sendbuf, recvbuf, count, datatype_cpu, MPI_SUM, *comm);
    
    return INFINICCL_STATUS_SUCCESS;
}