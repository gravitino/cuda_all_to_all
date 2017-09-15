#include <iostream>
#include <random>
#include <vector>
#include "hpc_helpers.cuh"
#include "cuda_all_to_all.cuh"


template <
    uint8_t num_gpus=4,
    uint8_t max_gpus=num_gpus,
    uint8_t verbosity=1,
    typename value_t=double>
void check (
    uint64_t * device_ids_ = 0,
    uint64_t   partition_size=1UL<<27,
    int64_t    partition_delta=1UL<<20,
    double     security_factor=1.1) {
    
    static_assert(num_gpus <= max_gpus, 
                  "choose less or equal GPUs out of the existing ones.");

    if(partition_size < partition_delta)
        throw std::invalid_argument(
            "partition_delta <= partition_size not supported.");
    
    uint64_t device_ids[max_gpus];
    for (uint64_t gpu = 0; gpu < max_gpus; gpu++)
        device_ids[gpu] = device_ids_ ? device_ids_[gpu] : gpu;

    // initialize RNG
    std::random_device rd;
    std::mt19937 engine(rd());
    
    if (!device_ids) {
        // scramble device ids using Fisher Yates
        std::uniform_int_distribution<uint64_t> tau;
        for (uint64_t src = max_gpus; src > 1; --src) {
            uint64_t trg = tau(engine) % (src-1);
            std::swap(device_ids[src-1], device_ids[trg]);
        }
    }
    
    // print current device ids
    if (verbosity > 0) {        
        std::cout << "STATUS: device identifiers: ";
        for (uint64_t gpu = 0; gpu < num_gpus; ++gpu)
            std::cout << device_ids[gpu] << " "; 
        std::cout << std::endl; 
    }
    
    // create the partion table
    std::uniform_int_distribution<int64_t> rho(-partition_delta,
                                               +partition_delta);
    uint64_t table[num_gpus][num_gpus];
    for (uint64_t gpu = 0; gpu < num_gpus; ++gpu)
        for (uint64_t part = 0; part < num_gpus; ++part)
            table[gpu][part] = partition_size + rho(engine);

    // show partition table
    if (verbosity > 0) {
        std::cout << "STATUS: partition table:" << std::endl;
        for (uint64_t gpu = 0; gpu < num_gpus; ++gpu)
            for (uint64_t part = 0; part < num_gpus; ++part)
                std::cout << (part == 0 ? "STATUS: |" : "") 
                          << table[gpu][part]
                          << (part+1 == num_gpus ? "|\n" : " ");
    }

    // determine lengths of source arrays
    value_t * srcs[num_gpus],  * dsts[num_gpus];
    uint64_t srcs_lens[num_gpus], dsts_lens[num_gpus];

    // determine lengths of source arrays
    for (uint64_t gpu = 0; gpu < num_gpus; ++gpu)
        srcs_lens[gpu] = uint64_t(num_gpus*partition_size*security_factor);

    // determine lengths of destination arrays
    for (uint64_t gpu = 0; gpu < num_gpus; ++gpu)
        dsts_lens[gpu] = uint64_t(num_gpus*partition_size*security_factor);

    // show srcs_lens and dsts_lens
    if (verbosity > 0) {
        std::cout << "STATUS: srcs_lens: ";
        for (uint64_t gpu = 0; gpu < num_gpus; ++gpu)
            std::cout << srcs_lens[gpu] << " ";
        std::cout << std::endl;

        std::cout << "STATUS: dsts_lens: ";
        for (uint64_t gpu = 0; gpu < num_gpus; ++gpu)
            std::cout << dsts_lens[gpu] << " ";
        std::cout << std::endl;
    }

    // compute prefix sums over the partition table
    uint64_t h_table[num_gpus][num_gpus+1] = {0}; // horizontal scan
    uint64_t v_table[num_gpus+1][num_gpus] = {0}; // vertical scan

    for (uint64_t gpu = 0; gpu < num_gpus; ++gpu) {
        for (uint64_t part = 0; part < num_gpus; ++part) {
            h_table[gpu][part+1] = table[gpu][part]+h_table[gpu][part];
            v_table[gpu+1][part] = table[gpu][part]+v_table[gpu][part];
        }
    }
    
    // check if src_lens are compatible with partition table
    for (uint64_t gpu = 0; gpu < num_gpus; ++gpu)
        if (h_table[gpu][num_gpus] > srcs_lens[gpu])
            throw std::invalid_argument(
                "not enough memory in src_lens, increase security_factor.");
    
    // check if dsts_lens are compatible with partition table
    for (uint64_t gpu = 0; gpu < num_gpus; ++gpu)
        if (v_table[num_gpus][gpu] > dsts_lens[gpu])
            throw std::invalid_argument(
                "not enough memory in dsts_lens, increase security_factor.");

    // create host-sided vectors for each gpu
    TIMERSTART(create_partitions)
    std::vector<value_t> srcs_host[num_gpus];
    for (uint64_t gpu = 0; gpu < num_gpus; ++gpu) {
        srcs_host[gpu] = std::vector<value_t>(srcs_lens[gpu], 0);
        for (uint64_t part = 0; part < num_gpus; ++part)
            for (uint64_t i = h_table[gpu][part]; i < h_table[gpu][part+1]; ++i)
                srcs_host[gpu][i] = part+1;
    }
    TIMERSTOP(create_partitions)

    // print sources arrays
    if (verbosity > 1) {
        std::cout << "srcs[gpu] entries:" << std::endl;
        for (uint64_t gpu = 0; gpu < num_gpus; ++gpu) {
            std::cout << "gpu " << gpu << ": ";
            for (uint64_t i = 0; i < srcs_lens[gpu]; ++i)
                std::cout << srcs_host[gpu][i] << " ";
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    // compute all entries that have to be communicated
    uint64_t comm_entries = 0, diag_entries = 0;
    for (uint64_t gpu = 0; gpu < num_gpus; ++gpu)
        for (uint64_t part = 0; part < num_gpus; ++part)
            if (gpu == part)
                diag_entries += table[gpu][part];
            else
                comm_entries += table[gpu][part];

    if (verbosity > 0) {
        std::cout << "STATUS: number of all inter-GPU communicated entries: "
                  << comm_entries << std::endl;
        std::cout << "STATUS: number of all intra-GPU copied entries: "
                  << diag_entries << std::endl;
    }

    // alloc srcs and dsts arrays on gpu and zero afterwards
    TIMERSTART(H2D_and_zero)
    for (uint64_t gpu = 0; gpu < num_gpus; ++gpu) {
        cudaSetDevice(device_ids[gpu]);                                   CUERR
        cudaMalloc(&srcs[gpu], sizeof(value_t)*srcs_lens[gpu]);           CUERR
        cudaMalloc(&dsts[gpu], sizeof(value_t)*dsts_lens[gpu]);           CUERR
        cudaMemcpy(srcs[gpu], srcs_host[gpu].data(),
                   sizeof(value_t)*srcs_lens[gpu], H2D);                  CUERR
        cudaMemset(dsts[gpu], 0, sizeof(value_t)*dsts_lens[gpu]);         CUERR
    }
    TIMERSTOP(H2D_and_zero)
   
    // perform all to all
    cudaSetDevice(0);
    cudaEvent_t start, stop;
    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    
    all2all::all2all_t<num_gpus, true> comm(device_ids);
    if (verbosity > 0)
        comm.print_connectivity_matrix();
    auto success = comm.execAsync(srcs, srcs_lens, dsts, dsts_lens, table);
    comm.sync();

    if (verbosity > 0)
        if (success)
            std::cout << "STATUS: (SUCCESS) all2all successful." << std::endl;
        else
            std::cout << "STATUS: (FAILURE) all2all unsuccessful." << std::endl;

    cudaSetDevice(0);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    double comm_bandwidth = comm_entries*sizeof(value_t)*1000/time/(1UL<<30);
    double diag_bandwidth = diag_entries*sizeof(value_t)*1000/time/(1UL<<30);
    std::cout << "TIMING: " << time << " ms "
              << "-> " << (num_gpus > 1 ? comm_bandwidth : diag_bandwidth)
              << " GB/s " << (num_gpus > 1 ? "inter-GPU" : "intra-GPU")
              << " bandwidth (all2all) " << std::endl;

    // check dsts for correct entries
    TIMERSTART(D2H_and_validation)
    std::vector<value_t> dsts_host[num_gpus];
    for (uint64_t gpu = 0; gpu < num_gpus; ++gpu) {
        cudaSetDevice(device_ids[gpu]);                               CUERR
        dsts_host[gpu] = std::vector<value_t>(dsts_lens[gpu], 0);
        cudaMemcpy(dsts_host[gpu].data(), dsts[gpu],
                   sizeof(value_t)*dsts_lens[gpu], D2H);              CUERR
        uint64_t accum = 0;
        for (uint64_t i = 0; i < dsts_lens[gpu]; ++i)
            accum += dsts_host[gpu][i] == gpu+1;
        if (accum != v_table[num_gpus][gpu])
            std::cout << "ERROR: dsts entries differ from expectation "
                      << "(expected: " << v_table[num_gpus][gpu]
                      << " seen: " << accum << " )"
                      << std::endl;
    }
    TIMERSTOP(D2H_and_validation)

    if (verbosity > 1) {
        for (uint64_t gpu = 0; gpu < num_gpus; ++gpu) {
            std::cout << "gpu " << gpu << ": ";
            for (uint64_t i = 0; i < dsts_lens[gpu]; ++i)
                std::cout << dsts_host[gpu][i] << " ";
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
    
    // free memory
    for (uint64_t gpu = 0; gpu < num_gpus; ++gpu) {
        cudaSetDevice(device_ids[gpu]); 
        cudaFree(srcs[gpu]);
        cudaFree(dsts[gpu]);
    } CUERR
    
    std::cout << std::endl;

}

int main (int argc, char* argv[]) {
    
    {
        uint64_t device_ids_0[1] = {0};
        check<1>(device_ids_0);    
        uint64_t device_ids_1[1] = {1};
        check<1>(device_ids_1);    
        uint64_t device_ids_2[1] = {2};
        check<1>(device_ids_2);    
        uint64_t device_ids_3[1] = {3};
        check<1>(device_ids_3);
    }
    
    
    {
        uint64_t device_ids_0[2] = {0, 1};
        check<2>(device_ids_0);    
        uint64_t device_ids_1[2] = {1, 0};
        check<2>(device_ids_1);    
        uint64_t device_ids_2[2] = {2, 3};
        check<2>(device_ids_2);    
        uint64_t device_ids_3[2] = {3, 2};
        check<2>(device_ids_3);
    }
    
    {
        uint64_t device_ids_0[4] = {0, 1, 2, 3};
        check<4>(device_ids_0);    
    
    }
    
}
