#ifndef CUDA_ALL_TO_ALL_CUH
#define CUDA_ALL_TO_ALL_CUH

#include <cstdint>
#include <stdexcept>
#include "hpc_helpers.cuh"

namespace all2all {

    template <
        uint8_t num_gpus,
        bool throw_exceptions=true,
        uint64_t PEER_STATUS_SLOW=0,
        uint64_t PEER_STATUS_DIAG=1,
        uint64_t PEER_STATUS_FAST=2>
    class context_t {

        cudaStream_t * streams;
        uint64_t * device_ids;
        uint64_t peer_status[num_gpus][num_gpus];

    public:

        context_t (uint64_t * device_ids_=0) {

            // copy num_gpus many device identifiers
            device_ids = new uint64_t[num_gpus];
            for (uint64_t src_gpu = 0; src_gpu < num_gpus; ++src_gpu)
                device_ids[src_gpu] = device_ids_ ?
                                      device_ids_[src_gpu] : src_gpu;

            // create num_gpus^2 streams where streams[gpu*num_gpus+part]
            // denotes the stream to be used for GPU gpu and partition part
            streams = new cudaStream_t[num_gpus*num_gpus];
            for (uint64_t src_gpu = 0; src_gpu < num_gpus; ++src_gpu) {
                cudaSetDevice(get_device_id(src_gpu));
                cudaDeviceSynchronize();
                for (uint64_t part = 0; part < num_gpus; ++part) {
                    cudaStreamCreate(get_streams(src_gpu)+part);
                }
            } CUERR


            // compute the connectivity matrix
            for (uint64_t src_gpu = 0; src_gpu < num_gpus; ++src_gpu) {
                const uint64_t src = get_device_id(src_gpu);
                cudaSetDevice(src);
                for (uint64_t dst_gpu = 0; dst_gpu < num_gpus; ++dst_gpu) {
                    const uint64_t dst = get_device_id(dst_gpu);

                    // check if src can access dst
                    if (src == dst) {
                        peer_status[src_gpu][dst_gpu] = PEER_STATUS_DIAG;
                    } else {
                        int32_t status;
                        cudaDeviceCanAccessPeer(&status, src, dst);
                        peer_status[src_gpu][dst_gpu] = status ?
                                                        PEER_STATUS_FAST :
                                                        PEER_STATUS_SLOW ;
                    }
                }
            } CUERR

            for (uint64_t src_gpu = 0; src_gpu < num_gpus; ++src_gpu) {
                const uint64_t src = get_device_id(src_gpu);
                cudaSetDevice(src);
                for (uint64_t dst_gpu = 0; dst_gpu < num_gpus; ++dst_gpu) {
                    const uint64_t dst = get_device_id(dst_gpu);

                    if (src_gpu != dst_gpu) {
                        if (throw_exceptions)
                            if (src == dst)
                                throw std::invalid_argument(
                                    "Device identifiers are not unique.");
                    }

                    if (peer_status[src_gpu][dst_gpu] == PEER_STATUS_FAST) {
                        cudaDeviceEnablePeerAccess(dst, 0);

                        // consume error for rendundant
                        // peer access initialization
                        const cudaError_t cuerr = cudaGetLastError();
                        if (cuerr == cudaErrorPeerAccessAlreadyEnabled)
                            std::cout << "STATUS: redundant enabling of "
                                      << "peer access from GPU " << src_gpu
                                      << " to GPU " << dst << " attempted."
                                      << std::endl;
                        else if (cuerr)
                            std::cout << "CUDA error: "
                                      << cudaGetErrorString(cuerr) << " : "
                                      << __FILE__ << ", line "
                                      << __LINE__ << std::endl;
                    }

                }
            } CUERR
        }

        ~context_t () {

            // synchronize and destroy streams
            for (uint64_t src_gpu = 0; src_gpu < num_gpus; ++src_gpu) {
                cudaSetDevice(get_device_id(src_gpu));
                cudaDeviceSynchronize();
                for (uint64_t part = 0; part < num_gpus; ++part) {
                    cudaStreamSynchronize(get_streams(src_gpu)[part]);
                    cudaStreamDestroy(get_streams(src_gpu)[part]);
                }
            } CUERR

            // disable peer access
            for (uint64_t src_gpu = 0; src_gpu < num_gpus; ++src_gpu) {
                const uint64_t src = get_device_id(src_gpu);
                cudaSetDevice(src);
                for (uint64_t dst_gpu = 0; dst_gpu < num_gpus; ++dst_gpu) {
                    const uint64_t dst = get_device_id(dst_gpu);

                    if (peer_status[src_gpu][dst_gpu] == PEER_STATUS_FAST) {
                        cudaDeviceDisablePeerAccess(dst);

                        // consume error for rendundant
                        // peer access deactivation
                        const cudaError_t cuerr = cudaGetLastError();
                        if (cuerr == cudaErrorPeerAccessNotEnabled)
                            std::cout << "STATUS: redundant disabling of "
                                      << "peer access from GPU " << src_gpu
                                      << " to GPU " << dst << " attempted."
                                      << std::endl;
                        else if (cuerr)
                            std::cout << "CUDA error: "
                                      << cudaGetErrorString(cuerr) << " : "
                                       << __FILE__ << ", line "
                                       << __LINE__ << std::endl;
                    }
                }
            } CUERR

            // free streams and device identifiers
            delete [] streams;
            delete [] device_ids;
        }

        uint64_t get_device_id (const uint64_t gpu) const noexcept {

            // return the actual device identifier of GPU gpu
            return device_ids[gpu];
        }

        cudaStream_t * get_streams (const uint64_t gpu) const noexcept {

            // return pointer to all num_gpus many streams of GPU gpu
            return streams+gpu*num_gpus;
        }

        void sync_gpu_streams (const uint64_t gpu) const noexcept {

            // sync all streams associated with the corresponding GPU
            cudaSetDevice(get_device_id(gpu)); CUERR
            for (uint64_t part = 0; part < num_gpus; ++part)
                cudaStreamSynchronize(get_streams(gpu)[part]);
            CUERR
        }

        void sync_all_streams () const noexcept {

            // sync all streams of the context
            for (uint64_t gpu = 0; gpu < num_gpus; ++gpu)
                sync_gpu_streams(gpu);
            CUERR
        }

        void sync_hard () const noexcept {

            // sync all GPUs
            for (uint64_t gpu = 0; gpu < num_gpus; ++gpu) {
                cudaSetDevice(get_device_id(gpu));
                cudaDeviceSynchronize();
            } CUERR
        }

        bool is_valid () const noexcept {

            // both streams and device identifiers are created
            return streams && device_ids;
        }

        void print_connectivity_matrix () const {
            std::cout << "STATUS: connectivity matrix:" << std::endl;
            for (uint64_t src_gpu = 0; src_gpu < num_gpus; ++src_gpu)
                for (uint64_t dst_gpu = 0; dst_gpu < num_gpus; ++dst_gpu)
                    std::cout << (dst_gpu == 0 ? "STATUS: |" : "")
                              << uint64_t(peer_status[src_gpu][dst_gpu])
                              << (dst_gpu+1 == num_gpus ? "|\n" : " ");
        }
    };

    template <
        uint8_t num_gpus,
        bool throw_exceptions=true>
    class point2point_t {

        const context_t<num_gpus> * context;
        bool external_context;

    public:

        point2point_t (
            uint64_t * device_ids_=0) : external_context (false) {

            if (device_ids_)
                context = new context_t<num_gpus>(device_ids_);
            else
                context = new context_t<num_gpus>();
        }

        point2point_t (
            context_t<num_gpus> * context_) : context(context_),
                                              external_context (true) {
                if (throw_exceptions)
                    if (!context->is_valid())
                        throw std::invalid_argument(
                            "You have to pass a valid context!"
                        );
        }

        ~point2point_t () {
            if (!external_context)
                delete context;
        }

        template <
            cudaMemcpyKind cudaMemcpyDirection,
            typename value_t,
            typename index_t>
        bool execAsync (
            value_t * srcs[num_gpus],
            value_t * dsts[num_gpus],
            index_t   lens[num_gpus]) const noexcept {

            for (uint64_t src_gpu = 0; src_gpu < num_gpus; ++src_gpu) {
                cudaSetDevice(context->get_device_id(src_gpu));
                cudaMemcpyAsync(dsts[src_gpu], srcs[src_gpu],
                                sizeof(value_t)*lens[src_gpu],
                                cudaMemcpyDirection,
                                context->get_streams(src_gpu)[0]);
            } CUERR

            return true;
        }

        template <
            typename value_t,
            typename index_t>
        bool execH2DAsync (
            value_t * srcs[num_gpus],
            value_t * dsts[num_gpus],
            index_t   lens[num_gpus]) const noexcept {

            return execAsync<cudaMemcpyHostToDevice>(srcs, dsts, lens);
        }

        template <
            typename value_t,
            typename index_t>
        bool execD2HAsync (
            value_t * srcs[num_gpus],
            value_t * dsts[num_gpus],
            index_t   lens[num_gpus]) const noexcept {

            return execAsync<cudaMemcpyDeviceToHost>(srcs, dsts, lens);
        }

        template <
            typename value_t,
            typename index_t>
        bool execD2DAsync (
            value_t * srcs[num_gpus],
            value_t * dsts[num_gpus],
            index_t   lens[num_gpus]) const noexcept {

            return execAsync<cudaMemcpyDeviceToDevice>(srcs, dsts, lens);
        }

        void sync () const noexcept {
            context->sync_all_streams();
        }

        void sync_hard () const noexcept {
            context->sync_hard();
        }
    };

    template<
        uint8_t num_gpus,
        uint8_t throw_exceptions=true>
    class all2all_t {

        context_t<num_gpus> * context;
        bool external_context;

    public:

        all2all_t (
            uint64_t * device_ids_=0) : external_context (false){

            if (device_ids_)
                context = new context_t<num_gpus>(device_ids_);
            else
                context = new context_t<num_gpus>();
        }

        all2all_t (
            context_t<num_gpus> * context_) : context(context_),
                                              external_context (true) {
                if (throw_exceptions)
                    if (!context->is_valid())
                        throw std::invalid_argument(
                            "You have to pass a valid context!"
                        );
        }

        ~all2all_t () {
            if (!external_context)
                delete context;
        }

        template <
            typename value_t,
            typename index_t,
            typename table_t>
        bool execAsync (
            value_t * srcs[num_gpus],        // src[k] resides on device_ids[k]
            index_t srcs_lens[num_gpus],     // src_len[k] is length of src[k]
            value_t * dsts[num_gpus],        // dst[k] resides on device_ids[k]
            index_t dsts_lens[num_gpus],     // dst_len[0] is length of dst[k]
            table_t table[num_gpus][num_gpus]) const {  // [src_gpu, partition]

            // syncs with zero stream in order to enforce sequential
            // consistency with traditional synchronous memcpy calls
            if (!external_context)
                context->sync_hard();

            // compute prefix sums over the partition table
            uint64_t h_table[num_gpus][num_gpus+1] = {0}; // horizontal scan
            uint64_t v_table[num_gpus+1][num_gpus] = {0}; // vertical scan

            for (uint64_t gpu = 0; gpu < num_gpus; ++gpu) {
                for (uint64_t part = 0; part < num_gpus; ++part) {
                    h_table[gpu][part+1] = table[gpu][part]+h_table[gpu][part];
                    v_table[gpu+1][part] = table[gpu][part]+v_table[gpu][part];
                }
            }

            // check src_lens for compatibility
            bool valid_srcs_lens = true;
            for (uint64_t src_gpu = 0; src_gpu < num_gpus; ++src_gpu)
                valid_srcs_lens &= h_table[src_gpu][num_gpus]
                                <= srcs_lens[src_gpu];
            if (!valid_srcs_lens)
                if (throw_exceptions)
                    throw std::invalid_argument(
                        "srcs_lens not compatible with partition_table.");
                else return false;

            // check dst_lens for compatibility
            bool valid_dsts_lens = true;
            for (uint64_t dst_gpu = 0; dst_gpu < num_gpus; ++dst_gpu)
                valid_dsts_lens &= v_table[num_gpus][dst_gpu]
                                <= dsts_lens[dst_gpu];
            if (!valid_dsts_lens)
                if (throw_exceptions)
                    throw std::invalid_argument(
                        "dsts_lens not compatible with partition_table.");
                else return false;

            // issue asynchronous copies
            for (uint64_t src_gpu = 0; src_gpu < num_gpus; ++src_gpu) {
                const uint64_t src = context->get_device_id(src_gpu);
                cudaSetDevice(src);
                for (uint64_t dst_gpu = 0; dst_gpu < num_gpus; ++dst_gpu) {
                    const uint64_t dst = context->get_device_id(dst_gpu);
                    const uint64_t len = table[src_gpu][dst_gpu];
                    value_t * from = srcs[src_gpu] + h_table[src_gpu][dst_gpu];
                    value_t * to   = dsts[dst_gpu] + v_table[src_gpu][dst_gpu];

                    cudaMemcpyPeerAsync(to, dst, from, src,
                                        len*sizeof(value_t),
                                        context->get_streams(src_gpu)[dst_gpu]);

                } CUERR
            }

            return true;
        }

        void print_connectivity_matrix () const noexcept {
            context->print_connectivity_matrix();
        }

        void sync () const noexcept {
            context->sync_all_streams();
        }

        void sync_hard () const noexcept {
            context->sync_hard();
        }
    };

    template<
        uint8_t num_gpus,
        uint8_t throw_exceptions=true>
    class memory_manager_t {

        context_t<num_gpus> * context;
        bool external_context;

    public:

        memory_manager_t (
            uint64_t * device_ids_=0) : external_context (false) {

            if (device_ids_)
                context = new context_t<num_gpus>(device_ids_);
            else
                context = new context_t<num_gpus>();
        }

        memory_manager_t (
            context_t<num_gpus> * context_) : context(context_),
                                              external_context (true) {
                if (throw_exceptions)
                    if (!context->is_valid())
                        throw std::invalid_argument(
                            "You have to pass a valid context!"
                        );
        }

        ~memory_manager_t () {
            if (!external_context)
                delete context;
        }


        template <
            typename value_t,
            typename index_t>
        value_t ** alloc_device(index_t lens[num_gpus], bool zero=true) const {

            value_t ** data = new value_t*[num_gpus];

            // malloc as device-sided memory
            for (uint64_t gpu = 0; gpu < num_gpus; ++gpu) {
                cudaSetDevice(context->get_device_id(gpu));
                cudaMalloc(&data[gpu], sizeof(value_t)*lens[gpu]);
                if (zero)
                    cudaMemsetAsync(data[gpu], 0, sizeof(value_t)*lens[gpu],
                                    context->get_streams(gpu)[0]);
            }
            CUERR

            return data;
        }

        template <
            typename value_t,
            typename index_t>
        value_t ** alloc_host(index_t lens[num_gpus], bool zero=true) const {

            value_t ** data = new value_t*[num_gpus];

            // malloc as host-sided pinned memory
            for (uint64_t gpu = 0; gpu < num_gpus; ++gpu) {
                cudaMallocHost(&data[gpu], sizeof(value_t)*lens[gpu]);
                if (zero)
                    std::memset(data[gpu], 0, sizeof(value_t)*lens[gpu]);
            }
            CUERR

            return data;
        }

        template <
            typename value_t>
        void free_device(value_t ** data) const {

            for (uint64_t gpu = 0; gpu < num_gpus; ++gpu) {
                cudaSetDevice(context->get_device_id(gpu));
                cudaFree(data[gpu]);
            }
            CUERR

            delete [] data;
        }

        template <
            typename value_t>
        void free_host(value_t ** data) const {

            for (uint64_t gpu = 0; gpu < num_gpus; ++gpu)
                cudaFreeHost(data[gpu]);
            CUERR

            delete [] data;
        }
    };

    template <
        uint64_t num_gpus,
        typename index_t>
    uint64_t part_hash(index_t x) {
        return (*((uint64_t*)(&x))) % num_gpus;
    }

    template <
        typename index_t> __device__ __forceinline__
    index_t atomicAggInc(
        index_t * counter) {

        // accumulate over whole warp to reduce atomic congestion
        const int lane = threadIdx.x % 32;
        const int mask = __ballot(1);
        const int leader = __ffs(mask) - 1;
        index_t res;
        if (lane == leader)
            res = atomicAdd(counter, __popc(mask));
        res = __shfl(res, leader);

        return res + __popc(mask & ((1 << lane) -1));
    }

    template <
        typename value_t,
        typename index_t,
        typename cnter_t,
        typename funct_t,
        typename desir_t> __global__
    void binary_split(
        value_t * src,
        value_t * dst,
        index_t   len,
        cnter_t * counter,
        funct_t   part_hash,
        desir_t   desired) {

        const auto thid = blockDim.x*blockIdx.x + threadIdx.x;

        for(index_t i = thid; i < len; i += gridDim.x*blockDim.x) {
            const value_t value = src[i];
            if (part_hash(value) == desired) {
                const index_t j = atomicAggInc(counter);
                dst[j] = value;
            }
        }
    }

    template <
        uint8_t num_gpus,
        typename table_t,
        typename cnter_t=uint32_t,
        bool throw_exceptions=true>
    class multisplit_t {

        const context_t<num_gpus> * context;
        bool external_context;
        table_t * table_device[num_gpus];
        cnter_t * counters_device[num_gpus];

    public:

        multisplit_t (
            uint64_t * device_ids_=0) : external_context (false) {

            if (device_ids_)
                context = new context_t<num_gpus>(device_ids_);
            else
                context = new context_t<num_gpus>();

            for (uint64_t gpu = 0; gpu < num_gpus; ++gpu) {
                cudaSetDevice(context->get_device_id(gpu));
                cudaMalloc(&table_device[gpu], sizeof(table_t)*num_gpus);
                cudaMalloc(&counters_device[gpu], sizeof(cnter_t));
            } CUERR
        }

        multisplit_t (
            context_t<num_gpus> * context_) : context(context_),
                                              external_context (true) {
            if (throw_exceptions)
                if (!context->is_valid())
                    throw std::invalid_argument(
                        "You have to pass a valid context!"
                    );

            for (uint64_t gpu = 0; gpu < num_gpus; ++gpu) {
                cudaSetDevice(context->get_device_id(gpu));
                cudaMalloc(&table_device[gpu], sizeof(table_t)*num_gpus);
                cudaMalloc(&counters_device[gpu], sizeof(cnter_t));
            } CUERR
        }

        ~multisplit_t () {

            for (uint64_t gpu = 0; gpu < num_gpus; ++gpu) {
                cudaSetDevice(context->get_device_id(gpu));
                cudaFree(table_device[gpu]);
                cudaFree(counters_device[gpu]);
            } CUERR

            if (!external_context)
                delete context;
        }

        template <
            typename value_t,
            typename index_t,
            typename funct_t>
        bool execAsync (
            value_t * srcs[num_gpus],
            index_t   srcs_lens[num_gpus],
            value_t * dsts[num_gpus],
            index_t   dsts_lens[num_gpus],
            table_t   table[num_gpus][num_gpus],
            funct_t   functor) const noexcept {

            for (uint64_t src_gpu = 0; src_gpu < num_gpus; ++src_gpu) {
                if (srcs_lens[src_gpu] > dsts_lens[src_gpu])
                    if (throw_exceptions)
                        throw std::invalid_argument(
                            "dsts_lens too small for given srcs_lens."
                        );
                    else return false;
            }
            
            for (uint64_t gpu = 0; gpu < num_gpus; ++gpu) {
                cudaSetDevice(context->get_device_id(gpu));
                cudaMemsetAsync(table_device[gpu], 0, sizeof(table_t)*num_gpus,
                                context->get_streams(gpu)[0]);
                uint64_t offset = 0;
                for (uint64_t part = 0; part < num_gpus; ++part) {
                    cudaMemsetAsync(counters_device[gpu], 0, sizeof(cnter_t),
                                    context->get_streams(gpu)[0]);
                    binary_split<<<128, 1024, 0, context->get_streams(gpu)[0]>>>
                       (srcs[gpu], dsts[gpu]+offset, srcs_lens[gpu],
                        counters_device[gpu], functor, part);
                                 
                    offset += table[gpu][part];
                }            
            } CUERR

            return true;
        }

        void sync () const noexcept {
            context->sync_all_streams();
        }

        void sync_hard () const noexcept {
            context->sync_hard();
        }
    };


    ///////////////////////////////////////////////////////////////////////////
    // experiments
    ///////////////////////////////////////////////////////////////////////////

    template <
        uint8_t num_gpus,
        bool throw_exceptions=true>
    class experiment_t {

        const context_t<num_gpus> * context;
        bool external_context;

    public:

        experiment_t (
            uint64_t * device_ids_=0) : external_context (false) {

            if (device_ids_)
                context = new context_t<num_gpus>(device_ids_);
            else
                context = new context_t<num_gpus>();
        }

        experiment_t (
            context_t<num_gpus> * context_) : context(context_),
                                              external_context (true) {
                if (throw_exceptions)
                    if (!context->is_valid())
                        throw std::invalid_argument(
                            "You have to pass a valid context!"
                        );
        }

        ~experiment_t () {
            if (!external_context)
                delete context;
        }

        template <
            typename value_t,
            typename index_t>
        bool create_partitions_host(
            value_t *srcs_host[num_gpus],
            index_t  srcs_lens[num_gpus],
            index_t  dsts_lens[num_gpus],
            index_t table[num_gpus][num_gpus]) const {

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
                    if (throw_exceptions)
                        throw std::invalid_argument(
                            "not enough memory in src_lens,"
                            "increase security_factor.");
                    else return false;

            // check if dsts_lens are compatible with partition table
            for (uint64_t gpu = 0; gpu < num_gpus; ++gpu)
                if (v_table[num_gpus][gpu] > dsts_lens[gpu])
                    if (throw_exceptions)
                        throw std::invalid_argument(
                            "not enough memory in dsts_lens,"
                            "increase security_factor.");
                    else return false;

            // fill the source array according to the partition table
            for (uint64_t gpu = 0; gpu < num_gpus; ++gpu) {
                for (uint64_t part = 0; part < num_gpus; ++part)
                    for (uint64_t i = h_table[gpu][part];
                         i < h_table[gpu][part+1]; ++i)
                        srcs_host[gpu][i] = part+1;
            }

            return true;
        }

        template <
            typename value_t,
            typename index_t>
        bool validate_all2all_host(
            value_t *dsts_host[num_gpus],
            index_t  dsts_lens[num_gpus],
            index_t table[num_gpus][num_gpus]) const {

            // compute prefix sums over the partition table
            uint64_t v_table[num_gpus+1][num_gpus] = {0}; // vertical scan

            for (uint64_t gpu = 0; gpu < num_gpus; ++gpu) {
                for (uint64_t part = 0; part < num_gpus; ++part) {
                    v_table[gpu+1][part] = table[gpu][part]+v_table[gpu][part];
                }
            }

            for (uint64_t gpu = 0; gpu < num_gpus; ++gpu) {
                uint64_t accum = 0;
                for (uint64_t i = 0; i < dsts_lens[gpu]; ++i)
                    accum += dsts_host[gpu][i] == gpu+1;
                if (accum != v_table[num_gpus][gpu]) {
                    std::cout << "ERROR: dsts entries differ from expectation "
                              << "(expected: " << v_table[num_gpus][gpu]
                              << " seen: " << accum << " )"
                              << std::endl;
                    return false;
                }
            }
            return true;
        }
    };
}
#endif

