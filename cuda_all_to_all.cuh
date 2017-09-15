#ifndef CUDA_ALL_TO_ALL_CUH
#define CUDA_ALL_TO_ALL_CUH

#include <cstdint>
#include <stdexcept>
#include "hpc_helpers.cuh"

namespace all2all {

    template<
        uint8_t num_gpus,
        uint8_t throw_exceptions=true>
    class all2all_t {

        cudaStream_t * streams;
        uint64_t * device_ids;
        uint64_t peer_status[num_gpus][num_gpus];
        bool external_device_ids;
        bool external_streams;

    public:
    
        void print_connectivity_matrix () const {
            std::cout << "STATUS: connectivity matrix:" << std::endl;
            for (uint64_t src_gpu = 0; src_gpu < num_gpus; ++src_gpu)
                for (uint64_t dst_gpu = 0; dst_gpu < num_gpus; ++dst_gpu)
                    std::cout << (dst_gpu == 0 ? "STATUS: |" : "")
                              << uint64_t(peer_status[src_gpu][dst_gpu])
                              << (dst_gpu+1 == num_gpus ? "|\n" : " ");
        }
    
        void sync () const {
            // synchronize all streams
            for (uint64_t src_gpu = 0; src_gpu < num_gpus; ++src_gpu) {
                const uint64_t src = device_ids[src_gpu];
                cudaSetDevice(src);
                for (uint64_t dst_gpu = 0; dst_gpu < num_gpus; ++dst_gpu) {
                    const uint64_t offset = src_gpu*num_gpus+dst_gpu;
                    cudaStreamSynchronize(streams[offset]);
                }
            } CUERR
        }

        all2all_t (
            uint64_t     * device_ids_=0,  // a list of unique device ids
            cudaStream_t * streams_=0)     // stream from src_gpu to dst_gpu
                : device_ids(device_ids_), streams(streams_) {

            // check if external device IDs are provides
            // if not create ascending list of device IDs
            if (device_ids) {
                external_device_ids = true;
            } else {
                external_device_ids = false;
                device_ids = new uint64_t[num_gpus];
                for (uint64_t src_gpu = 0; src_gpu < num_gpus; ++src_gpu)
                    device_ids[src_gpu] = src_gpu;
            }
            

            // check if external streams are provides
            // if not create num_gpus*num_gpus many streams
            if (streams) {
                external_streams = true;
                sync(); // do not remove this, don't!
            } else {
                external_streams = false;
                streams = new cudaStream_t[num_gpus*num_gpus];
                for (uint64_t src_gpu = 0; src_gpu < num_gpus; ++src_gpu) {
                    const uint64_t src = device_ids[src_gpu];
                    cudaSetDevice(src);
                    cudaDeviceSynchronize(); // do not remove this, don't!
                    for (uint64_t dst_gpu = 0; dst_gpu < num_gpus; ++dst_gpu) {
                        cudaStreamCreate(streams+src_gpu*num_gpus+dst_gpu);
                    }
                } CUERR
            }

            // compute the connectivity matrix
            for (uint64_t src_gpu = 0; src_gpu < num_gpus; ++src_gpu) {
                const uint64_t src = device_ids[src_gpu];
                cudaSetDevice(src);
                for (uint64_t dst_gpu = 0; dst_gpu < num_gpus; ++dst_gpu) {
                    const uint64_t dst = device_ids[dst_gpu];

                    // check if src can access dst
                    if (src == dst) {
                        peer_status[src_gpu][dst_gpu] = 1;
                    } else {
                        int32_t status;
                        cudaDeviceCanAccessPeer(&status, src, dst);
                        peer_status[src_gpu][dst_gpu] = status ? 2 : 0;
                    }
                    
                }
            } CUERR

            for (uint64_t src_gpu = 0; src_gpu < num_gpus; ++src_gpu) {
                const uint64_t src = device_ids[src_gpu];
                cudaSetDevice(src);
                for (uint64_t dst_gpu = 0; dst_gpu < num_gpus; ++dst_gpu) {
                    const uint64_t dst = device_ids[dst_gpu];
                    if (src_gpu != dst_gpu) {
                        if (throw_exceptions)
                            if (src == dst)
                                throw std::invalid_argument(
                                    "Device identifiers are not unique.");
                        
                        if (peer_status[src_gpu][dst_gpu] == 2)
                            cudaDeviceEnablePeerAccess(dst, 0);
                    }
                }
            } CUERR
            
        }

        ~all2all_t () {

            // free streams if self-managed
            if (!external_streams) {

                // synchronize all streams and destroy them
                for (uint64_t src_gpu = 0; src_gpu < num_gpus; ++src_gpu) {
                    const uint64_t src = device_ids[src_gpu]; 
                    cudaSetDevice(src);
                    for (uint64_t dst_gpu = 0; dst_gpu < num_gpus; ++dst_gpu) {
                        const uint64_t offset = src_gpu*num_gpus+dst_gpu;
                        cudaStreamSynchronize(streams[offset]);
                        cudaStreamDestroy(streams[offset]);
                    }
                } CUERR

                delete [] streams;
            }            
                       
            // disable peer access
            for (uint64_t src_gpu = 0; src_gpu < num_gpus; ++src_gpu) {
                const uint64_t src = device_ids[src_gpu];
                cudaSetDevice(src);
                for (uint64_t dst_gpu = 0; dst_gpu < num_gpus; ++dst_gpu) {
                    const uint64_t dst = device_ids[dst_gpu];
                    if (peer_status[src_gpu][dst_gpu] == 2)
                        cudaDeviceDisablePeerAccess(dst);                    
                }
            } CUERR
            
            // free device ids if self-managed
            if (!external_device_ids) {
                delete [] device_ids;
            }

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
                const uint64_t src_off = src_gpu*num_gpus;
                const uint64_t src = device_ids[src_gpu];
                cudaSetDevice(src);
                for (uint64_t dst_gpu = 0; dst_gpu < num_gpus; ++dst_gpu) {
                    const uint64_t dst = device_ids[dst_gpu];
                    const uint64_t len = table[src_gpu][dst_gpu];
                    value_t * from = srcs[src_gpu] + h_table[src_gpu][dst_gpu];
                    value_t * to   = dsts[dst_gpu] + v_table[src_gpu][dst_gpu];

                    cudaMemcpyPeerAsync(to, dst, from, src,
                                        len*sizeof(value_t),
                                        streams[src_off+dst_gpu]);     CUERR
                                    
                } CUERR
            }

            return true;
        }
    };
}
#endif
