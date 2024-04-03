// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/medusa_plugin/medusa_head.h"
#include "src/turbomind/models/medusa_plugin/medusa_weight.h"
#include "src/turbomind/utils/cuda_utils.h"
#include <iostream>
#include <thread>

template<typename T>
float T_to_float(T val)
{
    if (std::is_same<T, half>::value) {
        return __half2float((const half)val);
    }
    else if (std::is_same<T, __nv_bfloat16>::value) {
        return __bfloat162float((const __nv_bfloat16)val);
    }
}

template<typename T>
std::pair<turbomind::WeightType, turbomind::FtCudaDataType> get_type()
{
    turbomind::WeightType     weight_type;
    turbomind::FtCudaDataType model_file_type;
    if (std::is_same<T, half>::value) {
        weight_type     = turbomind::WeightType::kFP16;
        model_file_type = turbomind::FtCudaDataType::FP16;
    }
    else if (std::is_same<T, __nv_bfloat16>::value) {
        weight_type     = turbomind::WeightType::kBF16;
        model_file_type = turbomind::FtCudaDataType::BF16;
    }
    return std::make_pair(weight_type, model_file_type);
}

template<typename T>
class MedusaHeadExample {
public:
    MedusaHeadExample(size_t                      batch_size,
                      int                         medusa_num_heads,
                      size_t                      medusa_num_layers,
                      size_t                      hidden_size,
                      size_t                      vocab_size,
                      std::string                 dir_path,
                      turbomind::NcclParam        tensor_para,
                      cudaStream_t                stream,
                      turbomind::cublasMMWrapper* cublas_wrapper,
                      turbomind::IAllocator*      allocator):
        batch_size_(batch_size),
        medusa_num_heads_(medusa_num_heads),
        medusa_num_layers_(medusa_num_layers),
        hidden_size_(hidden_size),
        vocab_size_(vocab_size),
        model_(hidden_size, vocab_size, medusa_num_heads, stream, cublas_wrapper, allocator, tensor_para, false),
        input_buf_(nullptr),
        allocator_(allocator),
        rank_(tensor_para.rank_)
    {

        auto type            = get_type<T>();
        auto weight_type     = type.first;
        auto model_file_type = type.second;
        weights_             = std::make_unique<turbomind::MedusaWeight<T>>(medusa_num_heads,
                                                                medusa_num_layers,
                                                                hidden_size,
                                                                vocab_size,
                                                                weight_type,
                                                                tensor_para.world_size_,
                                                                tensor_para.rank_);
        weights_->load_model(dir_path, model_file_type);
    }

    ~MedusaHeadExample()
    {
        if (is_allocated) {
            allocator_->free((void**)&topk_output_ids_);
            allocator_->free((void**)&input_buf_);
        }
    }

    void forward(int seed = 7)
    {
        input_buf_ = nullptr;

        input_buf_ = (T*)allocator_->reMalloc(input_buf_, sizeof(T) * batch_size_ * hidden_size_, false);

        topk_output_ids_ =
            (int*)allocator_->reMalloc(topk_output_ids_, sizeof(int) * medusa_num_heads_ * batch_size_, false, true);

        size_t total_size = batch_size_ * std::max(hidden_size_, vocab_size_);
        buf_host_         = new T[total_size];

        for (int i = 0; i < total_size; i++) {
            buf_host_[i] = i % seed * 1.0;
        }

        cudaMemcpy(input_buf_, buf_host_, sizeof(T) * batch_size_ * hidden_size_, cudaMemcpyHostToDevice);

        is_allocated = true;

        turbomind::DataType  dtype = turbomind::getTensorType<T>();
        turbomind::TensorMap inputs{
            {"medusa_head_input", {turbomind::MEMORY_GPU, dtype, {batch_size_, hidden_size_}, input_buf_}},
        };

        // top 1
        turbomind::TensorMap outputs{
            {"medusa_head_output",
             {turbomind::MEMORY_GPU, dtype, {batch_size_, medusa_num_heads_, 1}, topk_output_ids_}},
        };

        model_.forward(&outputs, &inputs, *weights_.get());

        int* topk_output_ids = outputs.at("medusa_head_output").getPtr<int>();
        if (rank_ == 0) {
            for (int i = 0; i < batch_size_ * medusa_num_heads_; i++) {
                std::cout << "topk_output_ids[" << i << "]=" << topk_output_ids[i] << '\n';
            }
        }

        delete[] buf_host_;
    }

private:
    size_t batch_size_;
    size_t medusa_num_layers_;
    int    medusa_num_heads_;
    size_t hidden_size_;
    size_t vocab_size_;

    T* input_buf_;

    T* buf_host_ = nullptr;

    int* topk_output_ids_;

    turbomind::IAllocator* allocator_;

    turbomind::MedusaHead<T>                    model_;
    std::unique_ptr<turbomind::MedusaWeight<T>> weights_;

    bool is_allocated = false;
    int  rank_        = -1;
};

template<typename T>
void fire(int    tp,
          int    batch_size        = 2,
          int    seed              = 7,
          size_t medusa_num_heads  = 5,
          size_t medusa_num_layers = 1,
          size_t hidden_size       = 5120,
          size_t vocab_size        = 32000)
{
    std::string dtype;
    if (std::is_same<T, half>::value) {
        dtype = "fp16";
    }
    else if (std::is_same<T, __nv_bfloat16>::value) {
        dtype = "bf16";
    }

    std::string dir_path;
    if (tp == 1) {
        if (std::is_same<T, half>::value) {
            dir_path = "/workdir/medusa_output/fp16/tp1";
        }
        else if (std::is_same<T, __nv_bfloat16>::value) {
            dir_path = "/workdir/medusa_output/bf16/tp1";
        }
    }
    else if (tp == 2) {
        if (std::is_same<T, half>::value) {
            dir_path = "/workdir/medusa_output/fp16/tp2";
        }
        else if (std::is_same<T, __nv_bfloat16>::value) {
            dir_path = "/workdir/medusa_output/bf16/tp2";
        }
    }

    std::vector<cudaStream_t>                                                          streams(tp);
    std::vector<std::unique_ptr<turbomind::Allocator<turbomind::AllocatorType::CUDA>>> allocators(tp);
    std::vector<cublasHandle_t>                                                        cublas_handles(tp);
    std::vector<cublasLtHandle_t>                                                      cublaslt_handles(tp);
    std::vector<turbomind::cublasAlgoMap>                                              cublas_algo_maps(tp);
    std::vector<std::mutex>                                                            cublas_wrapper_mutexs(tp);
    std::vector<std::unique_ptr<turbomind::cublasMMWrapper>>                           cublas_wrappers(tp);
    std::vector<std::thread>                                                           threads;
    std::vector<std::unique_ptr<MedusaHeadExample<T>>>                                 models(tp);
    std::vector<turbomind::NcclParam>                                                  tensor_params(tp);

    turbomind::NcclUid tensor_para_nccl_uid;
    turbomind::ftNcclGetUniqueId(tensor_para_nccl_uid);
    const auto group_id = turbomind::ftNcclNextGroupId();
    turbomind::ftNcclGroupStart();
    for (int rank = 0; rank < tp; rank++) {
        turbomind::check_cuda_error(cudaSetDevice(rank));
        turbomind::ftNcclCommInitRank(tensor_params[rank], rank, tp, tensor_para_nccl_uid);
        tensor_params[rank].group_id_ = group_id;
    }
    turbomind::ftNcclGroupEnd();

    for (int rank = 0; rank < tp; rank++) {
        std::cout << "rank=" << rank << " tp=" << tp << " dtype=" << dtype << " batch=" << batch_size
                  << " seed=" << seed << '\n';

        turbomind::check_cuda_error(cudaSetDevice(rank));
        turbomind::check_cuda_error(cudaStreamCreate(&streams[rank]));
        allocators[rank] = std::unique_ptr<turbomind::Allocator<turbomind::AllocatorType::CUDA>>(
            new turbomind::Allocator<turbomind::AllocatorType::CUDA>(rank));
        allocators[rank]->setStream(streams[rank]);
        cublasCreate(&cublas_handles[rank]);
        cublasLtCreate(&cublaslt_handles[rank]);
        cublasSetStream(cublas_handles[rank], streams[rank]);
        cublas_algo_maps[rank] = turbomind::cublasAlgoMap();
        cublas_wrappers[rank] =
            std::unique_ptr<turbomind::cublasMMWrapper>(new turbomind::cublasMMWrapper(cublas_handles[rank],
                                                                                       cublaslt_handles[rank],
                                                                                       streams[rank],
                                                                                       &cublas_algo_maps[rank],
                                                                                       &cublas_wrapper_mutexs[rank],
                                                                                       allocators[rank].get()));
        if (std::is_same<T, half>::value) {
            cublas_wrappers[rank]->setFP16GemmConfig();
        }
        else if (std::is_same<T, __nv_bfloat16>::value) {
            cublas_wrappers[rank]->setBF16GemmConfig();
        }

        models[rank] = std::unique_ptr<MedusaHeadExample<T>>(new MedusaHeadExample<T>(batch_size,
                                                                                      medusa_num_heads,
                                                                                      medusa_num_layers,
                                                                                      hidden_size,
                                                                                      vocab_size,
                                                                                      dir_path,
                                                                                      tensor_params[rank],
                                                                                      streams[rank],
                                                                                      cublas_wrappers[rank].get(),
                                                                                      allocators[rank].get()));
    }

    auto threadForward = [streams, seed](int rank, MedusaHeadExample<T>* model) {
        turbomind::check_cuda_error(cudaSetDevice(rank));
        cudaDeviceSynchronize();
        model->forward(seed);
        cudaDeviceSynchronize();
        turbomind::check_cuda_error(cudaStreamSynchronize(streams[rank]));
    };

    for (int rank = 0; rank < tp; rank++) {
        threads.push_back(std::thread(threadForward, rank, models[rank].get()));
    }
    for (auto& t : threads) {
        t.join();
    }
}

int main(int argc, char** argv)
{
    std::vector<int>         seed_vec{7};
    std::vector<int>         batch_vec{2};
    std::vector<std::string> type_vec{"bf16", "fp16"};
    std::vector<int>         tp_vec{1, 2};
    for (const int seed : seed_vec) {
        for (const int batch : batch_vec) {
            for (const std::string& type : type_vec) {
                if (type == "bf16") {
                    for (const int tp : tp_vec) {
                        fire<__nv_bfloat16>(tp, batch, seed);
                    }
                }
                else if (type == "fp16") {
                    for (const int tp : tp_vec) {
                        fire<half>(tp, batch, seed);
                    }
                }
            }
        }
    }
    return 0;
}
