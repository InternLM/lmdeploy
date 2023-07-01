/*
 * Copyright (c) OpenMMLab. All rights reserved.
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Modified from https://github.com/NVIDIA/FasterTransformer/blob/main/examples/cpp/multi_gpu_gpt/multi_gpu_gpt_triton_example.cc

#include "3rdparty/INIReader.h"
#include <memory>
#include <thread>

#include "src/turbomind/triton_backend/llama/LlamaTritonModel.h"
#include "src/turbomind/triton_backend/llama/LlamaTritonModelInstance.h"
#include "src/turbomind/triton_backend/transformer_triton_backend.hpp"
#include "src/turbomind/utils/custom_ar_comm.h"
#include "src/turbomind/utils/mpi_utils.h"
#include "src/turbomind/utils/nccl_utils.h"
#include "src/turbomind/utils/nvtx_utils.h"
#include "src/turbomind/utils/word_list.h"

namespace ft = fastertransformer;

constexpr const bool kUSE_MPI = true;

struct RequestParam {
    int                    beam_width;
    int                    request_output_len;
    float                  beam_search_diversity_rate;
    uint                   runtime_top_k;
    float                  runtime_top_p;
    float                  temperature;
    float                  len_penalty;
    float                  repetition_penalty;
    float                  presence_penalty;
    int                    min_length;
    unsigned long long int random_seed;
    int                    start_id;
    int                    end_id;
};

std::vector<std::shared_ptr<std::unordered_map<std::string, triton::Tensor>>>
broadCastRequest(const std::vector<int>& v_start_ids,
                 const std::vector<int>& v_start_lengths,
                 const std::vector<int>& v_bad_words,
                 const int               node_id,
                 const int               gpu_count,
                 const RequestParam      param,
                 std::vector<void*>*     pointer_record)
{
    // broadcast the request to all nodes, and copy "gpu_count" copies on
    // different gpu
    int size_1         = v_start_ids.size();
    int size_2         = v_start_lengths.size();
    int size_bad_words = v_bad_words.size();
    if (kUSE_MPI) {
        ft::mpi::bcast(&size_1, 1, ft::mpi::MPI_TYPE_INT, 0, ft::mpi::COMM_WORLD);
        ft::mpi::bcast(&size_2, 1, ft::mpi::MPI_TYPE_INT, 0, ft::mpi::COMM_WORLD);
        ft::mpi::bcast(&size_bad_words, 1, ft::mpi::MPI_TYPE_INT, 0, ft::mpi::COMM_WORLD);
    }

    std::vector<int> v_input_ids(size_1);
    std::vector<int> v_input_lengths(size_2);
    std::vector<int> v_input_bad_words(size_bad_words);

    if (node_id == 0) {
        memcpy(v_input_ids.data(), v_start_ids.data(), size_1 * sizeof(int));
        memcpy(v_input_lengths.data(), v_start_lengths.data(), size_2 * sizeof(int));
        memcpy(v_input_bad_words.data(), v_bad_words.data(), size_bad_words * sizeof(int));
    }
    if (kUSE_MPI) {
        ft::mpi::barrier();
    }

    int request_batch_size = size_2;
    int max_input_len      = size_1 / size_2;

    std::cerr << "request_batch_size=" << request_batch_size << " max_input_len=" << max_input_len << "\n";

    if (kUSE_MPI) {
        ft::mpi::bcast(v_input_ids.data(), size_1, ft::mpi::MPI_TYPE_INT, 0, ft::mpi::COMM_WORLD);
        ft::mpi::bcast(v_input_lengths.data(), size_2, ft::mpi::MPI_TYPE_INT, 0, ft::mpi::COMM_WORLD);
        ft::mpi::bcast(v_input_bad_words.data(), size_bad_words, ft::mpi::MPI_TYPE_INT, 0, ft::mpi::COMM_WORLD);
    }

    std::vector<std::shared_ptr<std::unordered_map<std::string, triton::Tensor>>> request_list;
    for (int device_id = 0; device_id < gpu_count; device_id++) {
        ft::check_cuda_error(cudaSetDevice(device_id));

        int* d_input_ids;
        // int* d_input_lengths;
        int* d_input_bad_words;

        if (max_input_len == 0) {
            // unconditional case, no input ids, so do nothing.
            d_input_ids = nullptr;
            // d_input_lengths = nullptr;
            max_input_len = 0;
        }
        else {
            // conditional case.
            ft::deviceMalloc(&d_input_ids, size_1, false);
            // ft::deviceMalloc(&d_input_lengths, size_2, false);
            ft::cudaH2Dcpy(d_input_ids, v_input_ids.data(), size_1);
            // ft::cudaH2Dcpy(d_input_lengths, v_input_lengths.data(), size_2);
        }

        if (!v_input_bad_words.empty()) {
            ft::deviceMalloc(&d_input_bad_words, size_bad_words, false);
            ft::cudaH2Dcpy(d_input_bad_words, v_input_bad_words.data(), size_bad_words);
        }
        else {
            d_input_bad_words = nullptr;
        }

        uint32_t* request_output_len_ptr = (uint32_t*)malloc(request_batch_size * sizeof(uint32_t));
        int*      input_lengths_ptr      = (int*)malloc(request_batch_size * sizeof(int));
        for (int i = 0; i < request_batch_size; i++) {
            request_output_len_ptr[i] = param.request_output_len;
            input_lengths_ptr[i]      = v_input_lengths[i];
        }

        int* start_ids_ptr = (int*)malloc(request_batch_size * sizeof(int));
        int* end_ids_ptr   = (int*)malloc(request_batch_size * sizeof(int));
        for (int i = 0; i < request_batch_size; i++) {
            start_ids_ptr[i] = param.start_id;
            end_ids_ptr[i]   = param.end_id;
        }
        pointer_record->push_back(start_ids_ptr);
        pointer_record->push_back(end_ids_ptr);

        request_list.push_back(std::shared_ptr<std::unordered_map<std::string, triton::Tensor>>(
            new std::unordered_map<std::string, triton::Tensor>{
                {"input_ids",
                 triton::Tensor{triton::MEMORY_GPU,
                                triton::TYPE_INT32,
                                std::vector<size_t>{(size_t)request_batch_size, (size_t)max_input_len},
                                d_input_ids}},
                {"input_lengths",
                 triton::Tensor{triton::MEMORY_CPU,
                                triton::TYPE_INT32,
                                std::vector<size_t>{(size_t)request_batch_size},
                                input_lengths_ptr}},
                {"request_output_len",
                 triton::Tensor{triton::MEMORY_CPU,
                                triton::TYPE_INT32,
                                std::vector<size_t>{(size_t)request_batch_size},
                                request_output_len_ptr}},
                {"bad_words_list",
                 triton::Tensor{
                     triton::MEMORY_GPU, triton::TYPE_INT32, {2, v_input_bad_words.size() / 2}, d_input_bad_words}},
                {"start_id",
                 triton::Tensor{triton::MEMORY_CPU, triton::TYPE_INT32, {(size_t)request_batch_size}, start_ids_ptr}},
                {"end_id",
                 triton::Tensor{triton::MEMORY_CPU, triton::TYPE_INT32, {(size_t)request_batch_size}, end_ids_ptr}}}));

        int* beam_width_ptr = new int(param.beam_width);
        pointer_record->push_back(beam_width_ptr);
        request_list[device_id]->insert(
            {"beam_width",
             triton::Tensor{triton::MEMORY_CPU, triton::TYPE_INT32, std::vector<size_t>{1}, beam_width_ptr}});
        if (param.beam_width > 1) {
            float* beam_search_diversity_rate_ptr = new float(param.beam_search_diversity_rate);
            pointer_record->push_back(beam_search_diversity_rate_ptr);
            request_list[device_id]->insert(
                {"beam_search_diversity_rate",
                 triton::Tensor{
                     triton::MEMORY_CPU, triton::TYPE_FP32, std::vector<size_t>{1}, beam_search_diversity_rate_ptr}});
        }
        else {
            if (param.runtime_top_p != 0.0f) {
                float* runtime_top_p_ptr = new float(param.runtime_top_p);
                pointer_record->push_back(runtime_top_p_ptr);
                request_list[device_id]->insert(
                    {"runtime_top_p",
                     triton::Tensor{triton::MEMORY_CPU, triton::TYPE_FP32, std::vector<size_t>{1}, runtime_top_p_ptr}});
            }
            if (param.runtime_top_k != 0) {
                uint* runtime_top_k_ptr = new uint(param.runtime_top_k);
                pointer_record->push_back(runtime_top_k_ptr);
                request_list[device_id]->insert(
                    {"runtime_top_k",
                     triton::Tensor{
                         triton::MEMORY_CPU, triton::TYPE_UINT32, std::vector<size_t>{1}, runtime_top_k_ptr}});
            }
        }
        float* temperature_ptr = new float(param.temperature);
        pointer_record->push_back(temperature_ptr);
        request_list[device_id]->insert(
            {"temperature",
             triton::Tensor{triton::MEMORY_CPU, triton::TYPE_FP32, std::vector<size_t>{1}, temperature_ptr}});
        float* len_penalty_ptr = new float(param.len_penalty);
        pointer_record->push_back(len_penalty_ptr);
        request_list[device_id]->insert(
            {"len_penalty",
             triton::Tensor{triton::MEMORY_CPU, triton::TYPE_FP32, std::vector<size_t>{1}, len_penalty_ptr}});
        if (param.repetition_penalty != 1.0f) {
            float* repetition_penalty_ptr = new float(param.repetition_penalty);
            pointer_record->push_back(repetition_penalty_ptr);
            request_list[device_id]->insert(
                {"repetition_penalty",
                 triton::Tensor{
                     triton::MEMORY_CPU, triton::TYPE_FP32, std::vector<size_t>{1}, repetition_penalty_ptr}});
        }
        if (param.presence_penalty != 0.0f) {
            float* presence_penalty_ptr = new float(param.presence_penalty);
            pointer_record->push_back(presence_penalty_ptr);
            request_list[device_id]->insert(
                {"presence_penalty",
                 triton::Tensor{triton::MEMORY_CPU, triton::TYPE_FP32, std::vector<size_t>{1}, presence_penalty_ptr}});
        }
        int* min_length_ptr = new int(param.min_length);
        pointer_record->push_back(min_length_ptr);
        request_list[device_id]->insert(
            {"min_length",
             triton::Tensor{triton::MEMORY_CPU, triton::TYPE_INT32, std::vector<size_t>{1}, min_length_ptr}});
        unsigned long long int* random_seed_ptr = new unsigned long long int(param.random_seed);
        pointer_record->push_back(random_seed_ptr);
        request_list[device_id]->insert(
            {"random_seed",
             triton::Tensor{triton::MEMORY_CPU, triton::TYPE_UINT64, std::vector<size_t>{1}, random_seed_ptr}});

        pointer_record->push_back(d_input_ids);
        // pointer_record->push_back(d_input_lengths);
        pointer_record->push_back(d_input_bad_words);
        pointer_record->push_back(request_output_len_ptr);
        pointer_record->push_back(input_lengths_ptr);
    }

    return request_list;
}

int read_start_ids(size_t            batch_size,
                   std::vector<int>* v_start_lengths,
                   std::vector<int>* v_start_ids,
                   size_t&           max_input_len,
                   const int         end_id,
                   const int         beam_width,
                   std::string       file_name);

std::vector<std::shared_ptr<std::unordered_map<std::string, triton::Tensor>>>
prepareRequest(std::string ini_name, const int node_id, const int gpu_count, std::vector<void*>* pointer_record)
{
    INIReader reader = INIReader(ini_name);
    if (reader.ParseError() < 0) {
        std::cout << "[ERROR] Can't load '" << ini_name << "'\n";
        ft::FT_CHECK(false);
    }

    const size_t request_batch_size = reader.GetInteger("request", "request_batch_size");
    std::cerr << "request_batch_size=" << request_batch_size << "\n";

    const int start_id = reader.GetInteger("request", "start_id");
    const int end_id   = reader.GetInteger("request", "end_id");

    std::vector<int> v_start_ids;
    std::vector<int> v_start_lengths;

    size_t max_input_len = 0;
    read_start_ids(request_batch_size,
                   &v_start_lengths,
                   &v_start_ids,
                   max_input_len,
                   end_id,
                   1,
                   "../examples/cpp/llama/start_ids.csv");
    // drop requests > request_batch_size
    if (v_start_lengths.size() > request_batch_size) {
        v_start_lengths.resize(request_batch_size);
        v_start_ids.resize(request_batch_size * max_input_len);
    }
    std::cerr << "max_input_len=" << max_input_len << "\n";

    std::vector<int> v_bad_words;
    // ft::read_word_list("../examples/cpp/llama/bad_words.csv", v_bad_words);

    RequestParam param;
    param.beam_width                 = reader.GetInteger("request", "beam_width");
    param.request_output_len         = reader.GetInteger("request", "request_output_len");
    param.beam_search_diversity_rate = reader.GetFloat("request", "beam_search_diversity_rate");
    param.runtime_top_k              = reader.GetInteger("request", "top_k");
    param.runtime_top_p              = reader.GetFloat("request", "top_p");
    param.temperature                = reader.GetFloat("request", "temperature");
    param.len_penalty                = reader.GetFloat("request", "len_penalty");
    param.repetition_penalty         = reader.GetFloat("request", "repetition_penalty", 1.0f);
    param.presence_penalty           = reader.GetFloat("request", "presence_penalty", 0.0f);
    param.min_length                 = reader.GetInteger("request", "min_length", 0);
    param.random_seed                = (unsigned long long int)0;
    param.start_id                   = start_id;
    param.end_id                     = end_id;

    auto request_list =
        broadCastRequest(v_start_ids, v_start_lengths, v_bad_words, node_id, gpu_count, param, pointer_record);
    return request_list;
}

int threadCreateModelInstances(std::shared_ptr<AbstractTransformerModel>                         model,
                               std::vector<std::unique_ptr<AbstractTransformerModelInstance>>*   model_instances,
                               const int                                                         device_id,
                               const int                                                         rank,
                               std::pair<std::vector<ft::NcclParam>, std::vector<ft::NcclParam>> nccl_params,
                               std::shared_ptr<ft::AbstractCustomComm> custom_all_reduce_comm = nullptr)
{
    printf("[INFO] rank = %d \n", rank);
    ft::check_cuda_error(cudaSetDevice(device_id));
    cudaStream_t stream;
    ft::check_cuda_error(cudaStreamCreate(&stream));
    model->createSharedWeights(device_id, rank);
    auto model_instance = model->createModelInstance(device_id, rank, stream, nccl_params, custom_all_reduce_comm);
    model_instances->at(device_id) = std::move(model_instance);
    printf("model instance %d is created \n", device_id);
    ft::print_mem_usage();
    return 0;
}

int threadForward(std::unique_ptr<AbstractTransformerModelInstance>*                model_instance,
                  std::shared_ptr<std::unordered_map<std::string, triton::Tensor>>  request,
                  std::shared_ptr<std::unordered_map<std::string, triton::Tensor>>* output_tensors,
                  const int                                                         device_id,
                  ft::AbstractInstanceComm*                                         comm)
{
    ft::check_cuda_error(cudaSetDevice(device_id));
    cudaDeviceSynchronize();
    *output_tensors = (*model_instance)->forward(request, comm);
    cudaDeviceSynchronize();
    return 0;
}

int main(int argc, char* argv[])
{
    /*
        Prepare the nccl ids, node id, device id and world size
        by MPI or triton
    */

    int node_id  = 0;
    int node_num = 1;

    if (kUSE_MPI) {
        ft::mpi::initialize(&argc, &argv);
        node_id  = ft::mpi::getCommWorldRank();
        node_num = ft::mpi::getCommWorldSize();
    }

    printf("node_id=%d node_num=%d\n", node_id, node_num);

    // Note: Only supports that all nodes have same gpu count
    const int   gpu_count  = ft::getDeviceCount();
    const int   world_size = node_num * gpu_count;
    std::string ini_name   = argc >= 2 ? std::string(argv[1]) : "../examples/cpp/llama/llama_config.ini";

    // step 1: Create model
    std::shared_ptr<AbstractTransformerModel> model              = AbstractTransformerModel::createLlamaModel(ini_name);
    int                                       tensor_para_size   = model->getTensorParaSize();
    int                                       pipeline_para_size = model->getPipelineParaSize();
    printf(
        "world_size=%d tensor_para_size=%d pipeline_para_size=%d\n", world_size, tensor_para_size, pipeline_para_size);
    FT_CHECK_WITH_INFO(world_size == (tensor_para_size * pipeline_para_size),
                       "World Size != Tensor Parallel Size * Pipeline Parallel Size !");

    std::cout << model->toString();

    // step 2: Initialize the NCCL
    std::pair<std::vector<ft::NcclParam>, std::vector<ft::NcclParam>> nccl_comms = model->createNcclParams(node_id);
    cudaDeviceSynchronize();

    // Optional Step: create custom all reduce comm
    // std::vector<std::shared_ptr<ft::AbstractCustomComm>>
    // custom_all_reduce_comms; model->createCustomComms(&custom_all_reduce_comms,
    // world_size);

    // step 2.1 create instance comm
    auto instance_comm = model->createInstanceComm(gpu_count);

    // step 3: Create model instances
    std::vector<std::unique_ptr<AbstractTransformerModelInstance>> model_instances((size_t)gpu_count);
    std::vector<std::thread>                                       threads;
    for (int device_id = 0; device_id < gpu_count; device_id++) {
        const int rank = node_id * gpu_count + device_id;
        threads.push_back(
            std::thread(threadCreateModelInstances, model, &model_instances, device_id, rank, nccl_comms, nullptr));
        //   custom_all_reduce_comms[rank]));
    }
    for (auto& t : threads) {
        t.join();
    }

    // step 4: prepare request
    std::vector<void*> pointer_record;  // Used to prevent the pointers are
                                        // release after leaving functions
    std::vector<std::shared_ptr<std::unordered_map<std::string, triton::Tensor>>> request_list =
        prepareRequest(ini_name, node_id, gpu_count, &pointer_record);
    printf("[INFO] request is created \n");

    // step 5: Forward
    std::vector<std::shared_ptr<std::unordered_map<std::string, triton::Tensor>>> output_tensors_lists(
        (size_t)gpu_count);
    for (int i = 0; i < 1; i++) {
        threads.clear();
        for (int device_id = 0; device_id < gpu_count; device_id++) {
            threads.push_back(std::thread(threadForward,
                                          &model_instances[device_id],
                                          request_list[device_id],
                                          &output_tensors_lists[device_id],
                                          device_id,
                                          instance_comm.get()));
        }
        for (auto& t : threads) {
            t.join();
        }
    }
    printf("[INFO] forward is completed. \n");

    const int* d_output_ids = (const int*)output_tensors_lists[0].get()->at("output_ids").data;
    const int* d_seq_lens   = (const int*)output_tensors_lists[0].get()->at("sequence_length").data;
    const int  batch_size   = output_tensors_lists[0].get()->at("output_ids").shape[0];
    const int  beam_width   = output_tensors_lists[0].get()->at("output_ids").shape[1];
    const int  seq_len      = output_tensors_lists[0].get()->at("output_ids").shape[2];
    // step 6: check results
    if (node_id == 0) {
        std::string fName   = "out";
        auto        outFile = std::ofstream(fName, std::ios::out);
        if (!outFile.is_open()) {
            printf("[WARNING] Cannot write results into output file %s \n", fName.c_str());
        }
        else {
            size_t outCount = batch_size * beam_width * seq_len;
            // int*   hBuf     = new int[outCount];
            std::vector<int> hBuf(outCount);
            ft::cudaD2Hcpy(hBuf.data(), d_output_ids, outCount);
            std::vector<int> seq_lens(batch_size);
            ft::cudaD2Hcpy(seq_lens.data(), d_seq_lens, batch_size);
            std::cout << "sequence length: ";
            for (int i = 0; i < batch_size; ++i) {
                std::cout << (i ? ", " : "") << seq_lens[i];
            }
            std::cout << "\n";
            {
                std::cout << "Writing " << outCount << " elements\n";
                int zeroCount = 0;
                for (size_t i = 0; i < outCount; i++) {
                    if (hBuf[i] == int(0))
                        zeroCount++;
                    outFile << hBuf[i] << " ";
                    if ((i + 1) % (seq_len) == 0)
                        outFile << std::endl;

                    if (i < 10)
                        printf("%5d ", hBuf[i]);
                    if ((i + 1) % (seq_len) == 0 && i < 10)
                        std::cout << std::endl;
                }
                std::cout << std::endl << "zeroCount = " << zeroCount << std::endl;
            }
        }
    }

    if (kUSE_MPI) {
        ft::mpi::barrier();
    }
    cudaDeviceSynchronize();

    if (1) {
        // test time
        struct timeval start, end;
        gettimeofday(&start, NULL);

        const int ite = 1;
        for (int i = 0; i < ite; i++) {
            threads.clear();
            for (int device_id = 0; device_id < gpu_count; device_id++) {
                threads.push_back(std::thread(threadForward,
                                              &model_instances[device_id],
                                              request_list[device_id],
                                              &output_tensors_lists[device_id],
                                              device_id,
                                              instance_comm.get()));
            }
            for (auto& t : threads) {
                t.join();
            }
        }

        cudaDeviceSynchronize();
        if (kUSE_MPI) {
            ft::mpi::barrier();
        }

        gettimeofday(&end, NULL);

        printf("[INFO] batch_size %d beam_width %d seq_len %d"
               " FT-CPP-GPT-Triton-time %.2f ms\n",
               batch_size,
               beam_width,
               seq_len,
               ((end.tv_sec - start.tv_sec) * 1000 + (end.tv_usec - start.tv_usec) * 0.001) / ite);
    }

    if (kUSE_MPI) {
        ft::mpi::finalize();
    }
    return 0;
}

int read_start_ids(size_t            batch_size,
                   std::vector<int>* v_start_lengths,
                   std::vector<int>* v_start_ids,
                   size_t&           max_input_len,
                   const int         end_id,
                   const int         beam_width,
                   std::string       file_name)
{
    std::vector<std::vector<int>> tmp_start_ids;
    std::vector<int>              tmp_start_lengths;

    std::ifstream start_id_file(file_name, std::ios::in);
    int           line_num = 0;
    if (start_id_file.is_open()) {
        std::string line;
        while (std::getline(start_id_file, line)) {
            std::stringstream lineStream(line);
            std::string       vals;
            int               i1 = 0;
            std::vector<int>  tmp_vec;
            while (std::getline(lineStream, vals, ',')) {
                tmp_vec.push_back(std::stoi(vals));
                i1++;
            }
            tmp_start_ids.push_back(tmp_vec);
            tmp_start_lengths.push_back(i1);
            line_num++;
        }
        if (batch_size == 0) {
            batch_size = line_num;
        }
    }
    else {
        printf("[WARNING] Cannot open the file '%s'. \n", file_name.c_str());
        max_input_len = 0;
        return 0;
    }

    max_input_len = tmp_start_lengths.data()[0];
    for (uint i = 1; i < (uint)tmp_start_lengths.size(); i++) {
        max_input_len = max_input_len > tmp_start_lengths.data()[i] ? max_input_len : tmp_start_lengths.data()[i];
    }

    while ((int)tmp_start_lengths.size() < batch_size) {
        std::vector<int> padding_ids;
        for (int i = 0; i < max_input_len; i++) {
            padding_ids.push_back(end_id);
        }
        tmp_start_ids.push_back(padding_ids);
        tmp_start_lengths.push_back(max_input_len);
    }

    // Add padding
    for (int i = 0; i < (int)tmp_start_ids.size(); i++) {
        for (int j = (int)tmp_start_ids[i].size(); j < max_input_len; j++) {
            tmp_start_ids[i].push_back(end_id);
        }
    }

    for (int i = 0; i < (int)tmp_start_ids.size(); i++) {
        for (int b = 0; b < beam_width; b++) {
            for (int j = 0; j < (int)tmp_start_ids[i].size(); j++) {
                v_start_ids->push_back(tmp_start_ids[i][j]);
            }
            v_start_lengths->push_back(tmp_start_lengths[i]);
        }
    }
    return batch_size;
}
