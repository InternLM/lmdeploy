// Copyright 2024 KVCache.AI
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <pybind11/pybind11.h>
#include <sys/time.h>

#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <memory>
#include <stack>
#include <string>
#include <vector>

#include "transfer_engine.h"
#include "transport/rdma_transport/rdma_transport.h"
#include "transport/transport.h"

using namespace mooncake;

class LMDeployAdaptor {
public:
    LMDeployAdaptor(){};

    ~LMDeployAdaptor(){};

    int initialize(const std::string metadata_server, const std::string local_server_name);

private:
    std::shared_ptr<TransferEngine> engine_;
};
