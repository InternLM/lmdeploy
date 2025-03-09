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

#include <string>

#include "lmdeploy_adaptor.h"

#include <cuda_runtime.h>
#include <numa.h>

#include "transfer_engine.h"
#include "transport/transport.h"

int LMDeployAdaptor::initialize(const std::string metadata_server, const std::string local_server_name)
{
    engine_            = std::make_unique<TransferEngine>(true);
    auto hostname_port = parseHostNameWithPort(local_server_name);
    engine_->init(metadata_server, local_server_name, hostname_port.first, hostname_port.second);
    return 0;
}
