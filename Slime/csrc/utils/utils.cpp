#include "infiniband/verbs.h"

#include "utils/logging.h"
#include "utils/utils.h"

namespace slime {
std::vector<std::string> avaliable_nic()
{
    int                 num_devices;
    struct ibv_device** dev_list;

    dev_list = ibv_get_device_list(&num_devices);
    if (!dev_list) {
        SLIME_LOG_DEBUG("No RDMA devices");
        return {};
    }

    std::vector<std::string> avaliable_devices;
    for (int i = 0; i < num_devices; ++i) {
        avaliable_devices.push_back((char*)ibv_get_device_name(dev_list[i]));
    }
    return avaliable_devices;
}
}  // namespace slime