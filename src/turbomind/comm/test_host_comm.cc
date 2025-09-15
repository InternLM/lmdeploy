
#include <iostream>
#include <numeric>
#include <thread>

#include "src/turbomind/comm/host_comm.h"

using namespace turbomind;
using namespace turbomind::comm;

int main(int argc, char* argv[])
{
    const int                    N        = 32;
    std::unique_ptr<HostGroupId> group_id = CreateHostGroupId({});
    group_id->Initialize();
    std::vector<std::thread> threads;
    for (int r = 0; r < N; ++r) {
        threads.emplace_back([&, r] {
            HostComm world = group_id->CreateCommunicator(N, r);

            HostComm group = world;
            group          = world->Split(r / (N / 4), 0);

            auto tick = std::chrono::steady_clock::now();

            // int data = 100;
            // for (int i = 0; i < 10000; ++i, ++data) {
            //     group->Sync(true);
            // }

            volatile int a;
            volatile int b;
            for (int i = 0; i < 1; ++i) {
                a      = AllReduce(group, r, RedOp::kSum);
                auto v = AllGather(group, r);
                b      = std::accumulate(v.begin(), v.end(), 0);
                for (int j = 0; j < N; ++j) {
                    world->Sync();
                    if (j == r) {
                        std::cout << a << " " << b << std::endl;
                    }
                }
            }

            auto tock = std::chrono::steady_clock::now();

            for (int i = 0; i < N; ++i) {
                world->Sync();
                if (i == r) {
                    std::cout << std::chrono::duration<float, std::milli>(tock - tick).count() << std::endl;
                }
            }
        });
    }

    std::cout << "main thread waiting.\n";

    for (auto& t : threads) {
        t.join();
    }

    return 0;
}
