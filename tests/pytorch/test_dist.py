import unittest

import torch

from lmdeploy.pytorch.dist import (get_rank, master_only,
                                   master_only_and_broadcast_general,
                                   master_only_and_broadcast_tensor)


class SimpleTest(unittest.TestCase):

    @master_only
    def fake_input(self):
        print(f'Evaluate fake input 1 on {get_rank()}')
        return 'master only or none'

    @master_only_and_broadcast_general
    def fake_input21(self):
        print(f'Evaluate fake input 21 (str) on {get_rank()}')
        return 'master only and_broadcast'

    @master_only_and_broadcast_general
    def fake_input22(self):
        print(f'Evaluate fake input 22 (cpu tensor) on {get_rank()}')
        return torch.tensor([6, 66, 666])

    @master_only_and_broadcast_tensor
    def fake_input3(self):
        print(f'Evaluate fake input 3 (gpu tensor) on {get_rank()}')
        return torch.tensor([6, 66, 666]).cuda()

    def test(self):
        torch.distributed.init_process_group(backend='nccl')
        rank = get_rank()
        # unittest will discard --local_rank, thus set manually
        torch.cuda.set_device(rank)

        in1 = self.fake_input()
        in21 = self.fake_input21()
        in22 = self.fake_input22()
        in3 = self.fake_input3(dtype=torch.long, size=(1, 3))

        if rank == 0:
            self.assertEqual(in1, 'master only or none')
        else:
            self.assertEqual(in1, None)

        self.assertEqual(in21, 'master only and_broadcast')
        self.assertTrue(torch.allclose(in22, torch.tensor([6, 66, 666])))
        self.assertFalse(torch.allclose(in3.cpu(), torch.tensor([6, 6, 666])))
        self.assertTrue(torch.allclose(in3.cpu(), torch.tensor([6, 66, 666])))


if __name__ == '__main__':
    unittest.main()
