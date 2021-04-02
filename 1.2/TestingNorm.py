import unittest
import torch
import torch.nn as nn
from InstanceNorm import InstanceNorm2D
from BatchNorm import BatchNorm2D
from LayerNorm import LayerNorm2D


class NormTestCase(unittest.TestCase):
    def test_BatchNorm2D(self):
        m = BatchNorm2D(3)
        m_pytorch = nn.BatchNorm2d(3, affine = True)
        for i in range(1, 10):
            input = torch.randn(3, 3, i, i)
            output_pytorch = m_pytorch(input)
            output = m(input)
            print((output - output_pytorch).mean())
            self.assertTrue((output - output_pytorch).mean()< 1e-7)

    def test_InstanceNorm2D(self):
        m = InstanceNorm2D(3, rescale = True)
        m_pytorch = nn.InstanceNorm2d(3, affine = True)
        for i in range(1, 10):
            input = torch.randn(3, 3, i, i)
            output_pytorch = m_pytorch(input)
            output = m(input)
            print((output - output_pytorch).mean())
            self.assertTrue((output - output_pytorch).mean()< 1e-7)

    def test_LayerNorm2D(self):
        for i in range(1, 10):
            m = LayerNorm2D([3, i, i])
            m_pytorch = nn.LayerNorm([3, i, i])
            input = torch.randn(3, 3, i, i)
            output_pytorch = m_pytorch(input)
            output = m(input)
            print((output - output_pytorch).mean())
            self.assertTrue((output - output_pytorch).mean()< 1e-7)

if __name__ == '__main__':
    unittest.main()