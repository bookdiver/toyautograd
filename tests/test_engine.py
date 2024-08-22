import torch
import numpy as np
from src.engine import Scalar, Tensor

def test_scalar():
    # Home-made implementation
    x = Scalar(value=3.0)
    z = x * 2.0 + 3.0 
    y = (z / 4.0) ** 1.5
    z = y.tanh()
    z.backward()
    my_x, my_z = x, z
    # PyTorch implementation
    x = torch.tensor([3.0], dtype=torch.double, requires_grad=True)
    z = x * 2.0 + 3.0
    y = (z / 4.0) ** 1.5
    z = y.tanh() 
    z.backward()
    pt_x, pt_z = x, z
    
    tol = 1e-9
    # Check the values
    assert abs(my_z.value - pt_z.item()) < tol
    # Check the gradients
    assert abs(my_x.grad - pt_x.grad.item()) < tol

def test_tensor():
    # Home-made implementation
    x = Tensor(value=np.array([[1.0, 2.0],
                               [3.0, 1.0]]))
    y = Tensor(value=np.array([1.0, 2.0]))
    z = x @ y
    h = z.mean()
    h.backward()
    my_x, my_y, my_h = x, y, h
    # PyTorch implementation
    x = torch.tensor([[1.0, 2.0],
                      [3.0, 1.0]], 
                     dtype=torch.double,
                     requires_grad=True)
    y = torch.tensor([1.0, 2.0], 
                     dtype=torch.double,
                     requires_grad=True)
    z = torch.matmul(x, y)
    h = z.mean()
    h.backward()
    pt_x, pt_y, pt_h = x, y, h

    tol = 1e-6
    # Check the values
    assert abs(my_h.value - pt_h.item()) < tol
    # Check the gradients
    assert abs(my_x.grad - pt_x.grad) < tol
    
    