import torch
import torch.nn as nn
import numpy as np
import time
from torchdyn.core import NeuralODE

from .utils import get_free_gpu


class VectorField(nn.Module):
    def __init__(
        self, 
        dim, 
        hidden_list, 
        activation_fn=nn.ReLU(),
    ):
        super(VectorField, self).__init__()

        layer_dim_list = [dim + 1] + hidden_list + [dim]
        layers = []
        for l in range(len(layer_dim_list)-1):
            layers.append(nn.Linear(layer_dim_list[l], layer_dim_list[l+1]))
            if l < len(layer_dim_list)-2:
                layers.append(activation_fn)

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
    

class torch_wrapper(nn.Module):
    """Wraps model to torchdyn compatible format."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, t, x, *args, **kwargs):
        return self.model(torch.cat([x, t.repeat(x.shape[0])[:, None]], 1))
    

def sample_plan(X, Y, P):
    prob = P.flatten().detach().cpu().numpy()
    # prob = P.flatten()
    prob /= prob.sum()
    choices = np.random.choice(
            P.shape[0] * P.shape[1], p=prob, size=P.shape[0], replace=True
        )
    i, j = np.divmod(choices, P.shape[1])
    return X[i], Y[j]


def sample_plan_whole(X, Y, P, cdf):
    # Generate random samples and use CDF to find indices
    random_samples = np.random.rand(P.shape[0])
    indices = np.searchsorted(cdf, random_samples, side="left")

    # Map indices to the original 2D point indices
    i, j = np.divmod(indices, P.shape[1])

    return X[i], Y[j]


def sample_location_and_conditional_flow(X, Y):
    t_vector = torch.rand(X.shape[0]).type_as(X)
    t = t_vector.reshape(-1, *([1] * (X.dim() - 1)))
    X_t = t * Y + (1 - t) * X
    U_t = Y - X
    return t_vector, X_t, U_t


def train_flow_given_P(X, Y, P, max_iter, lr, model_save_name, hidden_dims=[256,256,256,256,256,256], model_ckpt=None, save_training_state=False, gpu_index=None):
    # Flatten the transport matrix and normalize to get the probability vector
    prob = P.flatten()
    prob /= prob.sum()
    # Compute cumulative distribution function (CDF)
    cdf = np.cumsum(prob)

    if gpu_index is None:
        gpu_index = get_free_gpu()
    print(f"Choose GPU:{gpu_index} as device")
    device = torch.device(f'cuda:{gpu_index}' if torch.cuda.is_available() else 'cpu')

    model = VectorField(dim=2, hidden_list=hidden_dims).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # 0.005
    loss_fn = torch.nn.MSELoss()
    
    if model_ckpt:
        checkpoint = torch.load(model_ckpt, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # return model

    start = time.time()
    for k in range(max_iter):
        optimizer.zero_grad()

        # start = time.time()
        X_sample, Y_sample = sample_plan_whole(X, Y, P, cdf)
        # print(f'sample plan: {time.time() - start}')

        # start = time.time()
        t, X_t, U_t = sample_location_and_conditional_flow(torch.tensor(X_sample).to(device).to(torch.float32), torch.tensor(Y_sample).to(device).to(torch.float32))
        # print(f'sample location and conditional flow: {time.time() - start}')

        V_t = model(torch.cat([X_t, t[:, None]], dim=-1))
        loss = loss_fn(V_t, U_t)

        loss.backward()
        optimizer.step()

        if k % 2000 == 0:
            end = time.time()
            print(f"iter {k} loss {loss.item()} time {end - start}")
            start = end
    
    if save_training_state:
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }
        torch.save(checkpoint, model_save_name)
    else:
        torch.save(model.state_dict(), model_save_name)
    return model


def flow_morphology(model_path, X, hidden_dims=[256,256,256,256,256,256], training_state=True):
    """
    Applies a trained neural ODE model to input data to obtain the morphologically aliged coordinates.

    Args:
        model_path (str): Path to the trained model file.
        X (numpy.ndarray or torch.Tensor): Input spatial coordinates to be non-linearly aligned.
        hidden_dims (list): List of integers specifying the architecture of the hidden layers of the model. Default is [256, 256, 256, 256, 256, 256].
        training_state (bool): If True, the saved input model contains its training state (optimizer state). If False, the model only has model weights. Default is True.

    Returns:
        tuple: A tuple containing two numpy arrays:
            - The initial state of the trajectory, same as X.
            - The final state of the trajectory, the X non-linearly aligned according to the model.
    """
    gpu_index = get_free_gpu()
    print(f"Choose GPU:{gpu_index} as device")
    device = torch.device(f'cuda:{gpu_index}' if torch.cuda.is_available() else 'cpu')

    model = VectorField(dim=2, hidden_list=hidden_dims).to(device)
    if training_state:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(torch.load(model_path))

    X = torch.tensor(X).to(device).to(torch.float32)

    n_ode = NeuralODE(torch_wrapper(model), solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4)
    t_span=torch.linspace(0, 1, 100)
    with torch.no_grad():
        traj = n_ode.trajectory(X, t_span)
    print(traj.shape)
    print(traj[-1, :].shape)
    return traj[0, :].detach().cpu().numpy(), traj[-1, :].detach().cpu().numpy()


def train_morphological_model(slice1, slice2, P, max_iter, model_save_name, hidden_dims=[256,256,256,256,256,256], lr=0.005, gpu_index=None):
    """
    Trains the nueral ODE model as the morphological alignment between physically aligned slice 1 and slice 2.

    param: slice1 - AnnData object of slice 1, already physically aligned with slice 2
    param: slice2 - AnnData object of slice 2
    param: P - the probabilistic mapping between slice 1 and slice 2, returned by the physical alignment step
    param: max_iter - the number of iterations to train the neural network
    param: model_save_name - the path to a location to save the trained model
    param: hidden_dims - the architecture of the hidden layers of the model, default [256,256,256,256,256,256], i.e. 6 hidden layers with 256 neurons each
    param: lr - learning rate, default 0.005
    param: gpu_index - the index of the gpu to use for training, if None, then choose the gpu with the lowest memory usage.

    return: trained_flow_model - the PyTorch model for the trained neural ODE.
    """
    X = slice1.obsm['spatial']
    Y = slice2.obsm['spatial']

    trained_flow_model = train_flow_given_P(X, Y, P, max_iter=max_iter, lr=lr, model_save_name=model_save_name, hidden_dims=hidden_dims, save_training_state=True, gpu_index=gpu_index)
    return trained_flow_model
