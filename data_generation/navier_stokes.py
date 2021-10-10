import torch
import math
from utils import GaussianRF, squeeze_result


@squeeze_result
def navier_stokes_2d(idx, batch_size=20, s=256, num_steps=100, T=50.0, device='cpu', **kwargs):
    """

    :param idx: batch number
    :param batch_size: batch size
    :param s: spatial resolution
    :param num_steps: number of time points in which solution computed
    :param T: end time
    :param device:
    :return: batch of solution of navier-stocks equatations
    """

    # Set up 2d GRF with covariance parameters
    GRF = GaussianRF(2, s, alpha=2.5, tau=7, device=device)

    # Forcing function: 0.1*(sin(2pi(x+y)) + cos(2pi(x+y)))
    t = torch.linspace(0, 1, s + 1, device=device)
    t = t[0:-1]

    X, Y = torch.meshgrid(t, t)
    f = 0.1 * (torch.sin(2 * math.pi * (X + Y)) + torch.cos(2 * math.pi * (X + Y)))
    w0 = GRF.sample(batch_size)
    sol, _ = navier_stokes_2d_(w0, f, 1e-3, T, 1e-4, num_steps)

    return w0.cpu().numpy(), sol.cpu().numpy()


def navier_stokes_2d_(w0, f, visc, T, delta_t=1e-4, record_steps=1):

    # Grid size - must be power of 2
    N = w0.size()[-1]

    # Maximum frequency
    k_max = math.floor(N/2.0)

    # Number of steps to final time
    steps = math.ceil(T/delta_t)

    # Initial vorticity to Fourier space
    w_h = torch.rfft(w0, 2, normalized=False, onesided=False)

    # Forcing to Fourier space
    f_h = torch.rfft(f, 2, normalized=False, onesided=False)

    # If same forcing for the whole batch
    if len(f_h.size()) < len(w_h.size()):
        f_h = torch.unsqueeze(f_h, 0)

    # Record solution every this number of steps
    record_time = math.floor(steps/record_steps)

    # Wavenumbers in y-direction
    k_y = torch.cat((torch.arange(start=0, end=k_max, step=1, device=w0.device), torch.arange(start=-k_max, end=0, step=1, device=w0.device)), 0).repeat(N,1)
    # Wavenumbers in x-direction
    k_x = k_y.transpose(0,1)
    # Negative Laplacian in Fourier space
    lap = 4 * (math.pi ** 2) * (k_x ** 2 + k_y ** 2)
    lap[0, 0] = 1.0
    # Dealiasing mask
    dealias = torch.unsqueeze(torch.logical_and(torch.abs(k_y) <= (2.0/3.0)*k_max, torch.abs(k_x) <= (2.0/3.0)*k_max).float(), 0)

    # Saving solution and time
    sol = torch.zeros(*w0.size(), record_steps, device=w0.device)
    sol_t = torch.zeros(record_steps, device=w0.device)

    # Record counter
    c = 0
    # Physical time
    t = 0.0
    for j in range(steps):
        # Stream function in Fourier space: solve Poisson equation
        psi_h = w_h.clone()
        psi_h[..., 0] = psi_h[..., 0] / lap
        psi_h[..., 1] = psi_h[..., 1] / lap

        # Velocity field in x-direction = psi_y
        q = psi_h.clone()
        temp = q[..., 0].clone()
        q[..., 0] = -2 * math.pi * k_y * q[..., 1]
        q[..., 1] = 2 * math.pi * k_y*temp
        q = torch.irfft(q, 2, normalized=False, onesided=False, signal_sizes=(N,N))

        # Velocity field in y-direction = -psi_x
        v = psi_h.clone()
        temp = v[..., 0].clone()
        v[..., 0] = 2 * math.pi*k_x*v[..., 1]
        v[..., 1] = -2 * math.pi*k_x*temp
        v = torch.irfft(v, 2, normalized=False, onesided=False, signal_sizes=(N, N))

        # Partial x of vorticity
        w_x = w_h.clone()
        temp = w_x[..., 0].clone()
        w_x[..., 0] = -2 * math.pi*k_x*w_x[..., 1]
        w_x[..., 1] = 2 * math.pi*k_x*temp
        w_x = torch.irfft(w_x, 2, normalized=False, onesided=False, signal_sizes=(N, N))

        # Partial y of vorticity
        w_y = w_h.clone()
        temp = w_y[..., 0].clone()
        w_y[..., 0] = -2 * math.pi * k_y * w_y[..., 1]
        w_y[..., 1] = 2 * math.pi * k_y * temp
        w_y = torch.irfft(w_y, 2, normalized=False, onesided=False, signal_sizes=(N, N))

        # Non-linear term (u.grad(w)): compute in physical space then back to Fourier space
        F_h = torch.rfft(q*w_x + v*w_y, 2, normalized=False, onesided=False)

        # Dealias
        F_h[..., 0] = dealias * F_h[..., 0]
        F_h[..., 1] = dealias * F_h[..., 1]

        # Cranck-Nicholson update
        w_h[..., 0] = (-delta_t * F_h[..., 0] + delta_t * f_h[..., 0] + (1.0 - 0.5 * delta_t*visc*lap) * w_h[..., 0]) / (1.0 + 0.5 * delta_t * visc * lap)
        w_h[..., 1] = (-delta_t * F_h[..., 1] + delta_t*f_h[..., 1] + (1.0 - 0.5 * delta_t*visc*lap) * w_h[..., 1]) / (1.0 + 0.5 * delta_t*visc*lap)

        # Update real time (used only for recording)
        t += delta_t

        if (j + 1) % record_time == 0:
            # Solution in physical space
            w = torch.irfft(w_h, 2, normalized=False, onesided=False, signal_sizes=(N, N))

            # Record solution and time
            sol[..., c] = w
            sol_t[c] = t

            c += 1

    return sol
