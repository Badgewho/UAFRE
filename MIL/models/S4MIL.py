# This code is taken from the original S4 repository https://github.com/HazyResearch/state-spaces
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import opt_einsum as oe

_c2r = torch.view_as_real
_r2c = torch.view_as_complex

def _nan_to_num_complex(x: torch.Tensor) -> torch.Tensor:
    # torch.nan_to_num 对 complex 有时不完全稳，这里转成 real-view 做一次
    if torch.is_complex(x):
        xr = torch.view_as_real(x)  # (..., 2)
        xr = torch.nan_to_num(xr, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.view_as_complex(xr)
    else:
        return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

def rfft_safe(x: torch.Tensor, n: int) -> torch.Tensor:
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = x.contiguous().to(torch.float32)
    try:
        return torch.fft.rfft(x, n=n)
    except RuntimeError as e:
        if "cuFFT" in str(e) or "CUFFT" in str(e):
            return torch.fft.rfft(x.detach().cpu(), n=n).to(x.device)
        raise

def irfft_safe(x: torch.Tensor, n: int) -> torch.Tensor:
    x = _nan_to_num_complex(x)
    x = x.contiguous()
    # 保证 complex64（对应 float32 的频域）
    if x.dtype != torch.complex64:
        x = x.to(torch.complex64)
    try:
        return torch.fft.irfft(x, n=n)
    except RuntimeError as e:
        if "cuFFT" in str(e) or "CUFFT" in str(e):
            return torch.fft.irfft(x.detach().cpu(), n=n).to(x.device)
        raise
class DropoutNd(nn.Module):
    def __init__(self, p: float = 0.5, tie=True, transposed=True):
        """
        tie: tie dropout mask across sequence lengths (Dropout1d/2d/3d)
        """
        super().__init__()
        if p < 0 or p >= 1:
            raise ValueError(
                "dropout probability has to be in [0, 1), " "but got {}".format(p))
        self.p = p
        self.tie = tie
        self.transposed = transposed
        self.binomial = torch.distributions.binomial.Binomial(probs=1-self.p)

    def forward(self, X):
        """ X: (batch, dim, lengths...) """
        if self.training:
            if not self.transposed:
                X = rearrange(X, 'b d ... -> b ... d')
            # binomial = torch.distributions.binomial.Binomial(probs=1-self.p) # This is incredibly slow
            mask_shape = X.shape[:2] + (1,)*(X.ndim-2) if self.tie else X.shape
            # mask = self.binomial.sample(mask_shape)
            mask = torch.rand(*mask_shape, device=X.device) < 1.-self.p
            X = X * mask * (1.0/(1-self.p))
            if not self.transposed:
                X = rearrange(X, 'b ... d -> b d ...')
            return X
        return X


class S4DKernel(nn.Module):
    """Wrapper around SSKernelDiag that generates the diagonal SSM parameters
    """

    def __init__(self, d_model, N=64, dt_min=0.001, dt_max=0.1, lr=None):
        super().__init__()
        # Generate dt
        H = d_model
        log_dt = torch.rand(H) * (
            math.log(dt_max) - math.log(dt_min)
        ) + math.log(dt_min)

        C = torch.randn(H, N // 2, dtype=torch.cfloat)
        self.C = nn.Parameter(_c2r(C))
        self.register("log_dt", log_dt, lr)

        log_A_real = torch.log(0.5 * torch.ones(H, N//2))
        A_imag = math.pi * repeat(torch.arange(N//2), 'n -> h n', h=H)
        self.register("log_A_real", log_A_real, lr)
        self.register("A_imag", A_imag, lr)

    def forward(self, L):
        """
        returns: (..., c, L) where c is number of channels (default 1)
        """

        # Materialize parameters
        dt = torch.exp(self.log_dt)  # (H)
        C = _r2c(self.C)  # (H N)
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag  # (H N)

        # Vandermonde multiplication
        dtA = A * dt.unsqueeze(-1)  # (H N)
        K = dtA.unsqueeze(-1) * torch.arange(L, device=A.device)  # (H N L)
        C = C * (torch.exp(dtA)-1.) / A
        K = 2 * torch.einsum('hn, hnl -> hl', C, torch.exp(K)).real

        return K

    def register(self, name, tensor, lr=None):
        """Register a tensor with a configurable learning rate and 0 weight decay"""

        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))

            optim = {"weight_decay": 0.0}
            if lr is not None:
                optim["lr"] = lr
            setattr(getattr(self, name), "_optim", optim)


class S4D(nn.Module):

    def __init__(self, d_model, d_state=64, dropout=0.0, transposed=True, **kernel_args):
        super().__init__()

        self.h = d_model
        self.n = d_state
        self.d_output = self.h
        self.transposed = transposed

        self.D = nn.Parameter(torch.randn(self.h))

        # SSM Kernel
        self.kernel = S4DKernel(self.h, N=self.n, **kernel_args)

        # Pointwise
        self.activation = nn.GELU()
        # dropout_fn = nn.Dropout2d # NOTE: bugged in PyTorch 1.11
        dropout_fn = DropoutNd
        self.dropout = dropout_fn(dropout) if dropout > 0.0 else nn.Identity()

        # position-wise output transform to mix features
        self.output_linear = nn.Sequential(
            nn.Conv1d(self.h, 2*self.h, kernel_size=1),
            nn.GLU(dim=-2),
        )

    def forward(self, u, **kwargs):  # absorbs return_output and transformer src mask
        """ Input and output shape (B, H, L) """
        if not self.transposed:
            u = u.transpose(-1, -2)
        L = u.size(-1)

        # Compute SSM Kernel
        k = self.kernel(L=L)  # (H L)

        k = k.contiguous()
        k = torch.nan_to_num(k, nan=0.0, posinf=0.0, neginf=0.0)

        k_f = rfft_safe(k, n=2*L)
        u_f = rfft_safe(u, n=2*L)
        k_f = _nan_to_num_complex(k_f).contiguous()
        u_f = _nan_to_num_complex(u_f).contiguous()
        prod = _nan_to_num_complex((u_f * k_f)).contiguous()
        y = irfft_safe(prod, n=2*L)[..., :L]  # (B H L)
        y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).contiguous()
        if y.is_cuda:
            torch.cuda.synchronize()
        # Compute D term in state space equation - essentially a skip connection
        y = y + u * self.D.unsqueeze(-1)

        y = self.dropout(self.activation(y))
        y = self.output_linear(y)
        if not self.transposed:
            y = y.transpose(-1, -2)
        return y


class S4Model(nn.Module):
    def __init__(self, in_dim, n_classes, dropout, act, survival = False):
        super(S4Model, self).__init__()
        self.n_classes = n_classes
        self._fc1 = [nn.Linear(in_dim, 512)]
        if act.lower() == 'relu':
            self._fc1 += [nn.ReLU()]
        elif act.lower() == 'gelu':
            self._fc1 += [nn.GELU()]
        if dropout:
            self._fc1 += [nn.Dropout(dropout)]
            print("dropout: ", dropout)
        self._fc1 = nn.Sequential(*self._fc1)
        self.s4_block = nn.Sequential(nn.LayerNorm(512),
                                      S4D(d_model=512, d_state=32, transposed=False))

        self.classifier = nn.Linear(512, self.n_classes)
        self.survival = survival
    def forward(self, x):
        x = x.unsqueeze(0)
        # print(x.shape)
        x = self._fc1(x)
        x = self.s4_block(x)
        x = torch.max(x, axis=1).values
        # print(x.shape)
        logits = self.classifier(x)
        Y_prob = F.softmax(logits, dim=1)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        A_raw = None
        results_dict = None
        if self.survival:
            Y_hat = torch.topk(logits, 1, dim = 1)[1]
            hazards = torch.sigmoid(logits)
            S = torch.cumprod(1 - hazards, dim=1)
            return hazards, S, Y_hat, None, None
        return logits, Y_prob, Y_hat, A_raw, results_dict


    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._fc1 = self._fc1.to(device)
        self.s4_block  = self.s4_block .to(device)
        self.classifier = self.classifier.to(device)
        
if __name__ == "__main__":
    data = torch.randn((6000, 1536))
    data.to('cuda')
    # model1 = TransMIL_l_v2(input_dim = 1024, layer =4, n_classes = 4, act = 'gelu', dropout = True)
    model = S4Model(in_dim = 1536, n_classes = 4, act = 'gelu', dropout = 0.25)
    print(model)
    results_dict = model(data)
    print(results_dict)