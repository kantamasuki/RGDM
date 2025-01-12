import numpy as np
import torch
from scipy.fftpack import dct as dct
from torch_geometric.transforms import BaseTransform
from functools import lru_cache


class DDPM_DiffusionTrainer(BaseTransform):
    def __init__(self, args):
        # Hyperparameters regarding the noise schedule
        self.T32 = args.T32
        self.tau = args.tau
        
        self.args = args
        
    def __call__(self, data):
        if data.skip: return data
        
        # seqlen is the length of a protein chain
        seqlen = data['resi'].num_nodes
        # TN is the generation step
        TN = self.T32 + int(np.ceil(self.tau * np.log(seqlen/32)))
        # t is the diffusion step
        t = np.random.randint(0, TN)
        step = t/TN
        
        # Make noisy data (=pos) and noise (=pred)
        # We choose i_cutoff=seqlen because 
        # \phi_tk[i_cutoff:, :] is set to be zero in the RG projection
        # (The RG projection with i_cutoff=seqlen does not do anything) 
        pos, ans = self.add_noise(data['resi'].pos, seqlen, t, TN)
        i_cutoff = seqlen
        
        # Save these informations to 'data'
        # The class of 'data' is the torch_geometric.data.HeteroData
        data.TN = TN
        data.step, data.t, data.i_cutoff = step, t, i_cutoff
        data['resi'].node_t = torch.ones(seqlen) * t
        data['resi'].pos = torch.from_numpy(pos)
        data.score = torch.from_numpy(ans)
        
        return data
        
    def add_noise(self, pos, seqlen, t, TN):
        # Make noise schedule
        betaTN = self.betaTN_find(seqlen)
        _, _, ab, bb = self.noise_make(0.0001, betaTN, TN, t)

        # Make phi_t and noise
        noise = np.random.normal(loc=0.0, scale=1.0, size=pos.shape)
        pos = np.sqrt(ab) * pos + np.sqrt(bb) * noise
        ans = noise
        
        # Make phi_{tk=0} = 0
        pos -= pos.mean(axis=0)
        ans -= ans.mean(axis=0)
        
        return pos, ans
    
    @lru_cache(maxsize=None)
    def betaTN_find(self, N):
        T32 = self.args.T32
        tau = self.args.tau
        
        # Calculate the SNR (=vTN) at t=T(N)
        vT32 = 4.0359926511684205e-05  # =SNR_calc(0.0001, 0.02, 1000)
        vTN = vT32 * (32/N)**2
        # Find betaT by binary search
        eps = vTN / 10000
        TN = T32 + int(np.ceil(tau * np.log(N/32)))
        beta1 = 0.0001
        # Assert that betaT_min < betaT (truth) < betaT_max
        betaT_min = 0.06
        betaT_max = 0.24
        while self.SNR_calc(beta1, betaT_max, TN) > vTN:
            betaT_max += 0.02
        while self.SNR_calc(beta1, betaT_min, TN) < vTN:
            betaT_min *= 0.5
        # Find betaT
        betaT_new = betaT_max
        vT_new = self.SNR_calc(beta1, betaT_new, TN)
        while np.abs(vTN-vT_new) > eps:
            if vT_new > vTN:
                betaT_min = betaT_new
            else:
                betaT_max = betaT_new
            betaT_new = (betaT_min + betaT_max) / 2.0
            vT_new = self.SNR_calc(beta1, betaT_new, TN)
        betaT = betaT_new
        return betaT
    
    def SNR_calc(self, beta1, betaT, T):
        betas = np.linspace(beta1, betaT, T)
        alphas = 1. - betas
        alphas_bar = np.cumprod(alphas)
        betas_bar = 1. - alphas_bar
        vT = alphas_bar[T-1] / betas_bar[T-1]
        return vT
    
    def noise_make(self, beta1, betaT, T, t):
        betas = np.linspace(beta1, betaT, T)
        alphas = 1. - betas
        alphas_bar = np.cumprod(alphas)
        betas_bar = 1. - alphas_bar
        return alphas[t], betas[t], alphas_bar[t], betas_bar[t]


class DDPM_DiffusionSampler(torch.nn.Module):
    def __init__(self, args, seqlen):
        super().__init__()
        
        # Hyperparameters regarding the noise schedule
        self.T32 = args.T32
        self.tau = args.tau

        self.args = args
        
        # seqlen is the length of a protein chain
        self.seqlen = seqlen
        # TN is the generation step
        self.TN = TN = self.T32 + int(np.ceil(self.tau * np.log(seqlen/32)))
        
        # make noise schedules and save parameters
        # We note that i_cutoff is related to the RG projection in the neural network.
        # Since we set i_cutoff=seqlen, the RG projection does nothing.
        sqrt_recip_alphas_bar, coef_pred_x0, coef1, coef2, sqrt_betas,\
            sqrt_alphas, coef_elbo\
                = self.denoise_schedule(seqlen, TN)
        i_cutoff = torch.tensor([seqlen])
        assert sqrt_recip_alphas_bar.shape == (TN,)
        assert coef_pred_x0.shape == (TN,)
        assert coef1.shape == (TN,)
        assert coef2.shape == (TN,)
        assert sqrt_betas.shape == (TN,)
        assert sqrt_alphas.shape == (TN,)
        assert coef_elbo.shape == (TN,)
        self.register_buffer('sqrt_recip_alphas_bar', sqrt_recip_alphas_bar)
        self.register_buffer('coef_pred_x0', coef_pred_x0)
        self.register_buffer('coef1', coef1)
        self.register_buffer('coef2', coef2)
        self.register_buffer('sqrt_betas', sqrt_betas)
        self.register_buffer('sqrt_alphas', sqrt_alphas)
        self.register_buffer('coef_elbo', coef_elbo)
        self.register_buffer('i_cutoff', i_cutoff)

    @torch.no_grad()
    def denoise_schedule(self, seqlen, TN):
        # make betaTN
        beta1 = 0.0001
        betaTN = self.betaTN_find(seqlen)
        # make a, b, ab, bb
        b = torch.linspace(beta1, betaTN, TN)
        a = 1. - b
        ab = torch.cumprod(a, dim=0)
        bb = 1. - ab
        # make other parameters
        sqb = torch.exp(0.5*torch.log(b))
        ab_prev = torch.cat((torch.tensor([1.]), ab[:-1]))
        bb_prev = torch.cat((torch.tensor([0.]), bb[:-1]))
        # make coefficients
        sqrt_recip_alphas_bar = torch.exp(-0.5*torch.log(ab))
        coef_pred_x0 = torch.exp(0.5*torch.log(bb)) * sqrt_recip_alphas_bar
        coef1 = torch.exp(0.5*torch.log(ab_prev)) * b / bb
        coef2 = torch.exp(0.5*torch.log(a)) * bb_prev / bb
        sqrt_alphas = torch.exp(0.5 * torch.log(a))
        coef_elbo = torch.exp(0.5 * torch.log(b/a/bb))
                
        return sqrt_recip_alphas_bar, coef_pred_x0, coef1, coef2, sqb,\
            sqrt_alphas, coef_elbo

    @torch.no_grad()
    def forward(self, model, data, elbo):
        # make phi_t at t=TN
        pos = torch.randn_like(data['resi'].pos)
        
        for t in reversed(range(self.TN)):
            pos = self.q_mean(t, model, data, pos)
        
        # In our paper, we set elbo=False
        if elbo is True:
             lnP = self.lnP_calc(model, data, pos)
             return pos.cpu().numpy(), lnP.cpu().numpy()
        else:
            return pos.cpu().numpy(), 0
    
    @torch.no_grad()
    def q_mean(self, t, model, data, pos):
        pos -= pos.mean(dim=0)
        # predict noise
        data['resi'].pos = pos
        data['resi'].node_t = torch.ones(self.seqlen, device=pos.device) * t
        data.i_cutoff = self.i_cutoff
        pred = model(data)
        # calculate the mean of q(x{t-1}|x0, xt)
        pos_0 = pos * self.sqrt_recip_alphas_bar[t] \
            - data.pred * self.coef_pred_x0[t]
        mean = pos_0 * self.coef1[t] + pos * self.coef2[t]
        # add noise
        if t > 0:
            noise = self.sqrt_betas[t] * torch.randn_like(pos)
            pos = mean + noise
        else:
            pos = mean
        return pos

    # In our paper, we did not use lnP_calc, which estimates log-likelihood.
    @torch.no_grad()
    def lnP_calc(self, model, data, pos):
        data['resi'].pos = pos
        
        lnP = 0.0
        for t in range(self.TN):
            x_tm1 = data['resi'].pos
            noise = torch.randn_like(x_tm1)
            x_t = self.sqrt_alphas[t] * x_tm1 + self.sqrt_betas[t] * noise
            x_t -= x_t.mean(dim=0)
            # 予測を行う
            data['resi'].pos = x_t
            data['resi'].node_t = torch.ones(self.seqlen, device=x_t.device) * t
            data.i_cutoff = self.i_cutoff
            pred = model(data)
            # ln P を計算する
            lnP += 0.5 * torch.sum(noise**2)
            lnP -= 0.5 * torch.sum(
                (noise/self.sqrt_alphas[t] - 
                 self.coef_elbo[t] * data.pred)**2
            )
        lnP -= 0.5 * torch.sum((data['resi'].pos)**2)
        return lnP

    def betaTN_find(self, N):
        T32 = self.args.T32
        tau = self.args.tau

        vT32 = 4.0359926511684205e-05
        vTN = vT32 * (32/N)**2
        eps = vTN / 10000
        TN = T32 + int(np.ceil(tau * np.log(N/32)))

        beta1 = 0.0001
        betaT_min = 0.06
        betaT_max = 0.12
        while self.SNR_calc(beta1, betaT_max, TN) > vTN:
            betaT_max += 0.02
        while self.SNR_calc(beta1, betaT_min, TN) < vTN:
            betaT_min *= 0.5

        betaT_new = betaT_max
        vT_new = self.SNR_calc(beta1, betaT_new, TN)
        while np.abs(vTN-vT_new) > eps:
            if vT_new > vTN:
                betaT_min = betaT_new
            else:
                betaT_max = betaT_new
            betaT_new = (betaT_min + betaT_max) / 2.0
            vT_new = self.SNR_calc(beta1, betaT_new, TN)
        betaT = betaT_new
        return betaT
    
    def SNR_calc(self, beta1, betaT, T):
        betas = np.linspace(beta1, betaT, T)
        alphas = 1. - betas
        alphas_bar = np.cumprod(alphas)
        betas_bar = 1. - alphas_bar
        vT = alphas_bar[T-1] / betas_bar[T-1]
        return vT
