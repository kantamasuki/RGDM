import numpy as np
import torch
from scipy.fftpack import dct as dct
from torch_geometric.transforms import BaseTransform
import sys

class RGDiffusionTrainer(BaseTransform):
    def __init__(self, args):
        # hyperparameters regarding the noise schedule
        self.T32 = args.T32
        self.mass = args.mass
        self.args = args
    
    def __call__(self, data):
        if data.skip: return data
        
        # seqlen is the length of a protein chain
        seqlen = data['resi'].num_nodes
        
        # paramters for the forward diffusion
        Lam0, tau, T_rg, T_fl = RGDM_param_calculator(seqlen, self.T32)
        TN = T_rg + T_fl
        t = np.random.randint(0, TN)
        step = t/TN
        
        # add noise 
        pos, ans, shell_kmin, shell_kmax = self.add_noise(data['resi'].pos, seqlen, t, Lam0, tau, T_rg, T_fl)
    
        # save information to 'data'
        # (class of 'data' is torch_geometric.data.HeteroData)
        data.TN = TN
        data.step, data.t = step, t
        data.shell_kmin, data.shell_kmax = shell_kmin, shell_kmax
        data['resi'].node_t = torch.ones(seqlen) * t
        data['resi'].pos = torch.from_numpy(pos)
        data.score = torch.from_numpy(ans)
        
        return data
    
    def add_noise(self, pos, seqlen, t, Lam0, tau, T_rg, T_fl, c=(3.8)**2/2.3211):
        # make noise schedule *[k] at t=t
        ab, _, one_m_ab, _, inshell, befIO, _\
            = noise_schedule_for_training(seqlen, Lam0, tau, T_rg, T_fl)
        # data augumentation を加える
        if self.args.data_aug == 1:
            one_m_ab[:T_fl] = self.args.data_aug_ratio
        k_list = np.pi * np.arange(seqlen) / seqlen
        k_list[0] = k_list[1] # we avoid zero-division
        kmax = np.pi * (seqlen-1) / seqlen
        t_of_k = t + T_fl - (np.floor(tau*np.log(kmax/k_list))).astype(int)
        sq_ab_tk = np.sqrt(ab[t_of_k])
        sq_bb_tk = np.sqrt(c / (k_list**2 + self.mass**2) * one_m_ab[t_of_k])
        # we perform denoising at shell_kmin <= k < shell_kmax
        shell_kmax = int(np.sum(befIO[t_of_k]))
        shell_size = int(np.sum(inshell[t_of_k]))
        shell_kmin = shell_kmax - shell_size
        
        # Add noise
        V = dct(np.eye(seqlen), norm="ortho")
        pos = V.T @ pos
        pos *= sq_ab_tk[:, np.newaxis]
        pos[0, :] = 0.0 # set k=0 Fourier components to be zero
        pos = V @ pos
        # Make colored noise = V(sq_bb @ normal noise)
        noise = np.random.normal(loc=0.0, scale=1.0, size=pos.shape)
        noise *= sq_bb_tk[:, np.newaxis]
        noise[0, :] = 0.0 # set k=0 Fourier components to be zero
        noise = V @ noise
        # make phi_t and rescaled noise
        pos = pos + noise
        weight = np.sqrt((seqlen+3.)/(shell_size+3.))
        ans = weight * noise
        return pos, ans, shell_kmin, shell_kmax


class RGDiffusionSampler(torch.nn.Module):
    def __init__(self, args, seqlen):
        super().__init__()
        
        # Hyperparameters regarding the noise schedule
        self.T32 = args.T32
        self.mass = args.mass

        self.args = args
        
        # seqlen is the length of a protein chain
        self.seqlen = seqlen
        Lam0, tau, T_rg, T_fl = RGDM_param_calculator(seqlen, self.T32)
        self.TN = TN =  T_rg + T_fl
        
        # V is the DCT matrix
        V = torch.from_numpy(dct(np.eye(seqlen), norm="ortho")).float()
        
        # make noise schedules and save parameters
        recip_sq_a, b_o_wbb, sq_b, shell_kmin, shell_kmax, sqbb_T\
            = self.denoise_schedule(seqlen, Lam0, tau, T_rg, T_fl)
        
        assert recip_sq_a.shape == (TN, seqlen, 1)
        assert b_o_wbb.shape == (TN, seqlen, 1)
        assert sq_b.shape == (TN, seqlen, 1)
        assert shell_kmin.shape == (TN, 1)
        assert shell_kmax.shape == (TN, 1)
        assert sqbb_T.shape == (seqlen, 1)

        self.register_buffer('recip_sq_a', recip_sq_a)
        self.register_buffer('b_o_wbb', b_o_wbb)
        self.register_buffer('sq_b', sq_b)
        self.register_buffer('shell_kmin', shell_kmin)
        self.register_buffer('shell_kmax', shell_kmax)
        self.register_buffer('sqbb_T', sqbb_T)
        self.register_buffer('V', V)
    
    @torch.no_grad()
    def denoise_schedule(self, seqlen, Lam0, tau, T_rg, T_fl, c=(3.8)**2/2.3211):
        # make index of tk
        t = np.arange(0, T_rg + T_fl)
        k = np.pi * np.arange(seqlen) / seqlen
        k[0] = k[1]
        kmax = k[-1]
        tk_index = t[:, np.newaxis] + T_fl\
            - np.floor(tau*np.log(kmax/k[np.newaxis, :])).astype(int)
        
        # make noise schedules
        a_, b_o_bb_, one_m_a_, inshell_, befIO_\
            = noise_schedule_for_sampling(seqlen, Lam0, tau, T_rg, T_fl)
        a = a_[tk_index]
        b_o_bb = b_o_bb_[tk_index]
        one_m_a = one_m_a_[tk_index]
        inshell = inshell_[tk_index]
        befIO = befIO_[tk_index]
        recip_sq_a = np.exp(-0.5*np.log(a))
        sq_b = np.sqrt(c/(k[np.newaxis, :]**2 + self.mass**2) * one_m_a)
        # make k_min, k_max
        shell_kmax = (np.sum(befIO, axis=1)).astype(int)
        shell_size = (np.sum(inshell, axis=1)).astype(int)
        shell_kmin = shell_kmax - shell_size
        weight = np.sqrt((seqlen+3.)/(shell_size+3.))
        b_o_wbb = b_o_bb / weight[:, np.newaxis]
        
        # torch from numpy
        TN = T_rg + T_fl
        recip_sq_a = torch.from_numpy(recip_sq_a).view(TN, seqlen, 1).float()
        b_o_wbb = torch.from_numpy(b_o_wbb).view(TN, seqlen, 1).float()
        sq_b = torch.from_numpy(sq_b).view(TN, seqlen, 1).float()
        shell_kmin = torch.from_numpy(shell_kmin).view(TN, 1)
        shell_kmax = torch.from_numpy(shell_kmax).view(TN, 1)
        
        # sq_bb
        sqbb_T = np.sqrt(c/(k**2+self.mass**2))
        sqbb_T = torch.from_numpy(sqbb_T).view(seqlen, 1).float()
        
        return recip_sq_a, b_o_wbb, sq_b, shell_kmin, shell_kmax, sqbb_T
    

    @torch.no_grad()
    def forward(self, model, data, elbo):
        # make phi_t at t=TN
        pos = torch.randn_like(data['resi'].pos)
        pos = self.sqbb_T * pos
        pos = self.V @ pos

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
        # RG projection params
        shell_kmax = self.shell_kmax[t]
        shell_kmin = self.shell_kmin[t]

        pos = self.V.T @ pos
        pos[0, :] = 0.0; pos[shell_kmax[0]:, :] = 0.0
        # predict noise
        data['resi'].pos = self.V @ pos
        data['resi'].node_t = torch.ones(self.seqlen, device=pos.device) * t
        data.shell_kmax = shell_kmax
        data.shell_kmin = shell_kmin
        pred = model(data)
        # calculate the mean of q(x{t-1}|x0, xt) in the DCT space
        mean = self.recip_sq_a[t] * (
            pos - self.b_o_wbb[t] * (self.V.T @ data.pred)
        )
        # add noise in the DCT space
        if t > 0:
            noise = torch.randn_like(pos)
            sqbb = torch.zeros_like(self.sqbb_T)
            sqbb[shell_kmax[0]:, :] = self.sqbb_T[shell_kmax[0]:, :] 
            corr_noise = (self.sq_b[t] + sqbb) * noise
        else:
            corr_noise = 0.0
        pos = mean + corr_noise
        return self.V @ pos
        
    # In our paper, we did not use lnP_calc, which estimates log-likelihood. 
    @torch.no_grad()
    def lnP_calc(self, model, data, pos):
        data['resi'].pos = pos
        
        lnP = 0.0
        for t in range(self.TN):
            x_tm1 = data['resi'].pos
            noise = torch.randn_like(x_tm1)
            x_t = (self.V.T @ x_tm1) * self.sqrt_alphas[t] + noise * self.sqrt_betas[t] # x_t in k-sp
            i_cutoff = self.i_cutoffs[t]
            noise[0, :] = 0.0; noise[i_cutoff[0]:, :] = 0.0
            x_t[0, :] = 0.0; x_t[i_cutoff[0]:, :] = 0.0
            data['resi'].pos = self.V @ x_t
            data['resi'].node_t = torch.ones(self.seqlen, device=x_t.device) * t
            data.i_cutoff = i_cutoff
            pred = model(data)
            lnP += 0.5 * torch.sum(noise**2)
            lnP -= 0.5 * torch.sum(
                (noise/self.sqrt_alphas[t] -
                 (self.V.T @ data.pred) * self.coef_elbo[t])**2
            )
        lnP -= 0.5 * torch.sum(
            ((self.V.T @ data['resi'].pos) / self.sqbb_T)**2)            
        
        return lnP


# regulator function
def r(x):
    return np.exp(-(np.log(x+1))**2) / -np.expm1(-(np.log(x+1))**2)


# the inverse function of the regulator function
def r_inv(x):
    return np.expm1((np.log((x+1)/x))**0.5)


# preliminary to calculate RGDM hyperparameters
def DDPM_param_calculator(beta1=1e-4, betaT=0.02, T=1000):
    betas = np.linspace(beta1, betaT, T)
    alphas = 1. - betas
    ab = np.cumprod(alphas)
    bb = 1. - ab
    Gamma = 1./bb[0] - 1.
    gamma = 1./bb[-1] - 1.
    return Gamma, gamma


# This function calculate RGDM hyperparameters
def RGDM_param_calculator(seqlen, T32):
    # DDPM parameters
    Gamma, gamma = DDPM_param_calculator()
    
    # hyperparameters
    N0 = 32
    tau = 2 * (T32-1) / np.log((N0-1)**2 * r_inv(gamma) / r_inv(Gamma))
    T_rg = int(np.floor(T32 - tau * np.log(N0-1)))
    Lam0 = np.pi * (seqlen-1) / seqlen / np.sqrt(r_inv(Gamma))
    T_fl = int(np.floor(tau * np.log(seqlen-1)))
    
    return Lam0, tau, T_rg, T_fl

# for training
def noise_schedule_for_training(seqlen, Lam0, tau, T_rg, T_fl):
    t_list = np.arange(T_rg)
    kmax = np.pi * (seqlen-1) / seqlen
    ln_xtk_p1 = np.log(
        kmax**2 / Lam0**2 * np.exp(2*t_list/tau) + 1.
        )

    ab_kmax = np.exp(-ln_xtk_p1**2) # \bar\alpha_{t,k_max}
    a_kmax = np.exp(-ln_xtk_p1[1:]**2 + ln_xtk_p1[:-1]**2) # \alpha_{t,k_max}
    a_kmax = np.insert(a_kmax, 0, ab_kmax[0])
    one_m_ab_kmax = -np.expm1(-ln_xtk_p1**2) # 1-\bar\alpha_{t,k_max}
    one_m_a_kmax = -np.expm1(-ln_xtk_p1[1:]**2 + ln_xtk_p1[:-1]**2) # 1-\alpha_{t,k_max}
    one_m_a_kmax = np.insert(one_m_a_kmax, 0, one_m_ab_kmax[0])
    
    ab = np.concatenate((np.ones(T_fl), ab_kmax, np.zeros(T_fl)))
    a = np.concatenate((np.ones(T_fl), a_kmax, np.ones(T_fl))) # we do not use this quantity in training
    one_m_ab = np.concatenate((np.zeros(T_fl), one_m_ab_kmax, np.zeros(T_fl)))
    one_m_a = np.concatenate((np.zeros(T_fl), one_m_a_kmax, np.zeros(T_fl))) # we do not use this quantity in training
    inshell = np.concatenate((np.zeros(T_fl), np.ones(T_rg), np.zeros(T_fl)))
    befIO = np.concatenate((np.ones(T_fl), np.ones(T_rg), np.zeros(T_fl)))
    aftIO = np.concatenate((np.zeros(T_fl), np.zeros(T_rg), np.ones(T_fl))) # we do not use this quantity in training

    return ab, a, one_m_ab, one_m_a, inshell, befIO, aftIO

# for sampling
def noise_schedule_for_sampling(seqlen, Lam0, tau, T_rg, T_fl):
    t_list = np.arange(T_rg)
    kmax = np.pi * (seqlen-1) / seqlen
    ln_xtk_p1 = np.log(
        kmax**2 / Lam0**2 * np.exp(2*t_list/tau) + 1.
        )

    ab_kmax = np.exp(-ln_xtk_p1**2) # \bar\alpha_{t,k_max}
    a_kmax = np.exp(-ln_xtk_p1[1:]**2 + ln_xtk_p1[:-1]**2) # \alpha_{t,k_max}
    a_kmax = np.insert(a_kmax, 0, ab_kmax[0])
    one_m_ab_kmax = -np.expm1(-ln_xtk_p1**2) # 1-\bar\alpha_{t,k_max}
    one_m_a_kmax = -np.expm1(-ln_xtk_p1[1:]**2 + ln_xtk_p1[:-1]**2) # 1-\alpha_{t,k_max}
    one_m_a_kmax = np.insert(one_m_a_kmax, 0, one_m_ab_kmax[0])

    b_o_bb = one_m_a_kmax / one_m_ab_kmax # beta/\bar\beta

    a = np.concatenate((np.ones(T_fl), a_kmax, np.ones(T_fl)))
    b_o_bb = np.concatenate((np.zeros(T_fl), b_o_bb, np.zeros(T_fl)))
    one_m_a = np.concatenate((np.zeros(T_fl), one_m_a_kmax, np.zeros(T_fl)))
    one_m_a[T_fl] = 0.0
    inshell = np.concatenate((np.zeros(T_fl), np.ones(T_rg), np.zeros(T_fl)))
    befIO = np.concatenate((np.ones(T_fl), np.ones(T_rg), np.zeros(T_fl)))
    
    return a, b_o_bb, one_m_a, inshell, befIO
