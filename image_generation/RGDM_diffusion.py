import numpy as np
import torch
import torch.nn as nn
import torch_dct as dct


def extract(noise_sch, t_list):
    out = torch.index_select(noise_sch, dim=0, index=t_list)
    return out.view([t_list.shape[0], 1, noise_sch.shape[1], noise_sch.shape[2]])


def make_hyperparam(beta_1, beta_T, T_ddpm, T, L, m0, mL, k_cutoff):
    # calculate SNR for DDPM
    b_list = np.linspace(beta_1, beta_T, T_ddpm)
    a_list = 1.0 - b_list
    ab_list = np.cumprod(a_list)
    bb_list = 1.0 - ab_list
    v_0 = ab_list[0]/bb_list[0]
    v_Tm1 = ab_list[-1]/bb_list[-1]
    # rescale SN ratio
    v_0 = v_0 * (32/L)**2
    v_Tm1 = v_Tm1 * (32/L)**2
    # calculate tau, Lam0
    tau = 2.0 * (T-1) / np.log(v_0/v_Tm1)
    Lam0 = m0 * np.sqrt(v_0)
    # calculate epsilon for RG projection
    epsilon = Lam0**2 * np.exp(-2.*(T-1)/tau) / (k_cutoff**2 + mL**2)
    return tau, Lam0, epsilon


class RGDiffusionTrainer(nn.Module):
    def __init__(self, model, T, L, beta_1, beta_2, m0, mL, k_cutoff):
        super().__init__()

        self.model = model
        self.T = T
        self.L = L
        
        T_ddpm = 1000
        tau, Lam0, epsilon = make_hyperparam(
            beta_1, beta_2, T_ddpm, T, L, m0, mL, k_cutoff)

        sq_ab, sq_bb, cutoff, w_t = self.noise_sch_train(
            Lam0, tau, m0, mL, T, L, epsilon)

        self.register_buffer('sqrt_alphas_bar', sq_ab)
        self.register_buffer('sqrt_betas_bar', sq_bb)
        self.register_buffer('cutoff', cutoff)
        self.register_buffer('w_t', w_t)

    def forward(self, x_0):
        t = torch.randint(self.T, size=(x_0.shape[0], ), device=x_0.device)
        x_t, res_noise, cutoff = self.make_xt(x_0, t)
        # predict with denoising DNN
        pred_res_noise = self.model(x_t, t)
        # RG projection layer
        pred_res_noise = dct.dct_2d(pred_res_noise)
        pred_res_noise = dct.idct_2d(cutoff * pred_res_noise)
        # calculate loss
        losses = (pred_res_noise - res_noise)**2
        loss = torch.mean(losses)
        return loss

    def make_xt(self, x_0, t):
        # extract
        sq_ab = extract(self.sqrt_alphas_bar, t)
        sq_bb = extract(self.sqrt_betas_bar, t)
        cutoff = extract(self.cutoff, t)
        w_t = extract(self.w_t, t)
        # transform image with RG projection
        x_0 = dct.dct_2d(x_0)
        x_0 = cutoff * sq_ab * x_0
        x_0 = dct.idct_2d(x_0)
        # transform noise with RG projection
        noise = torch.randn_like(x_0)
        noise = dct.dct_2d(noise)
        noise = cutoff * sq_bb * noise
        noise = dct.idct_2d(noise)
        # return noisy image and rescaled noise
        return x_0 + noise, w_t * noise, cutoff

    def noise_sch_train(self, Lam0, tau, m0, mL, T, L, epsilon):
        # make k_sq
        k_sq = torch.zeros(L, L)
        for i in range(L):
            for j in range(L):
                k_sq[i, j] = (i**2 + j**2) * (np.pi/L)**2
        k_sq = k_sq.view(1, L, L)

        t = torch.arange(0, T, 1)
        Lt_sq = Lam0**2 * torch.exp(-2.*t/tau)
        Lt_sq = Lt_sq.view(-1, 1, 1)
        # make r(k^2+mL^2 / Lam^2) = Lam^2 / (k^2+mL^2)
        r_tk = Lt_sq / (k_sq + mL**2)
        cutoff = torch.where(r_tk < epsilon, 0., 1.)
        # make alphas_bar, betas_bar
        ab = r_tk / (1. + r_tk)
        bb = m0**2 / (k_sq + mL**2) / (1. + r_tk)
        # make noise rescale weight : w_t (\sqrt(lambda_t) in paper)
        prefactor = torch.sum(m0**2 / (k_sq + mL**2))
        w_t = torch.sqrt(
            prefactor / torch.sum((cutoff*bb).reshape(bb.shape[0], -1), dim=-1)
            )
        w_t = w_t.view(-1, 1, 1)

        return torch.exp(0.5*torch.log(ab)), torch.exp(0.5*torch.log(bb)),\
            cutoff, w_t


class RGDiffusionSampler(nn.Module):
    def __init__(self, model, T, L, beta_1, beta_2, m0, mL, k_cutoff):
        super().__init__()

        self.model = model
        self.T = T
        self.L = L
        
        T_ddpm = 1000
        tau, Lam0, epsilon = make_hyperparam(
            beta_1, beta_2, T_ddpm, T, L, m0, mL, k_cutoff)

        sqrt_recip_alphas_bar, coef_pred_x0, coef1, coef2,\
            sqrt_betas, sqrt_betas_bar, sqbb_T, cutoff \
            = self.noise_sch_sample(Lam0, tau, m0, mL, T, L, epsilon)

        self.register_buffer('sqrt_recip_alphas_bar', sqrt_recip_alphas_bar)
        self.register_buffer('coef_pred_x0', coef_pred_x0)
        self.register_buffer('coef1', coef1)
        self.register_buffer('coef2', coef2)
        self.register_buffer('sqrt_betas', sqrt_betas)
        self.register_buffer('sqrt_betas_bar', sqrt_betas_bar)
        self.register_buffer('sqbb_T', sqbb_T)
        self.register_buffer('cutoff', cutoff)

    @torch.no_grad()
    def forward(self, batch_size, device):
        # make x_t at t=T
        x_t = torch.randn((batch_size, 3, self.L, self.L), device=device)
        x_t = dct.idct_2d(self.sqbb_T * dct.dct_2d(x_t))
        
        for time_step in reversed(range(self.T)):
            t = x_t.new_ones([x_t.shape[0], ], dtype=torch.long) * time_step
            mean = self.q_mean(x_t, t)
            if time_step > 0:
                cutoff = extract(self.cutoff, t)
                noise = torch.randn_like(x_t)
                noise = dct.dct_2d(noise)
                noise1 = cutoff * extract(self.sqrt_betas, t) * noise
                noise2 = (1.-cutoff) * extract(self.sqrt_betas_bar, t-1) * noise
                corr_noise = dct.idct_2d(noise1) + dct.idct_2d(noise2)
            else:
                corr_noise = 0.0
            x_t = mean + corr_noise
        x_0 = x_t
        return torch.clip(x_0, -1, 1)
    
    def q_mean(self, x_t, t):
        cutoff = extract(self.cutoff, t)
        # RG projection
        x_t = cutoff * dct.dct_2d(x_t)
        # predict noise
        res_noise = self.model(dct.idct_2d(x_t), t)
        res_noise = cutoff * dct.dct_2d(res_noise)
        # predict x0
        x_0 = extract(self.sqrt_recip_alphas_bar, t) * x_t -\
            extract(self.coef_pred_x0, t) * res_noise
        # compute mean of q(x_tm1|x_t, x_0)
        mean = extract(self.coef1, t) * x_0 +\
            extract(self.coef2, t) * x_t

        return dct.idct_2d(mean)
    
    def noise_sch_sample(self, Lam0, tau, m0, mL, T, L, epsilon):
        # make k_sq
        k_sq = torch.zeros(L, L)
        for i in range(L):
            for j in range(L):
                k_sq[i, j] = (i**2 + j**2) * (np.pi/L)**2
        k_sq = k_sq.view(1, L, L)

        t = torch.arange(0, T, 1)
        Lt_sq = Lam0**2 * torch.exp(-2.*t/tau)
        Lt_sq = Lt_sq.view(-1, 1, 1)
        # make r(k^2+mL^2 / Lam^2) = Lam^2 / (k^2+mL^2)
        r_tk = Lt_sq / (k_sq + mL**2)
        cutoff = torch.where(r_tk < epsilon, 0., 1.)
        # make alphas_bar, betas_bar
        ab = r_tk / (1. + r_tk)
        bb = m0**2 / (k_sq + mL**2) / (1. + r_tk)
        # make rescale weight
        prefactor = torch.sum(m0**2 / (k_sq + mL**2))
        w_t = torch.sqrt(
            prefactor / torch.sum((cutoff*bb).reshape(bb.shape[0], -1), dim=-1)
            )
        w_t = w_t.view(-1, 1, 1)

        # make a, b
        r_tm1k = r_tk[:-1, :, :]  # index: 0~T-2
        r_tk = r_tk[1:, :, :]  # index: 1~T-1
        a = np.exp(-2./tau) * (1. + r_tm1k) / (1. + r_tk)
        a = torch.cat([ab[0, :, :].unsqueeze(0), a], dim=0)
        b = -np.expm1(-2./tau) * m0**2 / (k_sq + mL**2) / (1. + r_tk)
        b = torch.cat([bb[0, :, :].unsqueeze(0), b], dim=0)
        # other necessary quantities
        sqrt_recip_alphas_bar = torch.exp(-0.5*torch.log(ab))
        coef_pred_x0 = sqrt_recip_alphas_bar / w_t
        # coef1, coef2
        ab_prev = torch.cat([torch.ones_like(ab[0]).unsqueeze(0), ab[:-1, :, :]], dim=0)
        bb_prev = torch.cat([torch.zeros_like(bb[0]).unsqueeze(0), bb[:-1, :, :]], dim=0)
        coef1 = torch.exp(0.5*torch.log(ab_prev)) * b / bb
        coef2 = torch.exp(0.5*torch.log(a)) * bb_prev / bb
        sqbb_T = torch.sqrt(bb[-1]).view(1, self.L, self.L)

        return sqrt_recip_alphas_bar, coef_pred_x0, coef1, coef2,\
            torch.exp(0.5*torch.log(b)), torch.exp(0.5*torch.log(bb)), sqbb_T, cutoff
