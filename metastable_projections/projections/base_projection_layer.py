from typing import Any, Dict, Optional, Type, Union, Tuple, final

import torch as th

from stable_baselines3.common.distributions import kl_divergence
from stable_baselines3.common.distributions import Distribution

from ..misc.distTools import *


class BaseProjectionLayer(object):

    def __init__(self,
                 mean_bound: float = 0.03,
                 cov_bound: float = 1e-3,
                 trust_region_coeff: float = 1.0,
                 scale_prec: bool = True,
                 do_entropy_proj: bool = False,
                 entropy_eq: bool = False,
                 entropy_first: bool = False,
                 ):
        self.mean_bound = mean_bound
        self.cov_bound = cov_bound
        self.trust_region_coeff = trust_region_coeff
        self.do_entropy_proj = do_entropy_proj
        self.entropy_first = scale_prec
        self.scale_prec = scale_prec
        self.mean_eq = False

        self.entropy_first = entropy_first
        self.entropy_proj = entropy_equality_projection if entropy_eq else entropy_inequality_projection

    def __call__(self, p, q, step, *args, **kwargs):
        # TODO: self.entropy_schedule(self.initial_entropy, self.target_entropy, self.temperature, step) * p[0].new_ones(p[0].shape[0])
        entropy_bound = 'lol'
        return self._projection(p, q, eps=self.mean_bound, eps_cov=self.cov_bound, beta=entropy_bound, **kwargs)

    @final
    def _projection(self, p, q, eps: th.Tensor, eps_cov: th.Tensor, beta: th.Tensor, **kwargs):
        """
        Template method with hook _trust_region_projection() to encode specific functionality.
        (Optional) entropy projection is executed before or after as specified by entropy_first.
        Do not override this. For Python >= 3.8 you can use the @final decorator to enforce not overwriting.
        Args:
            policy: policy instance
            p: current distribution
            q: old distribution
            eps: mean trust region bound
            eps_cov: covariance trust region bound
            beta: entropy bound
            **kwargs:

        Returns:
            projected mean, projected std
        """

        ####################################################################################################################
        # entropy projection in the beginning
        if self.do_entropy_proj and self.entropy_first:
            p = self.entropy_proj(p, beta)

        ####################################################################################################################
        # trust region projection for mean and cov bounds
        new_p = self._trust_region_projection(
            p, q, eps, eps_cov, **kwargs)

        ####################################################################################################################
        # entropy projection in the end
        if not self.do_entropy_proj or self.entropy_first:
            return new_p

        return self.entropy_proj(new_p, beta)

    def _trust_region_projection(self, p, q, eps: th.Tensor, eps_cov: th.Tensor, **kwargs):
        """
        Hook for implementing the specific trust region projection
        Args:
            p: current distribution
            q: old distribution
            eps: mean trust region bound
            eps_cov: covariance trust region bound
            **kwargs:

        Returns:
            projected
        """
        return p

    def get_trust_region_loss(self, p, proj_p):
        # p:
        #   predicted distribution from network output
        # proj_p:
        #   projected distribution

        proj_mean, proj_chol = get_mean_and_chol(proj_p)
        p_target = new_dist_like(p, proj_mean, proj_chol)
        kl_diff = self.trust_region_value(p, p_target)

        kl_loss = kl_diff.mean()

        return kl_loss * self.trust_region_coeff

    def trust_region_value(self, p, q):
        """
        Computes the KL divergence between two Gaussian distributions p and q_values.
        Returns:
            full kl divergence
        """
        return kl_divergence(p, q)


def entropy_inequality_projection(p: th.distributions.Normal,
                                  beta: Union[float, th.Tensor]):
    """
    Stolen and adapted from Fabian's Code (Public Version)

    Projects std to satisfy an entropy INEQUALITY constraint.
    Args:
        p: current distribution
        beta: target entropy for EACH std or general bound for all stds

    Returns:
        projected std that satisfies the entropy bound
    """
    mean, std = p.mean, p.stddev
    k = std.shape[-1]
    batch_shape = std.shape[:-2]

    ent = p.entropy()
    mask = ent < beta

    # if nothing has to be projected skip computation
    if (~mask).all():
        return p

    alpha = th.ones(batch_shape, dtype=std.dtype, device=std.device)
    alpha[mask] = th.exp((beta[mask] - ent[mask]) / k)

    proj_std = th.einsum('ijk,i->ijk', std, alpha)
    new_mean, new_std = mean, th.where(mask[..., None, None], proj_std, std)
    return th.distributions.Normal(new_mean, new_std)


def entropy_equality_projection(p: th.distributions.Normal,
                                beta: Union[float, th.Tensor]):
    """
    Stolen and adapted from Fabian's Code (Public Version)

    Projects std to satisfy an entropy EQUALITY constraint.
    Args:
        p: current distribution
        beta: target entropy for EACH std or general bound for all stds

    Returns:
        projected std that satisfies the entropy bound
    """
    mean, std = p.mean, p.stddev
    k = std.shape[-1]

    ent = p.entropy()
    alpha = th.exp((beta - ent) / k)
    proj_std = th.einsum('ijk,i->ijk', std, alpha)
    new_mean, new_std = mean, proj_std
    return th.distributions.Normal(new_mean, new_std)


def mean_projection(mean: th.Tensor, old_mean: th.Tensor, maha: th.Tensor, eps: th.Tensor):
    """
    Stolen from Fabian's Code (Public Version)

    Projects the mean based on the Mahalanobis objective and trust region.
    Args:
        mean: current mean vectors
        old_mean: old mean vectors
        maha: Mahalanobis distance between the two mean vectors
        eps: trust region bound

    Returns:
        projected mean that satisfies the trust region
    """
    batch_shape = mean.shape[:-1]
    mask = maha > eps

    ################################################################################################################
    # mean projection maha

    # if nothing has to be projected skip computation
    if mask.any():
        omega = th.ones(batch_shape, dtype=mean.dtype, device=mean.device)
        omega[mask] = th.sqrt(maha[mask] / eps) - 1.
        omega = th.max(-omega, omega)[..., None]

        m = (mean + omega * old_mean) / (1 + omega + 1e-16)
        proj_mean = th.where(mask[..., None], m, mean)
    else:
        proj_mean = mean

    return proj_mean


def mean_equality_projection(mean: th.Tensor, old_mean: th.Tensor, maha: th.Tensor, eps: th.Tensor):
    """
    Stolen from Fabian's Code (Private Version)

    Projections the mean based on the Mahalanobis objective and trust region for an EQUALITY constraint.
    Args:
        mean: current mean vectors
        old_mean: old mean vectors
        maha: Mahalanobis distance between the two mean vectors
        eps: trust region bound
    Returns:
        projected mean that satisfies the trust region
    """

    maha[maha == 0] += 1e-16
    omega = th.sqrt(maha / eps) - 1.
    omega = omega[..., None]

    proj_mean = (mean + omega * old_mean) / (1 + omega + 1e-16)

    return proj_mean
