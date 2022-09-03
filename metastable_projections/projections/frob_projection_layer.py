import torch as th
from typing import Tuple

from .base_projection_layer import BaseProjectionLayer, mean_projection

from ..misc.norm import mahalanobis, frob_sq
from ..misc.distTools import get_mean_and_chol, get_cov, new_dist_like


class FrobeniusProjectionLayer(BaseProjectionLayer):

    def _trust_region_projection(self, p, q, eps: th.Tensor, eps_cov: th.Tensor, **kwargs):
        """
        Stolen from Fabian's Code (Public Version)

        Runs Frobenius projection layer and constructs cholesky of covariance

        Args:
            policy: policy instance
            p: current distribution
            q: old distribution
            eps: (modified) kl bound/ kl bound for mean part
            eps_cov: (modified) kl bound for cov part
            beta: (modified) entropy bound
            **kwargs:
        Returns: mean, cov cholesky
        """

        mean, chol = get_mean_and_chol(p, expand=True)
        old_mean, old_chol = get_mean_and_chol(q, expand=True)
        batch_shape = mean.shape[:-1]

        ####################################################################################################################
        # precompute mean and cov part of frob projection, which are used for the projection.
        mean_part, cov_part, cov, cov_old = gaussian_frobenius(
            p, q, self.scale_prec, True)

        ################################################################################################################
        # mean projection maha/euclidean

        proj_mean = mean_projection(mean, old_mean, mean_part, eps)

        ################################################################################################################
        # cov projection frobenius

        cov_mask = cov_part > eps_cov

        if cov_mask.any():
            eta = th.ones(batch_shape, dtype=chol.dtype, device=chol.device)
            eta[cov_mask] = th.sqrt(cov_part[cov_mask] / eps_cov) - 1.
            eta = th.max(-eta, eta)

            new_cov = (cov + th.einsum('i,ijk->ijk', eta, cov_old)
                       ) / (1. + eta + 1e-16)[..., None, None]
            proj_chol = th.where(
                cov_mask[..., None, None], th.linalg.cholesky(new_cov), chol)
        else:
            proj_chol = chol

        proj_p = new_dist_like(p, proj_mean, proj_chol)
        return proj_p

    def trust_region_value(self, p, q):
        """
        Stolen from Fabian's Code (Public Version)

        Computes the Frobenius metric between two Gaussian distributions p and q.
        Args:
            policy: policy instance
            p: current distribution
            q: old distribution
        Returns:
            mean and covariance part of Frobenius metric
        """
        return gaussian_frobenius(p, q, self.scale_prec)

    def get_trust_region_loss(self, p, proj_p):
        """
        Stolen from Fabian's Code (Public Version)
        """

        mean_diff, _ = self.trust_region_value(p, proj_p)
        if False and policy.contextual_std:
            # Compute MSE here, because we found the Frobenius norm tends to generate values that explode for the cov
            p_mean, proj_p_mean = p.mean, proj_p.mean
            cov_diff = (p_mean - proj_p_mean).pow(2).sum([-1, -2])
            delta_loss = (mean_diff + cov_diff).mean()
        else:
            delta_loss = mean_diff.mean()

        return delta_loss * self.trust_region_coeff


def gaussian_frobenius(p, q, scale_prec: bool = False, return_cov: bool = False):
    """
    Stolen from Fabian' Code (Public Version)

    Compute (p - q_values) (L_oL_o^T)^-1 (p - 1)^T + |LL^T - L_oL_o^T|_F^2 with p,q_values ~ N(y, LL^T)
    Args:
        policy: current policy
        p: mean and chol of gaussian p
        q: mean and chol of gaussian q_values
        return_cov: return cov matrices for further computations
        scale_prec: scale objective with precision matrix
    Returns: mahalanobis distance, squared frobenius norm
    """

    mean, chol = get_mean_and_chol(p)
    mean_other, chol_other = get_mean_and_chol(q)

    if scale_prec:
        # maha objective for mean
        mean_part = mahalanobis(mean, mean_other, chol_other)
    else:
        # euclidean distance for mean
        # mean_part = ch.norm(mean_other - mean, ord=2, axis=1) ** 2
        mean_part = ((mean_other - mean) ** 2).sum(1)

    # frob objective for cov
    cov = get_cov(p)
    cov_other = get_cov(q)
    diff = cov_other - cov
    # Matrix is real symmetric PSD, therefore |A @ A^H|^2_F = tr{A @ A^H} = tr{A @ A}
    #cov_part = torch_batched_trace(diff @ diff)
    cov_part = frob_sq(diff, is_spd=True)

    if return_cov:
        return mean_part, cov_part, cov, cov_other

    return mean_part, cov_part
