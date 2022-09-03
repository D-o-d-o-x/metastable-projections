from ..misc.distTools import get_diag_cov_vec, get_mean_and_chol, get_cov, is_contextual, new_dist_like, has_diag_cov
from .base_projection_layer import BaseProjectionLayer, mean_projection, mean_equality_projection

import cpp_projection
import numpy as np
import torch as th
from typing import Tuple, Any

from ..misc.norm import mahalanobis

MAX_EVAL = 1000


class KLProjectionLayer(BaseProjectionLayer):
    """
    Stolen from Fabian's Code (Private Version)
    """

    def _trust_region_projection(self, p, q, eps: th.Tensor, eps_cov: th.Tensor, **kwargs):
        """
        Stolen from Fabian's Code (Private Version)

        runs kl projection layer and constructs sqrt of covariance
        Args:
            **kwargs:
            policy: policy instance
            p: current distribution
            q: old distribution
            eps: (modified) kl bound/ kl bound for mean part
            eps_cov: (modified) kl bound for cov part

        Returns:
            mean, cov sqrt
        """
        mean, chol = get_mean_and_chol(p, expand=True)
        old_mean, old_chol = get_mean_and_chol(q, expand=True)

        ################################################################################################################
        # project mean with closed form
        # orig code: mean_part, _ = gaussian_kl(policy, p, q)
        # But the mean_part is just the mahalanobis dist:
        mean_part = mahalanobis(mean, old_mean, old_chol)
        if self.mean_eq:
            proj_mean = mean_equality_projection(
                mean, old_mean, mean_part, eps)
        else:
            proj_mean = mean_projection(mean, old_mean, mean_part, eps)

        if has_diag_cov(p):
            cov_diag = get_diag_cov_vec(p)
            old_cov_diag = get_diag_cov_vec(q)
            proj_cov = KLProjectionGradFunctionDiagCovOnly.apply(cov_diag,
                                                                 old_cov_diag,
                                                                 eps_cov)
            proj_chol = proj_cov.sqrt()  # .diag_embed()
        else:
            cov = get_cov(p)
            old_cov = get_cov(q)
            proj_cov = KLProjectionGradFunctionCovOnly.apply(
                cov, old_cov, chol, old_chol, eps_cov)
            proj_chol = th.linalg.cholesky(proj_cov)
        proj_p = new_dist_like(p, proj_mean, proj_chol)
        return proj_p


class KLProjectionGradFunctionCovOnly(th.autograd.Function):
    projection_op = None

    @staticmethod
    def get_projection_op(batch_shape, dim, max_eval=MAX_EVAL):
        if not KLProjectionGradFunctionCovOnly.projection_op:
            KLProjectionGradFunctionCovOnly.projection_op = \
                cpp_projection.BatchedCovOnlyProjection(
                    batch_shape, dim, max_eval=max_eval)
        return KLProjectionGradFunctionCovOnly.projection_op

    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
        #std, old_std, eps_cov = args
        cov, old_cov, chol, old_chol, eps_cov = args

        batch_shape = chol.shape[0]
        dim = chol.shape[-1]

        cov_np = cov.cpu().detach().numpy()
        old_cov_np = old_cov.cpu().detach().numpy()
        chol_np = chol.cpu().detach().numpy()
        old_chol_np = old_chol.cpu().detach().numpy()
        # eps = eps_cov.cpu().detach().numpy().astype(old_std_np.dtype) * \
        eps = eps_cov * \
            np.ones(batch_shape, dtype=old_chol_np.dtype)

        p_op = KLProjectionGradFunctionCovOnly.get_projection_op(
            batch_shape, dim)
        ctx.proj = p_op

        proj_cov = p_op.forward(eps, old_chol_np, chol_np, cov_np)

        return th.Tensor(proj_cov)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        projection_op = ctx.proj
        d_std, = grad_outputs

        d_std_np = d_std.cpu().detach().numpy()
        d_std_np = np.atleast_2d(d_std_np)
        df_stds = projection_op.backward(d_std_np)
        df_stds = np.atleast_2d(df_stds)

        return d_std.new(df_stds), None, None, None, None


class KLProjectionGradFunctionDiagCovOnly(th.autograd.Function):
    projection_op = None

    @staticmethod
    def get_projection_op(batch_shape, dim: int, max_eval: int = MAX_EVAL):
        if not KLProjectionGradFunctionDiagCovOnly.projection_op:
            KLProjectionGradFunctionDiagCovOnly.projection_op = \
                cpp_projection.BatchedDiagCovOnlyProjection(
                    batch_shape, dim, max_eval=max_eval)
        return KLProjectionGradFunctionDiagCovOnly.projection_op

    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
        cov, old_std_np, eps_cov = args

        batch_shape = cov.shape[0]
        dim = cov.shape[-1]

        std_np = cov.to('cpu').detach().numpy()
        old_std_np = old_std_np.to('cpu').detach().numpy()
        # eps = eps_cov.to('cpu').detach().numpy().astype(old_std_np.dtype) * np.ones(batch_shape, dtype=old_std_np.dtype)
        eps = eps_cov * np.ones(batch_shape, dtype=old_std_np.dtype)

        p_op = KLProjectionGradFunctionDiagCovOnly.get_projection_op(
            batch_shape, dim)
        ctx.proj = p_op

        proj_std = p_op.forward(eps, old_std_np, std_np)

        return cov.new(proj_std)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        projection_op = ctx.proj
        d_std, = grad_outputs

        d_std_np = d_std.to('cpu').detach().numpy()
        d_std_np = np.atleast_2d(d_std_np)
        df_stds = projection_op.backward(d_std_np)
        df_stds = np.atleast_2d(df_stds)

        return d_std.new(df_stds), None, None


class KLProjectionGradFunctionDiagSplit(th.autograd.Function):
    projection_op = None

    @staticmethod
    def get_projection_op(batch_shape, dim: int, max_eval: int = MAX_EVAL):
        if not KLProjectionGradFunctionDiagSplit.projection_op:
            KLProjectionGradFunctionDiagSplit.projection_op = \
                cpp_projection.BatchedSplitDiagMoreProjection(
                    batch_shape, dim, max_eval=max_eval)
        return KLProjectionGradFunctionDiagSplit.projection_op

    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
        mean, cov, old_mean, old_cov, eps_mu, eps_sigma = args

        batch_shape, dim = mean.shape

        mean_np = mean.detach().numpy()
        cov_np = cov.detach().numpy()
        old_mean = old_mean.detach().numpy()
        old_cov = old_cov.detach().numpy()
        eps_mu = eps_mu * np.ones(batch_shape)
        eps_sigma = eps_sigma * np.ones(batch_shape)

        # p_op = cpp_projection.BatchedSplitDiagMoreProjection(batch_shape, dim, max_eval=100)
        p_op = KLProjectionGradFunctionDiagSplit.get_projection_op(
            batch_shape, dim)

        try:
            proj_mean, proj_cov = p_op.forward(
                eps_mu, eps_sigma, old_mean, old_cov, mean_np, cov_np)
        except Exception:
            # try a second time
            proj_mean, proj_cov = p_op.forward(
                eps_mu, eps_sigma, old_mean, old_cov, mean_np, cov_np)
        ctx.proj = p_op

        return mean.new(proj_mean), cov.new(proj_cov)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        p_op = ctx.proj
        d_means, d_std = grad_outputs

        d_std_np = d_std.detach().numpy()
        d_std_np = np.atleast_2d(d_std_np)
        d_mean_np = d_means.detach().numpy()
        dtarget_means, dtarget_covs = p_op.backward(d_mean_np, d_std_np)
        dtarget_covs = np.atleast_2d(dtarget_covs)

        return d_means.new(dtarget_means), d_std.new(dtarget_covs), None, None, None, None


class KLProjectionGradFunctionJoint(th.autograd.Function):
    projection_op = None

    @staticmethod
    def get_projection_op(batch_shape, dim: int, max_eval: int = MAX_EVAL):
        if not KLProjectionGradFunctionJoint.projection_op:
            KLProjectionGradFunctionJoint.projection_op = \
                cpp_projection.BatchedProjection(batch_shape, dim, eec=False, constrain_entropy=False,
                                                 max_eval=max_eval)
        return KLProjectionGradFunctionJoint.projection_op

    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
        mean, cov, old_mean, old_cov, eps, beta = args

        batch_shape, dim = mean.shape

        mean_np = mean.detach().numpy()
        cov_np = cov.detach().numpy()
        old_mean = old_mean.detach().numpy()
        old_cov = old_cov.detach().numpy()
        eps = eps * np.ones(batch_shape)
        beta = beta.detach().numpy() * np.ones(batch_shape)

        # projection_op = cpp_projection.BatchedProjection(batch_shape, dim, eec=False, constrain_entropy=False)
        # ctx.proj = projection_op

        p_op = KLProjectionGradFunctionJoint.get_projection_op(
            batch_shape, dim)
        ctx.proj = p_op

        proj_mean, proj_cov = p_op.forward(
            eps, beta, old_mean, old_cov, mean_np, cov_np)

        return mean.new(proj_mean), cov.new(proj_cov)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        projection_op = ctx.proj
        d_means, d_covs = grad_outputs
        df_means, df_covs = projection_op.backward(
            d_means.detach().numpy(), d_covs.detach().numpy())
        return d_means.new(df_means), d_means.new(df_covs), None, None, None, None
