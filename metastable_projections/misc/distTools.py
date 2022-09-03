from typing import Any, Dict, List, Optional, Tuple, Union

import torch as th

from stable_baselines3.common.distributions import Distribution as SB3_Distribution


class UniversalGaussianDistribution(SB3_Distribution):
    pass


AnyDistribution = Union[SB3_Distribution, UniversalGaussianDistribution]


def get_mean_and_chol(p: AnyDistribution, expand=False):
    if isinstance(p, th.distributions.Normal) or isinstance(p, th.distributions.Independent):
        if expand:
            return p.mean, th.diag_embed(p.stddev)
        else:
            return p.mean, p.stddev
    elif isinstance(p, th.distributions.MultivariateNormal):
        return p.mean, p.scale_tril
    elif isinstance(p, SB3_Distribution):
        return get_mean_and_chol(p.distribution, expand=expand)
    else:
        raise Exception('Dist-Type not implemented')


def get_mean_and_sqrt(p: UniversalGaussianDistribution, expand=False):
    if not hasattr(p, 'cov_sqrt'):
        raise Exception(
            'Distribution was not induced from sqrt. On-demand calculation is not supported.')
    else:
        mean, chol = get_mean_and_chol(p, expand=False)
        sqrt_cov = p.cov_sqrt
        if mean.shape[0] != sqrt_cov.shape[0]:
            shape = list(sqrt_cov.shape)
            shape[0] = mean.shape[0]
            shape = tuple(shape)
            sqrt_cov = sqrt_cov.expand(shape)
        if expand and len(sqrt_cov.shape) <= 2:
            sqrt_cov = th.diag_embed(sqrt_cov)
        return mean, sqrt_cov


def get_cov(p: AnyDistribution):
    if isinstance(p, th.distributions.Normal) or isinstance(p, th.distributions.Independent):
        return th.diag_embed(p.variance)
    elif isinstance(p, th.distributions.MultivariateNormal):
        return p.covariance_matrix
    elif isinstance(p, SB3_Distribution):
        return get_cov(p.distribution)
    else:
        raise Exception('Dist-Type not implemented')


def has_diag_cov(p: AnyDistribution, numerical_check=False):
    if isinstance(p, SB3_Distribution):
        return has_diag_cov(p.distribution, numerical_check=numerical_check)
    if isinstance(p, th.distributions.Normal) or isinstance(p, th.distributions.Independent):
        return True
    if not numerical_check:
        return False
    # Check if matrix is diag
    cov = get_cov(p)
    return th.equal(cov - th.diag_embed(th.diagonal(cov, dim1=-2, dim2=-1)), th.zeros_like(cov))


def is_contextual(p: AnyDistribution):
    # TODO: Implement for UniveralGaussianDist
    return False


def get_diag_cov_vec(p: AnyDistribution, check_diag=True, numerical_check=False):
    if check_diag and not has_diag_cov(p, numerical_check=numerical_check):
        raise Exception('Cannot reduce cov-mat to diag-vec: Is not diagonal')
    return th.diagonal(get_cov(p), dim1=-2, dim2=-1)


def new_dist_like(orig_p: AnyDistribution, mean: th.Tensor, chol: th.Tensor):
    if isinstance(orig_p, UniversalGaussianDistribution):
        return orig_p.new_dist_like_me(mean, chol)
    elif isinstance(orig_p, th.distributions.Normal):
        if orig_p.stddev.shape != chol.shape:
            chol = th.diagonal(chol, dim1=1, dim2=2)
        return th.distributions.Normal(mean, chol)
    elif isinstance(orig_p, th.distributions.Independent):
        if orig_p.stddev.shape != chol.shape:
            chol = th.diagonal(chol, dim1=1, dim2=2)
        return th.distributions.Independent(th.distributions.Normal(mean, chol), 1)
    elif isinstance(orig_p, th.distributions.MultivariateNormal):
        return th.distributions.MultivariateNormal(mean, scale_tril=chol)
    elif isinstance(orig_p, SB3_Distribution):
        p = orig_p.distribution
        if isinstance(p, th.distributions.Normal):
            p_out = orig_p.__class__(orig_p.action_dim)
            p_out.distribution = th.distributions.Normal(mean, chol)
        elif isinstance(p, th.distributions.Independent):
            p_out = orig_p.__class__(orig_p.action_dim)
            p_out.distribution = th.distributions.Independent(
                th.distributions.Normal(mean, chol), 1)
        elif isinstance(p, th.distributions.MultivariateNormal):
            p_out = orig_p.__class__(orig_p.action_dim)
            p_out.distribution = th.distributions.MultivariateNormal(
                mean, scale_tril=chol)
        else:
            raise Exception('Dist-Type not implemented (of sb3 dist)')
        return p_out
    else:
        raise Exception('Dist-Type not implemented')


def new_dist_like_from_sqrt(orig_p: AnyDistribution, mean: th.Tensor, cov_sqrt: th.Tensor):
    chol = _sqrt_to_chol(cov_sqrt, only_diag=has_diag_cov(orig_p))

    new = new_dist_like(orig_p, mean, chol)

    new.cov_sqrt = cov_sqrt
    if hasattr(new, 'distribution'):
        new.distribution.cov_sqrt = cov_sqrt

    return new


def _sqrt_to_chol(cov_sqrt, only_diag=False):
    cov = th.bmm(cov_sqrt.mT, cov_sqrt)
    cov += th.eye(cov.shape[-1]).expand(cov.shape)*(1e-6)

    chol = th.linalg.cholesky(cov)

    if only_diag:
        chol = th.diagonal(chol, dim1=-2, dim2=-1)

    return chol
