"""NVlabs/DiffusionNFT endpoint objective; not a complete online RL trainer."""
import torch


def nft_loss(xt, x0, sigma, current, old, reference, advantage, beta=0.1, adv_clip=5.0, kl_weight=0.004):
    if beta <= 0 or adv_clip <= 0 or kl_weight < 0:
        raise ValueError('Invalid objective coefficients')
    if any(x.shape != x0.shape for x in (xt, current, old, reference)) or x0.ndim < 2:
        raise ValueError('Equal batched endpoint and velocity shapes required')
    if sigma.shape != (len(x0),) or advantage.shape != (len(x0),):
        raise ValueError('Batched sigma and advantage vectors required')
    if not torch.isfinite(sigma).all() or ((sigma < 0) | (sigma > 1)).any() or not torch.isfinite(advantage).all():
        raise ValueError('Invalid sigma or advantage')
    dims = tuple(range(1, x0.ndim))
    t = sigma.view(-1, *([1] * (x0.ndim - 1)))
    old = old.detach()
    reference = reference.detach()
    target = x0.detach()
    positive = beta * current + (1 - beta) * old
    negative = (1 + beta) * old - beta * current
    xp = xt.detach() - t * positive
    xn = xt.detach() - t * negative
    with torch.no_grad():
        wp = (xp.double() - target.double()).abs().mean(dims, keepdim=True).clamp(min=1e-5)
        wn = (xn.double() - target.double()).abs().mean(dims, keepdim=True).clamp(min=1e-5)
        r = (advantage.detach().clamp(-adv_clip, adv_clip) / adv_clip / 2 + 0.5).clamp(0, 1)
    lp = ((xp - target).square() / wp).mean(dims)
    ln = ((xn - target).square() / wn).mean(dims)
    policy = ((r * lp + (1 - r) * ln) / beta * adv_clip).mean()
    kl = (current - reference).square().mean()
    return policy + kl_weight * kl, {'policy': policy.detach(), 'reference_mse': kl.detach()}
