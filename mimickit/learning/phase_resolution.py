"""Phase resolution constrained by the existing PPO minibatch budgets."""
import math

import torch


@torch.no_grad()
def supported_resolution(phase, control_intervals, budgets, masks):
    """Largest K for which every phase can fill each population's quota.

    Uses only phases, eligibility and existing actor/critic batch budgets.
    An empty actor population imposes no constraint and is skipped by PPO.
    No loss values, tracking errors or tunable support thresholds are used.
    """
    p = phase.detach().flatten().double()
    if not len(p) or int(control_intervals) < 1:
        raise ValueError("Phase population and motion resolution must be positive")
    if not bool((torch.isfinite(p) & (p >= 0) & (p <= 1)).all()):
        raise ValueError("Motion phases must be finite and in [0, 1]")
    if len(budgets) != len(masks) or not budgets:
        raise ValueError("Each population requires a minibatch budget and mask")
    populations = []
    upper = min(int(control_intervals), len(p))
    for budget, mask in zip(budgets, masks):
        if int(budget) < 1:
            raise ValueError("Batch budgets must be positive")
        if mask is not None and (mask.dtype != torch.bool or mask.numel() != len(p)):
            raise ValueError("Eligibility must be a Boolean mask of the population")
        population = p if mask is None else p[mask.flatten()]
        if not len(population):
            continue
        b = min(int(budget), len(population))
        population = population.sort().values
        populations.append((population, b))
        upper = min(upper, b)
        # A sample-free interval wider than two bin widths necessarily
        # contains an empty bin. This only prunes impossible candidates.
        gap = float(torch.cat((population[:1], population[1:]-population[:-1],
                               1-population[-1:])).max())
        upper = min(upper, math.ceil(2/gap)+1)
    if not populations:
        raise ValueError("No eligible phase population")
    candidates = torch.arange(1, upper+1, device=p.device)
    valid = torch.ones(upper, device=p.device, dtype=torch.bool)
    # Cheap necessary first/last-pool checks remove most fine candidates.
    # Expand boundary searches by one ulp: pruning must be conservative when
    # phase*K rounds exactly onto an integer. Exact counts are checked below.
    inf = torch.full((upper,), float("inf"), device=p.device, dtype=torch.float64)
    first_edge = torch.nextafter(1/candidates.double(), inf)
    last_edge = torch.nextafter((candidates-1)/candidates.double(), -inf)
    for population, b in populations:
        quota = (b+candidates-1)//candidates
        first = torch.searchsorted(population, first_edge, right=True)
        last = len(population)-torch.searchsorted(population, last_edge)
        first[0] = last[0] = len(population)
        valid &= (first >= quota) & (last >= quota)
    # Batch interior support checks as well. Scanning every candidate over
    # the entire rollout is needlessly expensive for long motions with a
    # sparse interior. Chunk size controls temporary memory only.
    candidates = candidates[valid].flip(0)
    for block in candidates.split(64):
        columns = torch.arange(int(block.max())+1, device=p.device)
        edges = (columns[None,:]/block[:,None].double()).clamp_max(1)
        positive_inf = torch.full_like(edges,float("inf"))
        lower = torch.nextafter(edges,-positive_inf)
        upper_edges = torch.nextafter(edges,positive_inf)
        real_pool = columns[None,:-1] < block[:,None]
        possible = torch.ones(len(block), device=p.device, dtype=torch.bool)
        for population,b in populations:
            # Overinclusive boundaries make this a necessary condition. The
            # final bincount uses exactly the sampler's phase*K arithmetic.
            counts_upper = (torch.searchsorted(population,upper_edges[:,1:].contiguous(),right=True)
                            -torch.searchsorted(population,lower[:,:-1].contiguous()))
            quota = (b+block-1)//block
            possible &= ((counts_upper>=quota[:,None]) | ~real_pool).all(1)
        for k in block[possible].tolist():
            if all(int(torch.bincount((population*k).long().clamp_max(k-1), minlength=k).min())
                   >= math.ceil(b/k) for population, b in populations):
                return k
    return 1  # The whole population always fills its capped batch budget.
