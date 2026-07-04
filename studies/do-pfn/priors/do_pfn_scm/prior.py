from __future__ import annotations

from typing import Any, Callable

import random
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from pfnstudio_core import Prior, register_prior


def _identity(x):
    return x


def activation_sampling(nonlins: str):
    if nonlins == "paper_text":
        return np.random.choice([torch.square, torch.relu, torch.tanh])
    if nonlins in ("mixed", "post"):
        return np.random.choice([torch.square, torch.relu, torch.tanh, _identity])
    if nonlins == "tanh":
        return torch.tanh
    if nonlins == "relu":
        return torch.relu
    if nonlins == "id":
        return _identity
    return np.random.choice([torch.square, torch.relu, torch.tanh])


def make_exo_dist_samples(shape: tuple[int, ...], exo_std: float):
    def sample():
        return torch.normal(0, exo_std, shape)
    return sample


def make_additive_noise_gaussian(shape: tuple[int, ...], std: float):
    def sample():
        return torch.normal(0, std, shape)
    return sample


class MakeStructuralEquations(nn.Module):
    def __init__(
        self,
        parents: list[str],
        samples_shape: tuple[int, ...],
        noise_std: float,
        noise_dist: str = "gaussian",
        nonlins: str = "paper_text",
        max_hidden_layers: int = 0,
    ):
        super().__init__()
        self.parents = parents
        self.nonlins = nonlins
        self.layers = nn.Linear(len(parents), 1, bias=False) if parents else None
        self.activation = activation_sampling(nonlins)

        if noise_dist != "gaussian":
            raise ValueError("Do-PFN v1 training prior uses gaussian noise.")
        self.additive_noise = make_additive_noise_gaussian(samples_shape, noise_std)()

    def forward(self, **kwargs):
        if not self.parents:
            return self.additive_noise

        parent_values = [kwargs[parent] for parent in self.parents]
        parent_tensor = torch.stack(parent_values, dim=-1)

        with torch.no_grad():
            out = self.layers(parent_tensor).squeeze(-1)
            if self.nonlins == "post":
                return self.activation(out + self.additive_noise)
            return self.activation(out) + self.additive_noise


class StructuralCausalModel:
    def __init__(self):
        self.endogenous_vars = {}
        self.exogenous_vars = {}
        self.functions = {}
        self.exogenous_distributions = {}
        self.saved_functions = {}
        self.binary_strategy = "mean"

    def add_endogenous_var(self, name: str, function: Callable, param_varnames: dict):
        name = name.upper()
        self.endogenous_vars[name] = None
        self.functions[name] = (function, param_varnames)

    def add_exogenous_var(self, name: str, distribution: Callable, distribution_kwargs: dict):
        name = name.upper()
        self.exogenous_vars[name] = None
        self.exogenous_distributions[name] = (distribution, distribution_kwargs)

    def create_graph(self):
        graph = nx.DiGraph()
        [graph.add_node(v.upper(), type="endo") for v in self.endogenous_vars]
        [graph.add_node(v.upper(), type="exo") for v in self.exogenous_vars]
        for var in self.functions:
            for parent in self.functions[var][1].values():
                graph.add_edge(parent.upper(), var.upper())
        return graph

    def set_binarization_params(self, treatment):
        threshs, t1s, t2s = [], [], []
        for b in range(treatment.shape[0]):
            vals = torch.nan_to_num(treatment[b])
            not_min_max = (vals > vals.min()) & (vals < vals.max())
            if not bool(not_min_max.any()):
                threshs.append(vals[0])
                t1s.append(vals[0])
                t2s.append(vals[0])
                continue

            thresh = vals[not_min_max].mean()
            low = vals[vals < thresh]
            high = vals[vals >= thresh]
            t1 = low.mean() if len(low) else vals.min()
            t2 = high.mean() if len(high) else vals.max()
            threshs.append(thresh)
            t1s.append(t1)
            t2s.append(t2)

        self.t_threshs = torch.stack([x.reshape(()) for x in threshs])
        self.t1s = torch.stack([x.reshape(()) for x in t1s])
        self.t2s = torch.stack([x.reshape(()) for x in t2s])

    def get_binarized_treatment(self, treatment):
        for b in range(treatment.shape[0]):
            lt = treatment[b] < self.t_threshs[b]
            treatment[b][lt] = self.t1s[b]
            treatment[b][~lt] = self.t2s[b]
        return treatment

    def get_zero_one_treatment(self, treatment):
        for b in range(treatment.shape[1]):
            treatment[:, b] = (treatment[:, b] < treatment[:, b].mean()).float()
        return treatment

    def get_next_sample(self, exogenous_vars=None, binarize=False, graph=None):
        if exogenous_vars is None:
            for key, dist in self.exogenous_distributions.items():
                self.exogenous_vars[key] = dist[0](**dist[1])
        else:
            self.exogenous_vars = exogenous_vars

        if binarize and self.t_key in self.exogenous_vars and exogenous_vars is None:
            self.set_binarization_params(self.exogenous_vars[self.t_key])
            self.exogenous_vars[self.t_key] = self.get_binarized_treatment(self.exogenous_vars[self.t_key])

        structure = graph if graph is not None else self.create_graph()
        for node in nx.topological_sort(structure):
            if node in self.exogenous_vars:
                continue
            lookup = {**self.exogenous_vars, **self.endogenous_vars}
            param_map = self.functions[node][1]
            params = {p: lookup[param_map[p]] for p in param_map}
            self.endogenous_vars[node] = self.functions[node][0](**params)

            if binarize and self.t_key == node:
                self.set_binarization_params(self.endogenous_vars[node])
                self.endogenous_vars[node] = self.get_binarized_treatment(self.endogenous_vars[node])

        return dict(self.endogenous_vars), dict(self.exogenous_vars)

    def do_interventions(self, interventions):
        self.saved_functions = {}
        for target, intervention in interventions:
            self.saved_functions[target] = self.functions[target]
            self.functions[target] = intervention

    def undo_interventions(self):
        for key, value in self.saved_functions.items():
            self.functions[key] = value
        self.saved_functions.clear()


class SCMGenerator:
    def __init__(
        self,
        all_functions: dict[str, Callable],
        seed: int,
        samples_shape: tuple[int, ...],
        noise_std: float,
        noise_dist: str,
        nonlins: str,
        max_hidden_layers: int = 0,
    ):
        self.all_functions = all_functions
        self.seed = seed
        self.samples_shape = samples_shape
        self.noise_std = noise_std
        self.noise_dist = noise_dist
        self.nonlins = nonlins
        self.max_hidden_layers = max_hidden_layers

    def create_graph_from_nodes(self, num_nodes: int, p: float):
        graph = nx.DiGraph()
        nodes = list(range(num_nodes))
        graph.add_nodes_from(nodes)
        perm = np.random.permutation(nodes)
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if random.random() < p:
                    graph.add_edge(perm[i], perm[j])
        return graph

    def create_scm_from_graph(self, graph, possible_functions, exo_distribution, exo_distribution_kwargs):
        scm = StructuralCausalModel()

        mapping = {}
        for n in graph.nodes:
            parents = list(graph.predecessors(n))
            mapping[n] = "X" + str(n) if parents else "U" + str(n)
        graph = nx.relabel_nodes(graph, mapping, copy=True)

        random.seed(self.seed)
        for n in graph.nodes:
            parents = list(graph.predecessors(n))
            if parents:
                fn_name = random.choice(possible_functions)
                scm.add_endogenous_var(
                    n,
                    self.all_functions[fn_name](
                        parents=parents,
                        samples_shape=self.samples_shape,
                        noise_std=self.noise_std,
                        noise_dist=self.noise_dist,
                        nonlins=self.nonlins,
                        max_hidden_layers=self.max_hidden_layers,
                    ),
                    {p: p for p in parents},
                )
            else:
                scm.add_exogenous_var(n, exo_distribution, exo_distribution_kwargs)

        return scm


def _idx(name: str) -> int:
    return int(name[1:])


def _choice(items):
    return items[int(torch.randint(0, len(items), (1,)).item())]


def _adjacency_from_graph(graph, k: int):
    adj = np.zeros((k, k), dtype=np.int8)
    for src, dst in graph.edges:
        adj[_idx(src), _idx(dst)] = 1
    return adj


class DoPfnBatch:
    def __init__(self, x, y, target_y, x_int, **extra):
        self.x = x
        self.y = y
        self.target_y = target_y
        self.x_int = x_int
        for key, value in extra.items():
            setattr(self, key, value)


def sample_do_pfn_torch_batch(
    *,
    seed: int,
    batch_size: int,
    seq_len: int,
    num_features: int,
    num_unobserved: int = 1,
    noise_dist: str = "gaussian",
    exo_dist: str = "gaussian",
    nonlins: str = "paper_text",
    max_hidden_layers: int = 0,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if exo_dist != "gaussian":
        raise ValueError("Do-PFN v1 training prior uses gaussian exogenous noise.")

    k = int(num_features + num_unobserved + 2)
    samples_shape = (batch_size, seq_len)
    p_edge = float(np.random.uniform(1.0 / (num_features + 1), 1.0))
    sigma_exo = float(np.random.uniform(1.0, 3.0))
    sigma_eps = float(0.3 * np.random.beta(1.0, 5.0))

    gen = SCMGenerator(
        all_functions={"nonlinear": MakeStructuralEquations},
        seed=seed,
        samples_shape=samples_shape,
        noise_std=sigma_eps,
        noise_dist=noise_dist,
        nonlins=nonlins,
        max_hidden_layers=max_hidden_layers,
    )

    raw_graph = gen.create_graph_from_nodes(k, p_edge)
    scm = gen.create_scm_from_graph(
        raw_graph,
        possible_functions=["nonlinear"],
        exo_distribution=make_exo_dist_samples(samples_shape, sigma_exo),
        exo_distribution_kwargs={},
    )
    graph = scm.create_graph()
    nodes = list(graph.nodes)

    t_candidates = [n for n in nodes if graph.out_degree(n) > 0]
    if not t_candidates:
        raise RuntimeError("sampled graph has no treatment candidate")
    scm.t_key = _choice(t_candidates)

    descendants = list(nx.descendants(graph, scm.t_key))
    if not descendants:
        raise RuntimeError("sampled treatment has no descendants")
    scm.y_key = _choice(descendants)

    endo_obs, exo_obs = scm.get_next_sample(binarize=True, graph=graph)
    sample_obs = endo_obs | exo_obs

    coin = torch.randint(0, 2, (batch_size, seq_len))
    t1s = scm.t1s.unsqueeze(1).expand(-1, seq_len)
    t2s = scm.t2s.unsqueeze(1).expand(-1, seq_len)
    t_int = torch.where(coin == 0, t1s, t2s)

    if scm.t_key in scm.endogenous_vars:
        scm.do_interventions([(scm.t_key, (lambda: t_int, {}))])
    else:
        exo_obs[scm.t_key] = t_int

    endo_int, exo_int = scm.get_next_sample(exogenous_vars=exo_obs, graph=graph)
    sample_int = endo_int | exo_int
    scm.undo_interventions()

    x_candidates = list(set(graph.nodes) - {scm.t_key, scm.y_key})
    x_keys = [scm.t_key] + list(np.random.choice(x_candidates, size=num_features, replace=False))

    x_obs = torch.stack([sample_obs[key] for key in x_keys]).permute(-1, 1, 0)
    x_int = torch.stack([sample_int[key] for key in x_keys]).permute(-1, 1, 0)
    x_obs[:, :, 0] = scm.get_zero_one_treatment(x_obs[:, :, 0])
    x_int[:, :, 0] = scm.get_zero_one_treatment(x_int[:, :, 0])

    y_obs = sample_obs[scm.y_key].T.unsqueeze(-1)
    y_int = sample_int[scm.y_key].T.unsqueeze(-1)

    for tensor in (x_obs, x_int, y_obs, y_int):
        if torch.any(torch.isnan(tensor)) or torch.any(torch.isinf(tensor)):
            raise FloatingPointError("non-finite sample")

    adj = _adjacency_from_graph(graph, k)
    cov_indices = [_idx(key) for key in x_keys[1:]]
    t_idx = _idx(scm.t_key)
    y_idx = _idx(scm.y_key)

    return DoPfnBatch(
        x=x_obs.detach(),
        y=y_obs.detach(),
        target_y=y_int.detach(),
        x_int=x_int.detach(),
        adjacency=adj,
        task_meta={
            "K": k,
            "num_features": int(num_features),
            "num_unobserved": int(num_unobserved),
            "sigma_exo": sigma_exo,
            "sigma_eps": sigma_eps,
            "p_edge": p_edge,
            "edge_density": float(adj.sum() / (k * k)),
            "t_idx": t_idx,
            "y_idx": y_idx,
            "x_indices": cov_indices,
            "t_key": scm.t_key,
            "y_key": scm.y_key,
            "x_keys": x_keys,
        },
    )


def sample_do_pfn_studio_task(
    *,
    seed: int,
    num_samples: int = 2200,
    ctx_frac: float = 0.75,
    num_features: int = 6,
    num_unobserved: int = 1,
    K_max: int = 10,
    max_retries: int = 5,
    **params: Any,
):
    k = int(num_features + num_unobserved + 2)
    if k > K_max:
        raise ValueError(f"K={k} exceeds K_max={K_max}")

    last_exc = None
    for attempt in range(max_retries):
        try:
            batch = sample_do_pfn_torch_batch(
                seed=seed + attempt * 1_000_003,
                batch_size=1,
                seq_len=num_samples,
                num_features=num_features,
                num_unobserved=num_unobserved,
                **params,
            )
            break
        except Exception as exc:
            last_exc = exc
    else:
        raise RuntimeError(f"Do-PFN sampling failed: {last_exc}") from last_exc

    n_ctx = int(num_samples * ctx_frac)
    n_query = num_samples - n_ctx

    x_obs = batch.x[:n_ctx, 0, :].cpu().numpy().astype(np.float32)
    x_int = batch.x_int[:n_query, 0, :].cpu().numpy().astype(np.float32)
    y_obs = batch.y[:n_ctx, 0, 0].cpu().numpy().astype(np.float32)
    y_int = batch.target_y[:n_query, 0, 0].cpu().numpy().astype(np.float32)

    x_full = np.concatenate([x_obs, x_int], axis=0).astype(np.float32)
    y_full = np.concatenate([y_obs, y_int], axis=0).astype(np.float32)

    y_col = np.empty(num_samples, dtype=np.float32)
    y_col[:n_ctx] = y_full[:n_ctx]
    y_col[n_ctx:] = np.nan
    x_with_y = np.concatenate([x_full, y_col[:, None]], axis=1).astype(np.float32)

    contrast = (batch.target_y[:, 0, 0] - batch.y[:, 0, 0]).cpu().numpy().astype(np.float32)

    meta = dict(batch.task_meta)
    meta["n_ctx"] = n_ctx
    meta["cate_true_source"] = "paired_interventional_minus_observational"

    return {
        "X": x_with_y,
        "y": y_full[n_ctx:].astype(np.float32),
        "n_ctx": n_ctx,
        "cate_true": contrast[:num_samples].astype(np.float32),
        "adjacency": batch.adjacency.astype(np.int8),
        "task_meta": meta,
    }


@register_prior("do_pfn_scm")
class DoPfnSCMPrior(Prior):
    def sample(
        self,
        *,
        seed: int,
        num_samples: int = 2200,
        ctx_frac: float = 0.75,
        num_features: int = 6,
        num_unobserved: int = 1,
        K_max: int = 10,
        max_retries: int = 5,
        noise_dist: str = "gaussian",
        exo_dist: str = "gaussian",
        nonlins: str = "paper_text",
        **params: Any,
    ):
        params.pop("tag", None)
        return sample_do_pfn_studio_task(
            seed=seed,
            num_samples=num_samples,
            ctx_frac=ctx_frac,
            num_features=num_features,
            num_unobserved=num_unobserved,
            K_max=K_max,
            max_retries=max_retries,
            noise_dist=noise_dist,
            exo_dist=exo_dist,
            nonlins=nonlins,
            **params,
        )

    def sample_batch(
        self,
        *,
        batch_size: int,
        seed: int,
        num_samples: int = 2200,
        ctx_frac: float = 0.75,
        min_ctx: int = 10,
        vary_ctx_per_batch: bool = True,
        vary_num_features_per_batch: bool = True,
        num_features: int = 6,
        num_features_min: int = 1,
        num_features_max: int = 6,
        **params: Any,
    ):
        rng = np.random.default_rng(seed)

        if vary_ctx_per_batch:
            n_ctx = int(rng.integers(min_ctx, num_samples))
            ctx_frac = n_ctx / num_samples

        if vary_num_features_per_batch:
            num_features = int(rng.integers(num_features_min, num_features_max + 1))

        return [
            self.sample(
                seed=seed + i,
                num_samples=num_samples,
                ctx_frac=ctx_frac,
                num_features=num_features,
                **params,
            )
            for i in range(batch_size)
        ]