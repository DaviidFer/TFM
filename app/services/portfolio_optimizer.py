"""
Optimizador hibrido GA + PSO para el PortfolioManagerProcess.

Este modulo encapsula toda la logica de optimizacion semanal de cartera:

1. Metricas de cartera (Sharpe neto, MDD, correlacion media).
2. Funcion de fitness unica:
       Fitness = Sharpe_neto - lambda_dd * MDD - lambda_corr * CorrMedia
3. Algoritmo Genetico (GA) que selecciona subconjuntos de traders sobre
   un cromosoma como lista variable de traders activos validos.
4. Particle Swarm Optimization (PSO) que asigna pesos dentro del subconjunto
   elegido (incluido peso de cash), con reparacion/proyeccion para cumplir
   max_weight_per_trader y max_cash_weight.
5. Orquestador `optimize_portfolio_ga_pso` que combina GA + PSO sobre los
   mejores `top_k_subsets_for_pso` cromosomas.

No depende de PPO, modelos preentrenados ni RL: trabaja directamente sobre
una matriz de retornos historica (T x N) y devuelve pesos validos.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Configuracion
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PortfolioOptimizerConfig:
    """
    Parametros fijos del optimizador hibrido GA + PSO (portfolio manager).

    Valores por defecto elegidos de forma explicita para produccion; no hay
    busqueda automatica ni calibracion en runtime. No incluye coste de transaccion
    ni hiperparametros de PPO.
    """

    # Modo y datos
    portfolio_manager_mode: str = "ga_pso"
    weekly_frequency: str = "W-FRI"
    lookback_weeks: int = 104

    # Restricciones de cartera
    min_selected_traders: int = 5
    max_selected_traders: int = 20
    max_weight_per_trader: float = 0.15
    min_live_weight: float = 0.02
    max_cash_weight: float = 1.0

    # Penalizaciones de la fitness
    lambda_dd: float = 1.0
    lambda_corr: float = 0.50

    # Parametros del Algoritmo Genetico
    ga_population_size: int = 80
    ga_generations: int = 80
    ga_tournament_size: int = 3
    ga_crossover_rate: float = 0.85
    ga_elitism: int = 4
    ga_early_stopping_generations: int = 20

    # Evaluacion rapida de pesos durante GA
    ga_weight_simulations: int = 20

    # Mutacion estructural
    ga_mutation_probability: float = 0.75
    ga_mutation_new_traders_min: int = 1
    ga_mutation_new_traders_max: int = 3
    ga_mutation_remove_max: int = 2
    ga_correlation_prune_threshold: float = 0.75

    # Cuantos subconjuntos pasan del GA al PSO
    top_k_subsets_for_pso: int = 10

    # Parametros del Particle Swarm Optimization
    pso_swarm_size: int = 40
    pso_iterations: int = 80
    pso_inertia_start: float = 0.85
    pso_inertia_end: float = 0.35
    pso_cognitive_coef: float = 1.6
    pso_social_coef: float = 1.4
    pso_early_stopping_iterations: int = 15

    # Reproducibilidad
    random_seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Metricas de cartera
# ---------------------------------------------------------------------------


_EPS = 1e-12


def compute_portfolio_returns(returns: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Retornos de una cartera ponderada a partir de una matriz (T, N) y pesos (N,).
    Si `weights` tiene longitud N+1, el ultimo componente se interpreta como cash
    (retorno cero) y se ignora salvo para escalar el resto.
    """
    if returns.size == 0:
        return np.zeros(0, dtype=float)
    n_traders = returns.shape[1]
    weights = np.asarray(weights, dtype=float)
    if weights.size == n_traders + 1:
        weights = weights[:n_traders]
    elif weights.size != n_traders:
        raise ValueError(
            f"weights tiene longitud {weights.size} incompatible con N={n_traders}"
        )
    return returns @ weights


def compute_sharpe_neto(portfolio_returns: np.ndarray) -> float:
    """
    Sharpe sin anualizar y con denominador estable. Si los retornos por periodo
    incluyen costes (coste neto en backtest) entonces es 'neto' por construccion.
    """
    arr = np.asarray(portfolio_returns, dtype=float)
    if arr.size < 2:
        return 0.0
    sigma = float(np.std(arr, ddof=0))
    if sigma <= _EPS:
        return 0.0
    return float(np.mean(arr) / sigma)


def compute_mdd(portfolio_returns: np.ndarray) -> float:
    """
    Maximum Drawdown sobre la curva acumulada de los retornos. Devuelve un
    valor positivo en tanto por uno (0.20 = caida maxima del 20%).
    """
    arr = np.asarray(portfolio_returns, dtype=float)
    if arr.size == 0:
        return 0.0
    equity = np.cumprod(1.0 + arr)
    running_max = np.maximum.accumulate(equity)
    drawdown = (equity - running_max) / np.maximum(running_max, _EPS)
    return float(-np.min(drawdown))


def compute_corr_media(
    returns: np.ndarray,
    selected_indices: Sequence[int],
    weights: np.ndarray | None = None,
) -> float:
    """
    Correlacion absoluta media entre los traders seleccionados.

    Si `weights` se proporciona y contiene los pesos de los traders activos
    (sin cash), se usa una version ponderada:
        CorrMedia_w = sum_{i<j} w_i * w_j * |corr(i,j)| / sum_{i<j} w_i * w_j
    En caso contrario se usa la media simple sobre el triangulo superior.
    """
    if len(selected_indices) < 2:
        return 0.0
    submatrix = returns[:, list(selected_indices)]
    if submatrix.shape[0] < 2:
        return 0.0
    std = np.std(submatrix, axis=0, ddof=0)
    if np.any(std <= _EPS):
        # Si una columna es plana, su correlacion con cualquier otra es 0.
        active_cols = std > _EPS
        if active_cols.sum() < 2:
            return 0.0
        submatrix = submatrix[:, active_cols]
        if weights is not None:
            weights = np.asarray(weights, dtype=float)[active_cols]
    corr = np.corrcoef(submatrix, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    n = corr.shape[0]
    if n < 2:
        return 0.0
    iu = np.triu_indices(n, k=1)
    abs_corr = np.abs(corr[iu])
    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        wi = weights[iu[0]]
        wj = weights[iu[1]]
        wsum = float(np.sum(wi * wj))
        if wsum > _EPS:
            return float(np.sum(wi * wj * abs_corr) / wsum)
    return float(np.mean(abs_corr))


def compute_fitness(
    sharpe_neto: float,
    mdd: float,
    corr_media: float,
    *,
    lambda_dd: float,
    lambda_corr: float,
) -> float:
    """
    Fitness unica:

        Fitness = Sharpe_neto - lambda_dd * MDD - lambda_corr * CorrMedia
    """
    return float(sharpe_neto - float(lambda_dd) * float(mdd) - float(lambda_corr) * float(corr_media))


# ---------------------------------------------------------------------------
# Reparaciones (cromosoma y pesos)
# ---------------------------------------------------------------------------


def repair_chromosome(
    chromosome: Sequence[int] | np.ndarray,
    *,
    universe_size: int,
    min_selected: int,
    max_selected: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Repara un cromosoma como lista variable de traders seleccionados.

    - Elimina duplicados y genes fuera del universo activo.
    - Garantiza `min_selected <= len(chromosome) <= max_selected`.
    - Anyade traders aleatorios del universo si faltan genes.
    """
    universe_size = max(0, int(universe_size))
    if universe_size <= 0:
        return np.zeros(0, dtype=int)

    eff_min = max(1, min(int(min_selected), universe_size))
    eff_max = max(eff_min, min(int(max_selected), universe_size))
    genes: List[int] = []
    seen: set[int] = set()
    for raw_gene in np.asarray(list(chromosome), dtype=int).tolist():
        gene = int(raw_gene)
        if gene < 0 or gene >= universe_size or gene in seen:
            continue
        genes.append(gene)
        seen.add(gene)

    if len(genes) > eff_max:
        picks = rng.choice(np.asarray(genes, dtype=int), size=eff_max, replace=False)
        genes = [int(v) for v in picks.tolist()]
        seen = set(genes)

    if len(genes) < eff_min:
        available = [idx for idx in range(universe_size) if idx not in seen]
        if available:
            need = min(eff_min - len(genes), len(available))
            picks = rng.choice(np.asarray(available, dtype=int), size=need, replace=False)
            genes.extend(int(v) for v in picks.tolist())

    return np.asarray(sorted(set(genes))[:eff_max], dtype=int)


def repair_weights(
    weights: np.ndarray,
    *,
    max_weight_per_trader: float,
    max_cash_weight: float,
    min_live_weight: float = 0.0,
) -> np.ndarray:
    """
    Proyecta un vector de pesos `[w_1, ..., w_K, w_cash]` al simplex valido:

    - Pesos no negativos.
    - Suma total exactamente 1.
    - Cada w_i (traders) no supera `max_weight_per_trader`.
    - El peso de cash no supera `max_cash_weight`.
    - Los pesos por debajo de `min_live_weight` se ponen a 0 (residuo va a cash).

    El algoritmo es una proyeccion iterativa: se clampa por encima, se renormaliza
    y se reinyecta el sobrante en cash hasta tope; si todavia hay sobrante, se
    distribuye entre traders activos respetando su tope.
    """
    w = np.asarray(weights, dtype=float).copy()
    if w.size < 1:
        return w
    cap_t = float(max_weight_per_trader)
    cap_c = float(max_cash_weight)
    floor_t = float(min_live_weight)

    # Asegurar no negatividad.
    w = np.maximum(w, 0.0)
    if float(np.sum(w)) <= _EPS:
        # Caso degenerado: todo a cash.
        out = np.zeros_like(w)
        out[-1] = 1.0
        return _enforce_caps_with_cash(out, cap_t=cap_t, cap_c=cap_c, floor_t=floor_t)

    w = w / float(np.sum(w))
    return _enforce_caps_with_cash(w, cap_t=cap_t, cap_c=cap_c, floor_t=floor_t)


def _enforce_caps_with_cash(
    w: np.ndarray,
    *,
    cap_t: float,
    cap_c: float,
    floor_t: float,
) -> np.ndarray:
    """
    Aplica los topes max_weight_per_trader y max_cash_weight. Cualquier exceso
    se mueve al lado opuesto. Si ni traders ni cash pueden absorberlo (por
    ejemplo K * cap_t + cap_c < 1) se da prioridad al simplex valido y se
    llena hasta los topes (la suma puede quedar < 1, en cuyo caso se
    devuelve esa configuracion limite tras renormalizar).
    """
    out = w.astype(float).copy()
    if out.size < 1:
        return out
    cash_idx = out.size - 1
    cap_t = float(max(0.0, min(1.0, cap_t)))
    cap_c = float(max(0.0, min(1.0, cap_c)))
    floor_t = float(max(0.0, floor_t))

    # Si floor > cap, dejar floor a 0 para evitar bloqueos.
    if floor_t >= cap_t:
        floor_t = 0.0

    # Aplicar suelo: pesos de traders por debajo de floor van a cash.
    trader_mask = np.zeros_like(out, dtype=bool)
    trader_mask[:cash_idx] = True
    if floor_t > 0.0:
        below = trader_mask & (out < floor_t)
        if np.any(below):
            out[cash_idx] += float(np.sum(out[below]))
            out[below] = 0.0

    for _ in range(50):
        # Cap por trader.
        trader_excess = 0.0
        for i in range(cash_idx):
            if out[i] > cap_t:
                trader_excess += out[i] - cap_t
                out[i] = cap_t
        if trader_excess > _EPS:
            out[cash_idx] += trader_excess
        # Cap de cash.
        if out[cash_idx] > cap_c:
            cash_excess = out[cash_idx] - cap_c
            out[cash_idx] = cap_c
            available = np.array(
                [max(cap_t - out[i], 0.0) for i in range(cash_idx)],
                dtype=float,
            )
            cap_room = float(np.sum(available))
            if cap_room <= _EPS:
                # No hay donde meter el sobrante: el simplex no se puede
                # cumplir con caps tan agresivos. Renormalizamos lo que hay.
                break
            scale = min(1.0, cash_excess / cap_room)
            out[:cash_idx] += available * scale
            cash_excess -= cap_room * scale
            if cash_excess > _EPS:
                # Sobra todavia: lo dejamos en cash (rompe cap minimamente).
                out[cash_idx] += cash_excess
        # Asegurar no negatividad.
        out = np.maximum(out, 0.0)
        total = float(np.sum(out))
        if total <= _EPS:
            out = np.zeros_like(out)
            out[cash_idx] = 1.0
            break
        # Si total != 1, renormalizar y volver a comprobar topes.
        if abs(total - 1.0) <= 1e-9:
            break
        out = out / total

    # Aplicar floor final: si tras la proyeccion algun trader queda muy bajo,
    # se manda a 0 y se reinyecta en cash respetando su cap.
    if floor_t > 0.0:
        below = trader_mask & (out > 0.0) & (out < floor_t)
        if np.any(below):
            slack = float(np.sum(out[below]))
            out[below] = 0.0
            room_cash = max(cap_c - out[cash_idx], 0.0)
            take_cash = min(slack, room_cash)
            out[cash_idx] += take_cash
            slack -= take_cash
            if slack > _EPS:
                # Reparte el restante en traders activos respetando cap.
                active_mask = trader_mask & (out > 0.0)
                if np.any(active_mask):
                    available = np.where(
                        active_mask,
                        np.maximum(cap_t - out, 0.0),
                        0.0,
                    )
                    cap_room = float(np.sum(available))
                    if cap_room > _EPS:
                        scale = min(1.0, slack / cap_room)
                        out += available * scale

    # Renormalizacion final por seguridad numerica.
    total = float(np.sum(out))
    if total > _EPS:
        out = out / total
    return out


# ---------------------------------------------------------------------------
# Evaluacion de subconjuntos (sin asignacion fina de pesos)
# ---------------------------------------------------------------------------


def _evaluate_subset_equal_weight(
    returns: np.ndarray,
    indices: Sequence[int],
    *,
    lambda_dd: float,
    lambda_corr: float,
) -> Tuple[float, float, float, float]:
    """Evalua un subconjunto con equiponderacion. Devuelve (fitness, sharpe, mdd, corr)."""
    if len(indices) == 0:
        return float("-inf"), 0.0, 0.0, 0.0
    sub = returns[:, list(indices)]
    n = sub.shape[1]
    weights = np.full(n, 1.0 / float(n), dtype=float)
    port_returns = sub @ weights
    sharpe = compute_sharpe_neto(port_returns)
    mdd = compute_mdd(port_returns)
    corr = compute_corr_media(returns, list(indices), weights=None)
    fitness = compute_fitness(sharpe, mdd, corr, lambda_dd=lambda_dd, lambda_corr=lambda_corr)
    return fitness, sharpe, mdd, corr


# ---------------------------------------------------------------------------
# Algoritmo Genetico para seleccion de subconjuntos
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GASubsetResult:
    chromosome: List[int]
    indices: List[int]
    fitness: float
    sharpe: float
    mdd: float
    corr_media: float


def _initial_population(
    n_assets: int,
    *,
    population_size: int,
    min_selected: int,
    max_selected: int,
    rng: np.random.Generator,
) -> List[np.ndarray]:
    eff_min = max(1, min(int(min_selected), n_assets))
    eff_max = max(eff_min, min(int(max_selected), n_assets))
    population: List[np.ndarray] = []
    for _ in range(int(population_size)):
        k = int(rng.integers(eff_min, eff_max + 1))
        idx = rng.choice(n_assets, size=k, replace=False)
        population.append(np.asarray(sorted(int(v) for v in idx.tolist()), dtype=int))
    return population


def _tournament_selection(
    population: Sequence[np.ndarray],
    fitnesses: np.ndarray,
    *,
    tournament_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    n = len(population)
    k = max(1, min(int(tournament_size), n))
    idx = rng.choice(n, size=k, replace=False)
    winner = idx[int(np.argmax(fitnesses[idx]))]
    return np.asarray(population[winner], dtype=int).copy()


def _structural_crossover(
    parent_a: np.ndarray,
    parent_b: np.ndarray,
    *,
    returns: np.ndarray,
    config: PortfolioOptimizerConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    union_pool = sorted({int(v) for v in parent_a.tolist()} | {int(v) for v in parent_b.tolist()})
    if not union_pool:
        return repair_chromosome(
            [],
            universe_size=returns.shape[1],
            min_selected=config.min_selected_traders,
            max_selected=config.max_selected_traders,
            rng=rng,
        )
    pool = _prune_correlation_redundancy(
        union_pool,
        returns=returns,
        threshold=float(config.ga_correlation_prune_threshold),
        min_selected=int(config.min_selected_traders),
        rng=rng,
    )
    if not pool:
        pool = list(union_pool)
    target_size = int(
        np.clip(
            int(round((len(parent_a) + len(parent_b)) / 2.0)) + int(rng.integers(-1, 2)),
            max(1, min(int(config.min_selected_traders), returns.shape[1])),
            max(1, min(int(config.max_selected_traders), returns.shape[1])),
        )
    )
    if len(pool) > target_size:
        picks = rng.choice(np.asarray(pool, dtype=int), size=target_size, replace=False)
        child = np.asarray(picks, dtype=int)
    else:
        child = np.asarray(pool, dtype=int)
    return repair_chromosome(
        child,
        universe_size=returns.shape[1],
        min_selected=config.min_selected_traders,
        max_selected=config.max_selected_traders,
        rng=rng,
    )


def _mutate_chromosome_structural(
    chromosome: np.ndarray,
    *,
    universe_size: int,
    config: PortfolioOptimizerConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    out = np.asarray(chromosome, dtype=int).copy()
    if universe_size <= 0 or rng.random() >= float(config.ga_mutation_probability):
        return out
    genes = {int(v) for v in out.tolist()}
    available = [idx for idx in range(int(universe_size)) if idx not in genes]

    if available:
        add_low = max(0, int(config.ga_mutation_new_traders_min))
        add_high = max(add_low, int(config.ga_mutation_new_traders_max))
        n_add = int(rng.integers(add_low, add_high + 1))
        if n_add > 0:
            picks = rng.choice(np.asarray(available, dtype=int), size=min(n_add, len(available)), replace=False)
            genes.update(int(v) for v in picks.tolist())
            available = [idx for idx in available if idx not in genes]

    removable = min(max(0, len(genes) - int(config.min_selected_traders)), int(config.ga_mutation_remove_max))
    if removable > 0 and rng.random() < 0.75:
        n_remove = int(rng.integers(0, removable + 1))
        if n_remove > 0:
            drops = rng.choice(np.asarray(sorted(genes), dtype=int), size=n_remove, replace=False)
            for gene in drops.tolist():
                genes.discard(int(gene))

    if genes and available and rng.random() < 0.50:
        drop_gene = int(rng.choice(np.asarray(sorted(genes), dtype=int)))
        add_gene = int(rng.choice(np.asarray(available, dtype=int)))
        genes.discard(drop_gene)
        genes.add(add_gene)

    return repair_chromosome(
        sorted(genes),
        universe_size=universe_size,
        min_selected=config.min_selected_traders,
        max_selected=config.max_selected_traders,
        rng=rng,
    )


def _pairwise_abs_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or b.size < 2:
        return 0.0
    if float(np.std(a, ddof=0)) <= _EPS or float(np.std(b, ddof=0)) <= _EPS:
        return 0.0
    corr = np.corrcoef(np.column_stack([a, b]), rowvar=False)[0, 1]
    if np.isnan(corr) or np.isinf(corr):
        return 0.0
    return float(abs(corr))


def _prune_correlation_redundancy(
    indices: Sequence[int],
    *,
    returns: np.ndarray,
    threshold: float,
    min_selected: int,
    rng: np.random.Generator,
) -> List[int]:
    if len(indices) <= 1 or threshold <= 0.0 or threshold >= 1.0:
        return list(indices)
    order = [int(v) for v in indices]
    rng.shuffle(order)
    kept: List[int] = []
    for idx in order:
        if len(kept) < int(min_selected):
            kept.append(int(idx))
            continue
        candidate_series = returns[:, int(idx)]
        is_redundant = False
        for existing in kept:
            if _pairwise_abs_corr(candidate_series, returns[:, int(existing)]) >= threshold:
                is_redundant = True
                break
        if not is_redundant:
            kept.append(int(idx))
    if len(kept) < min(int(min_selected), len(order)):
        for idx in order:
            if idx not in kept:
                kept.append(int(idx))
            if len(kept) >= min(int(min_selected), len(order)):
                break
    return kept


def _evaluate_subset_fast(
    returns: np.ndarray,
    indices: Sequence[int],
    *,
    config: PortfolioOptimizerConfig,
    rng: np.random.Generator,
) -> Tuple[float, float, float, float]:
    if len(indices) == 0:
        return float("-inf"), 0.0, 0.0, 0.0
    sub_returns = returns[:, list(indices)]
    n = sub_returns.shape[1]
    if n == 0:
        return float("-inf"), 0.0, 0.0, 0.0

    candidate_weights: List[np.ndarray] = []
    candidate_weights.append(np.append(np.full(n, 1.0 / float(n), dtype=float), 0.0))

    vol = np.std(sub_returns, axis=0, ddof=0)
    if np.any(vol > _EPS):
        inv_vol = np.where(vol > _EPS, 1.0 / np.maximum(vol, _EPS), 0.0)
        if float(np.sum(inv_vol)) > _EPS:
            inv_vol = inv_vol / float(np.sum(inv_vol))
            candidate_weights.append(np.append(inv_vol, 0.0))

    for _ in range(max(0, int(config.ga_weight_simulations))):
        candidate_weights.append(rng.random(n + 1))

    best_fit = float("-inf")
    best_sharpe = 0.0
    best_mdd = 0.0
    best_corr = 0.0
    for raw_weights in candidate_weights:
        repaired = repair_weights(
            raw_weights,
            max_weight_per_trader=config.max_weight_per_trader,
            max_cash_weight=config.max_cash_weight,
            min_live_weight=config.min_live_weight,
        )
        fit, sharpe, mdd, corr = _evaluate_weights(
            sub_returns,
            repaired,
            lambda_dd=config.lambda_dd,
            lambda_corr=config.lambda_corr,
        )
        if fit > best_fit:
            best_fit = float(fit)
            best_sharpe = float(sharpe)
            best_mdd = float(mdd)
            best_corr = float(corr)
    return best_fit, best_sharpe, best_mdd, best_corr


def genetic_select_subsets(
    returns: np.ndarray,
    *,
    config: PortfolioOptimizerConfig,
    rng: np.random.Generator | None = None,
) -> List[GASubsetResult]:
    """
    Ejecuta el GA y devuelve los `top_k_subsets_for_pso` mejores subconjuntos
    (cromosomas validos) ordenados por fitness descendente. Cada cromosoma se
    representa como lista variable de traders del universo activo valido.

    Si la matriz de retornos es vacia o tiene 0 columnas, devuelve lista vacia.
    """
    if returns.size == 0 or returns.shape[1] == 0:
        return []
    rng = rng if rng is not None else np.random.default_rng(int(config.random_seed))
    n_assets = returns.shape[1]

    population = _initial_population(
        n_assets,
        population_size=config.ga_population_size,
        min_selected=config.min_selected_traders,
        max_selected=config.max_selected_traders,
        rng=rng,
    )

    cache: Dict[bytes, Tuple[float, float, float, float]] = {}

    def evaluate(chrom: np.ndarray) -> Tuple[float, float, float, float]:
        repaired = repair_chromosome(
            chrom,
            universe_size=n_assets,
            min_selected=config.min_selected_traders,
            max_selected=config.max_selected_traders,
            rng=rng,
        )
        key = repaired.tobytes()
        cached = cache.get(key)
        if cached is not None:
            return cached
        indices = [int(v) for v in repaired.tolist()]
        result = _evaluate_subset_fast(
            returns,
            indices,
            config=config,
            rng=rng,
        )
        cache[key] = result
        return result

    best_fitness_history: List[float] = []
    no_improve = 0
    best_overall_fitness = float("-inf")

    fitnesses = np.array([evaluate(ind)[0] for ind in population], dtype=float)

    for _ in range(int(config.ga_generations)):
        order = np.argsort(-fitnesses)
        elite_count = min(int(config.ga_elitism), len(population))
        elites = [np.asarray(population[int(idx)], dtype=int).copy() for idx in order[:elite_count]]

        children: List[np.ndarray] = []
        while len(children) < (len(population) - elite_count):
            parent_a = _tournament_selection(
                population, fitnesses, tournament_size=config.ga_tournament_size, rng=rng
            )
            parent_b = _tournament_selection(
                population, fitnesses, tournament_size=config.ga_tournament_size, rng=rng
            )
            if rng.random() < float(config.ga_crossover_rate):
                child = _structural_crossover(parent_a, parent_b, returns=returns, config=config, rng=rng)
            else:
                child = parent_a.copy()
            child = _mutate_chromosome_structural(
                child,
                universe_size=n_assets,
                config=config,
                rng=rng,
            )
            child = repair_chromosome(
                child,
                universe_size=n_assets,
                min_selected=config.min_selected_traders,
                max_selected=config.max_selected_traders,
                rng=rng,
            )
            children.append(child)

        population = elites + children
        fitnesses = np.array([evaluate(ind)[0] for ind in population], dtype=float)

        gen_best = float(np.max(fitnesses))
        best_fitness_history.append(gen_best)
        if gen_best > best_overall_fitness + 1e-9:
            best_overall_fitness = gen_best
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= int(config.ga_early_stopping_generations):
            break

    seen: Dict[bytes, bool] = {}
    ordered: List[Tuple[float, np.ndarray]] = []
    final_order = np.argsort(-fitnesses)
    for idx in final_order:
        chrom = repair_chromosome(
            population[int(idx)],
            universe_size=n_assets,
            min_selected=config.min_selected_traders,
            max_selected=config.max_selected_traders,
            rng=rng,
        )
        key = chrom.tobytes()
        if key in seen:
            continue
        seen[key] = True
        ordered.append((float(fitnesses[idx]), chrom.copy()))
        if len(ordered) >= int(config.top_k_subsets_for_pso):
            break

    out: List[GASubsetResult] = []
    for fit, chrom in ordered:
        indices = [int(i) for i in chrom.tolist()]
        eval_fit, sharpe, mdd, corr = evaluate(chrom)
        out.append(
            GASubsetResult(
                chromosome=[int(v) for v in chrom.tolist()],
                indices=indices,
                fitness=float(eval_fit),
                sharpe=float(sharpe),
                mdd=float(mdd),
                corr_media=float(corr),
            )
        )
    return out


# ---------------------------------------------------------------------------
# Particle Swarm Optimization para asignar pesos
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PSOResult:
    weights: np.ndarray  # tamanyo K + 1 (ultimo = cash)
    fitness: float
    sharpe: float
    mdd: float
    corr_media: float
    iterations_run: int


def _evaluate_weights(
    returns_subset: np.ndarray,
    weights_with_cash: np.ndarray,
    *,
    lambda_dd: float,
    lambda_corr: float,
) -> Tuple[float, float, float, float]:
    n_traders = returns_subset.shape[1]
    weights_traders = weights_with_cash[:n_traders]
    port_returns = returns_subset @ weights_traders
    sharpe = compute_sharpe_neto(port_returns)
    mdd = compute_mdd(port_returns)
    corr = compute_corr_media(
        returns_subset,
        list(range(n_traders)),
        weights=weights_traders,
    )
    fitness = compute_fitness(sharpe, mdd, corr, lambda_dd=lambda_dd, lambda_corr=lambda_corr)
    return fitness, sharpe, mdd, corr


def pso_optimize_weights(
    returns_subset: np.ndarray,
    *,
    config: PortfolioOptimizerConfig,
    rng: np.random.Generator | None = None,
) -> PSOResult:
    """
    PSO clasico con reparacion. `returns_subset` tiene shape (T, K). El
    espacio de busqueda son vectores `[w_1, ..., w_K, w_cash]` proyectados
    al simplex con `repair_weights`.

    Si K == 0 devuelve un PSOResult degenerado con todo en cash.
    """
    rng = rng if rng is not None else np.random.default_rng(int(config.random_seed))
    if returns_subset.shape[1] == 0:
        empty = np.array([1.0], dtype=float)
        return PSOResult(
            weights=empty,
            fitness=0.0,
            sharpe=0.0,
            mdd=0.0,
            corr_media=0.0,
            iterations_run=0,
        )

    K = returns_subset.shape[1]
    dim = K + 1  # ultima dimension = cash
    swarm_size = int(config.pso_swarm_size)
    iterations = int(config.pso_iterations)

    positions = rng.random((swarm_size, dim))
    repaired = np.zeros_like(positions)
    for i in range(swarm_size):
        repaired[i] = repair_weights(
            positions[i],
            max_weight_per_trader=config.max_weight_per_trader,
            max_cash_weight=config.max_cash_weight,
            min_live_weight=config.min_live_weight,
        )
    positions = repaired
    velocities = (rng.random((swarm_size, dim)) - 0.5) * 0.2

    personal_best = positions.copy()
    personal_fitness = np.full(swarm_size, float("-inf"))
    fit_meta = np.zeros((swarm_size, 3))  # sharpe, mdd, corr para el global best
    for i in range(swarm_size):
        fit, sh, dd, co = _evaluate_weights(
            returns_subset,
            positions[i],
            lambda_dd=config.lambda_dd,
            lambda_corr=config.lambda_corr,
        )
        personal_fitness[i] = fit
        fit_meta[i] = (sh, dd, co)

    g_idx = int(np.argmax(personal_fitness))
    global_best = personal_best[g_idx].copy()
    global_best_fitness = float(personal_fitness[g_idx])
    global_best_meta = fit_meta[g_idx].copy()

    no_improve = 0
    iterations_run = 0
    for it in range(iterations):
        iterations_run = it + 1
        inertia = float(
            config.pso_inertia_start
            + (config.pso_inertia_end - config.pso_inertia_start) * (it / max(1, iterations - 1))
        )
        r1 = rng.random((swarm_size, dim))
        r2 = rng.random((swarm_size, dim))
        velocities = (
            inertia * velocities
            + float(config.pso_cognitive_coef) * r1 * (personal_best - positions)
            + float(config.pso_social_coef) * r2 * (global_best - positions)
        )
        positions = positions + velocities
        improved = False
        for i in range(swarm_size):
            positions[i] = repair_weights(
                positions[i],
                max_weight_per_trader=config.max_weight_per_trader,
                max_cash_weight=config.max_cash_weight,
                min_live_weight=config.min_live_weight,
            )
            fit, sh, dd, co = _evaluate_weights(
                returns_subset,
                positions[i],
                lambda_dd=config.lambda_dd,
                lambda_corr=config.lambda_corr,
            )
            if fit > personal_fitness[i]:
                personal_fitness[i] = fit
                personal_best[i] = positions[i].copy()
                fit_meta[i] = (sh, dd, co)
                if fit > global_best_fitness + 1e-9:
                    global_best_fitness = fit
                    global_best = positions[i].copy()
                    global_best_meta = np.array([sh, dd, co], dtype=float)
                    improved = True
        if improved:
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= int(config.pso_early_stopping_iterations):
            break

    return PSOResult(
        weights=global_best,
        fitness=float(global_best_fitness),
        sharpe=float(global_best_meta[0]),
        mdd=float(global_best_meta[1]),
        corr_media=float(global_best_meta[2]),
        iterations_run=int(iterations_run),
    )


# ---------------------------------------------------------------------------
# Orquestador GA + PSO
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OptimizationResult:
    """
    Salida del orquestador GA+PSO.

    `selected_indices` indexa las columnas de la matriz de retornos original.
    `weights` tiene longitud K (solo traders); `cash_weight` es escalar.
    """

    selected_indices: List[int]
    weights: np.ndarray
    cash_weight: float
    fitness: float
    sharpe_neto: float
    mdd: float
    corr_media: float
    ga_top_subsets: List[GASubsetResult] = field(default_factory=list)
    pso_iterations: int = 0
    status: str = "ok"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "selected_indices": list(self.selected_indices),
            "weights": [float(w) for w in self.weights.tolist()],
            "cash_weight": float(self.cash_weight),
            "fitness": float(self.fitness),
            "sharpe_neto": float(self.sharpe_neto),
            "mdd": float(self.mdd),
            "corr_media": float(self.corr_media),
            "pso_iterations": int(self.pso_iterations),
            "status": str(self.status),
            "ga_top_subsets": [
                {
                    "chromosome": list(sub.chromosome),
                    "indices": list(sub.indices),
                    "fitness": float(sub.fitness),
                    "sharpe": float(sub.sharpe),
                    "mdd": float(sub.mdd),
                    "corr_media": float(sub.corr_media),
                }
                for sub in self.ga_top_subsets
            ],
        }


def equal_weight_fitness(
    returns: np.ndarray,
    *,
    lambda_dd: float,
    lambda_corr: float,
) -> Tuple[float, float, float, float]:
    """Baseline: equiponderacion sobre todos los traders disponibles (sin cash)."""
    if returns.size == 0 or returns.shape[1] == 0:
        return 0.0, 0.0, 0.0, 0.0
    return _evaluate_subset_equal_weight(
        returns,
        list(range(returns.shape[1])),
        lambda_dd=lambda_dd,
        lambda_corr=lambda_corr,
    )


def optimize_portfolio_ga_pso(
    returns: np.ndarray,
    *,
    config: PortfolioOptimizerConfig,
) -> OptimizationResult:
    """
    Pipeline completo: GA elige top_k subconjuntos y el PSO afina pesos dentro
    de cada uno. Si el universo activo valido es pequeno (<= max_selected),
    se optimiza directamente ese conjunto con PSO.
    """
    if returns.size == 0 or returns.shape[1] == 0:
        return OptimizationResult(
            selected_indices=[],
            weights=np.zeros(0, dtype=float),
            cash_weight=1.0,
            fitness=0.0,
            sharpe_neto=0.0,
            mdd=0.0,
            corr_media=0.0,
            ga_top_subsets=[],
            pso_iterations=0,
            status="empty",
        )

    n_assets = returns.shape[1]
    if n_assets < int(config.min_selected_traders):
        return OptimizationResult(
            selected_indices=[],
            weights=np.zeros(0, dtype=float),
            cash_weight=1.0,
            fitness=0.0,
            sharpe_neto=0.0,
            mdd=0.0,
            corr_media=0.0,
            ga_top_subsets=[],
            pso_iterations=0,
            status="not_enough_valid_traders",
        )

    rng = np.random.default_rng(int(config.random_seed))
    if n_assets <= int(config.max_selected_traders):
        direct_indices = list(range(n_assets))
        direct_fit, direct_sharpe, direct_mdd, direct_corr = _evaluate_subset_fast(
            returns,
            direct_indices,
            config=config,
            rng=rng,
        )
        direct_subset = GASubsetResult(
            chromosome=[int(v) for v in direct_indices],
            indices=direct_indices,
            fitness=float(direct_fit),
            sharpe=float(direct_sharpe),
            mdd=float(direct_mdd),
            corr_media=float(direct_corr),
        )
        pso_out = pso_optimize_weights(returns[:, direct_indices], config=config, rng=rng)
        return OptimizationResult(
            selected_indices=direct_indices,
            weights=pso_out.weights[:-1].copy(),
            cash_weight=float(pso_out.weights[-1]),
            fitness=float(pso_out.fitness),
            sharpe_neto=float(pso_out.sharpe),
            mdd=float(pso_out.mdd),
            corr_media=float(pso_out.corr_media),
            ga_top_subsets=[direct_subset],
            pso_iterations=int(pso_out.iterations_run),
            status="ok",
        )

    top_subsets = genetic_select_subsets(returns, config=config, rng=rng)
    if not top_subsets:
        return OptimizationResult(
            selected_indices=[],
            weights=np.zeros(0, dtype=float),
            cash_weight=1.0,
            fitness=0.0,
            sharpe_neto=0.0,
            mdd=0.0,
            corr_media=0.0,
            ga_top_subsets=[],
            pso_iterations=0,
            status="ga_empty",
        )

    best_fitness = float("-inf")
    best_indices: List[int] = []
    best_weights = np.zeros(0, dtype=float)
    best_cash = 1.0
    best_sharpe = 0.0
    best_mdd = 0.0
    best_corr = 0.0
    total_iters = 0

    for subset in top_subsets:
        if len(subset.indices) == 0:
            continue
        sub_returns = returns[:, subset.indices]
        pso_out = pso_optimize_weights(sub_returns, config=config, rng=rng)
        total_iters += int(pso_out.iterations_run)
        if pso_out.fitness > best_fitness:
            best_fitness = float(pso_out.fitness)
            best_indices = list(subset.indices)
            best_weights = pso_out.weights[:-1].copy()
            best_cash = float(pso_out.weights[-1])
            best_sharpe = float(pso_out.sharpe)
            best_mdd = float(pso_out.mdd)
            best_corr = float(pso_out.corr_media)

    if best_fitness == float("-inf"):
        return OptimizationResult(
            selected_indices=[],
            weights=np.zeros(0, dtype=float),
            cash_weight=1.0,
            fitness=0.0,
            sharpe_neto=0.0,
            mdd=0.0,
            corr_media=0.0,
            ga_top_subsets=top_subsets,
            pso_iterations=total_iters,
            status="pso_empty",
        )

    return OptimizationResult(
        selected_indices=best_indices,
        weights=best_weights,
        cash_weight=best_cash,
        fitness=best_fitness,
        sharpe_neto=best_sharpe,
        mdd=best_mdd,
        corr_media=best_corr,
        ga_top_subsets=top_subsets,
        pso_iterations=total_iters,
        status="ok",
    )
