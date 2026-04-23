# ============================================================
# Algoritmo genético tipo SQX - Multi-Seed
# Átomos (bins y binarias) + evolución de reglas AND
# Output: (df_alcistas_ga, df_bajistas_ga) homogéneo
# ============================================================

import numpy as np
import pandas as pd
import random


def _build_genetic_rules_sqx_single_seed(
    data: pd.DataFrame,
    target_col: str = "Target",
    exclude_cols=None,
    n_bins: int = 4,
    min_coverage: int = 100,
    top_k_features: int = 200,
    max_atoms: int = 350,
    population_size: int = 180,
    n_generations: int = 50,
    max_rule_len: int = 2,
    tournament_k: int = 4,
    mutation_rate: float = 0.30,
    crossover_rate: float = 0.80,
    elite_frac: float = 0.10,
    random_state: int = 42
):
    """
    Genera reglas con un algoritmo genético para UNA sola semilla.
    Átomos = condiciones simples (bins o binarias); evolución por torneo, cruce y mutación.
    """
    if exclude_cols is None:
        exclude_cols = ["open", "high", "low", "close", "Target", "Return"]

    random.seed(random_state)
    np.random.seed(random_state)

    df = data.copy()

    if target_col not in df.columns:
        raise ValueError(f"No existe la columna target '{target_col}'.")

    if max_rule_len < 1:
        raise ValueError("max_rule_len debe ser >= 1.")
    if population_size < 4:
        raise ValueError("population_size debe ser >= 4.")
    if n_generations < 1:
        raise ValueError("n_generations debe ser >= 1.")
    if not (0 < elite_frac < 1):
        raise ValueError("elite_frac debe estar en (0,1).")
    if not (0 <= mutation_rate <= 1):
        raise ValueError("mutation_rate debe estar en [0,1].")
    if not (0 <= crossover_rate <= 1):
        raise ValueError("crossover_rate debe estar en [0,1].")

    y = pd.to_numeric(df[target_col], errors="coerce")
    global_mean = float(y.mean())

    candidate_cols = [
        c for c in df.columns
        if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])
    ]
    if len(candidate_cols) == 0:
        raise ValueError("No hay columnas numéricas candidatas.")

    corrs = df[candidate_cols].corrwith(y, method="spearman").abs().fillna(0.0)
    feature_cols = corrs.sort_values(ascending=False).head(min(top_k_features, len(corrs))).index.tolist()

    def _is_binary_01(s):
        vals = pd.Series(s).dropna().unique()
        if len(vals) == 0:
            return False
        return set(vals).issubset({0, 1})

    def _mask_for_bin(values, left, right):
        if np.isclose(left, right):
            return np.isfinite(values) & np.isclose(values, left)
        return np.isfinite(values) & (values > left) & (values <= right)

    def _rule_string(col, left, right):
        l = format(float(left), ".12g")
        r = format(float(right), ".12g")
        if np.isclose(left, right):
            return f"({col} == {l})"
        return f"({col} > {l}) & ({col} <= {r})"

    # Átomos de reglas
    atoms = []
    for col in feature_cols:
        s = pd.to_numeric(df[col], errors="coerce")
        dfc = pd.DataFrame({"x": s, "y": y}).dropna()

        if dfc.empty or dfc["x"].nunique(dropna=True) < 2:
            continue

        if _is_binary_01(dfc["x"]):
            for val in [0.0, 1.0]:
                mask = _mask_for_bin(s.to_numpy(dtype=float), val, val)
                idx = np.flatnonzero(mask)

                if len(idx) < min_coverage:
                    continue

                yt = y.iloc[idx].dropna()
                if yt.empty:
                    continue

                atoms.append({
                    "indicator": col,
                    "label": f"[{int(val)}, {int(val)}]",
                    "left": val,
                    "right": val,
                    "mask": mask,
                    "indices": idx,
                    "rule": _rule_string(col, val, val),
                    "score": abs(float(yt.mean()) - global_mean) * np.log1p(len(idx))
                })
        else:
            q = min(n_bins, int(dfc["x"].nunique(dropna=True)))
            try:
                bins = pd.qcut(dfc["x"], q=q, duplicates="drop")
            except Exception:
                continue

            intervals = pd.Series(bins).dropna().unique()
            full_vals = s.to_numpy(dtype=float)

            for interval in intervals:
                left = float(interval.left)
                right = float(interval.right)
                mask = _mask_for_bin(full_vals, left, right)
                idx = np.flatnonzero(mask)

                if len(idx) < min_coverage:
                    continue

                yt = y.iloc[idx].dropna()
                if yt.empty:
                    continue

                atoms.append({
                    "indicator": col,
                    "label": f"[{left}, {right}]",
                    "left": left,
                    "right": right,
                    "mask": mask,
                    "indices": idx,
                    "rule": _rule_string(col, left, right),
                    "score": abs(float(yt.mean()) - global_mean) * np.log1p(len(idx))
                })

    if len(atoms) == 0:
        empty = pd.DataFrame(columns=["indicators", "bin_labels", "coverage", "indices", "target_promedio", "regla"])
        return empty.copy(), empty.copy()

    atoms = sorted(atoms, key=lambda d: d["score"], reverse=True)[:max_atoms]

    cache = {}

    def _normalize_individual(ind):
        seen_atoms = set()
        out = []
        used_indicators = set()

        for a in ind:
            if a in seen_atoms:
                continue
            ind_name = atoms[a]["indicator"]
            if ind_name in used_indicators:
                continue
            seen_atoms.add(a)
            used_indicators.add(ind_name)
            out.append(a)

        return tuple(sorted(out))

    def _evaluate(ind):
        ind = _normalize_individual(ind)

        if len(ind) == 0:
            return None

        if ind in cache:
            return cache[ind]

        mask = np.ones(len(df), dtype=bool)
        for a in ind:
            mask &= atoms[a]["mask"]
            if not mask.any():
                cache[ind] = None
                return None

        idx = np.flatnonzero(mask)
        coverage = int(idx.size)

        if coverage < min_coverage:
            cache[ind] = None
            return None

        yt = y.iloc[idx].dropna()
        if yt.empty:
            cache[ind] = None
            return None

        mean_t = float(yt.mean())
        fitness = abs(mean_t - global_mean) * np.log1p(coverage) * (0.97 ** (len(ind) - 1))

        row = {
            "indicators": tuple(atoms[a]["indicator"] for a in ind),
            "bin_labels": tuple(atoms[a]["label"] for a in ind),
            "coverage": coverage,
            "indices": idx,
            "target_promedio": mean_t,
            "regla": " & ".join(atoms[a]["rule"] for a in ind),
            "fitness": fitness
        }
        cache[ind] = row
        return row

    all_atom_ids = list(range(len(atoms)))

    def _random_individual():
        k = random.randint(1, max_rule_len)
        shuffled = all_atom_ids.copy()
        random.shuffle(shuffled)

        chosen = []
        used_feats = set()
        for a in shuffled:
            f = atoms[a]["indicator"]
            if f in used_feats:
                continue
            chosen.append(a)
            used_feats.add(f)
            if len(chosen) == k:
                break

        return _normalize_individual(chosen)

    population = []
    while len(population) < population_size:
        ind = _random_individual()
        if len(ind) >= 1:
            population.append(ind)

    seen_rules = {}

    def _tournament(pop):
        cand = random.sample(pop, min(tournament_k, len(pop)))
        scored = []

        for ind in cand:
            ev = _evaluate(ind)
            if ev is not None:
                scored.append((ev["fitness"], ind))

        if len(scored) == 0:
            return random.choice(pop)

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1]

    def _crossover(p1, p2):
        pool = list(set(p1).union(set(p2)))
        random.shuffle(pool)

        chosen = []
        used_feats = set()
        k = random.randint(1, max_rule_len)

        for a in pool:
            f = atoms[a]["indicator"]
            if f in used_feats:
                continue
            chosen.append(a)
            used_feats.add(f)
            if len(chosen) == k:
                break

        return _normalize_individual(chosen)

    def _mutate(ind):
        ind = list(ind)
        used_feats = {atoms[a]["indicator"] for a in ind}

        op = random.choice(["replace", "add", "drop"])

        if op == "replace" and len(ind) > 0:
            pos = random.randrange(len(ind))
            old_feat = atoms[ind[pos]]["indicator"]
            candidates = [
                a for a in all_atom_ids
                if atoms[a]["indicator"] != old_feat
                and atoms[a]["indicator"] not in (used_feats - {old_feat})
            ]
            if len(candidates) > 0:
                ind[pos] = random.choice(candidates)

        elif op == "add" and len(ind) < max_rule_len:
            candidates = [a for a in all_atom_ids if atoms[a]["indicator"] not in used_feats]
            if len(candidates) > 0:
                ind.append(random.choice(candidates))

        elif op == "drop" and len(ind) > 1:
            pos = random.randrange(len(ind))
            ind.pop(pos)

        return _normalize_individual(ind)

    elite_n = max(5, int(round(population_size * elite_frac)))

    for _ in range(n_generations):
        evaluated = []
        for ind in population:
            ev = _evaluate(ind)
            if ev is not None:
                evaluated.append((ev["fitness"], ind, ev))
                seen_rules[ev["regla"]] = ev

        evaluated.sort(key=lambda x: x[0], reverse=True)
        elites = [x[1] for x in evaluated[:elite_n]]

        new_population = elites.copy()

        while len(new_population) < population_size:
            p1 = _tournament(population)
            child = p1

            if random.random() < crossover_rate:
                p2 = _tournament(population)
                child = _crossover(p1, p2)

            if random.random() < mutation_rate:
                child = _mutate(child)

            if len(child) >= 1:
                new_population.append(child)

        population = new_population[:population_size]

    out = pd.DataFrame(list(seen_rules.values()))
    if out.empty:
        empty = pd.DataFrame(columns=["indicators", "bin_labels", "coverage", "indices", "target_promedio", "regla"])
        return empty.copy(), empty.copy()

    out = out.drop_duplicates(subset=["regla"]).reset_index(drop=True)

    df_alcistas_ga = out[out["target_promedio"] > 0].copy()
    df_bajistas_ga = out[out["target_promedio"] < 0].copy()

    if not df_alcistas_ga.empty:
        df_alcistas_ga = df_alcistas_ga.sort_values(
            by=["target_promedio", "coverage"], ascending=[False, False]
        ).reset_index(drop=True)

    if not df_bajistas_ga.empty:
        df_bajistas_ga = df_bajistas_ga.sort_values(
            by=["target_promedio", "coverage"], ascending=[True, False]
        ).reset_index(drop=True)

    wanted_cols = ["indicators", "bin_labels", "coverage", "indices", "target_promedio", "regla"]
    return df_alcistas_ga[wanted_cols], df_bajistas_ga[wanted_cols]


def build_genetic_rules_sqx_multiseed(
    data: pd.DataFrame,
    target_col: str = "Target",
    exclude_cols=None,
    n_bins: int = 4,
    min_coverage: int = 100,
    top_k_features: int = 200,
    max_atoms: int = 350,
    population_size: int = 180,
    n_generations: int = 50,
    max_rule_len: int = 2,
    tournament_k: int = 4,
    mutation_rate: float = 0.30,
    crossover_rate: float = 0.80,
    elite_frac: float = 0.10,
    start_random_state: int = 1,
    target_n_rules: int = 3000,
    progress_every: int = 25
):
    """
    Multi-seed: itera sobre random_state hasta acumular target_n_rules reglas únicas.
    Devuelve (df_alcistas_ga, df_bajistas_ga).
    """
    if target_n_rules < 1:
        raise ValueError("target_n_rules debe ser >= 1")

    all_rules = []
    seen = set()
    rs = start_random_state

    while len(seen) < target_n_rules:
        df_alc_seed, df_baj_seed = _build_genetic_rules_sqx_single_seed(
            data=data,
            target_col=target_col,
            exclude_cols=exclude_cols,
            n_bins=n_bins,
            min_coverage=min_coverage,
            top_k_features=top_k_features,
            max_atoms=max_atoms,
            population_size=population_size,
            n_generations=n_generations,
            max_rule_len=max_rule_len,
            tournament_k=tournament_k,
            mutation_rate=mutation_rate,
            crossover_rate=crossover_rate,
            elite_frac=elite_frac,
            random_state=rs
        )

        out_seed = pd.concat([df_alc_seed, df_baj_seed], axis=0, ignore_index=True)

        if not out_seed.empty:
            for _, row in out_seed.iterrows():
                regla = row["regla"]
                if regla not in seen:
                    seen.add(regla)
                    all_rules.append(row.to_dict())

        if progress_every and rs % progress_every == 0:
            print(f"random_state={rs} | reglas únicas acumuladas={len(seen)}")

        rs += 1

    out = pd.DataFrame(all_rules).drop_duplicates(subset=["regla"]).reset_index(drop=True)

    if out.empty:
        empty = pd.DataFrame(columns=["indicators", "bin_labels", "coverage", "indices", "target_promedio", "regla"])
        return empty.copy(), empty.copy()

    df_alcistas_ga = out[out["target_promedio"] > 0].copy()
    df_bajistas_ga = out[out["target_promedio"] < 0].copy()

    if not df_alcistas_ga.empty:
        df_alcistas_ga = df_alcistas_ga.sort_values(
            by=["target_promedio", "coverage"], ascending=[False, False]
        ).reset_index(drop=True)

    if not df_bajistas_ga.empty:
        df_bajistas_ga = df_bajistas_ga.sort_values(
            by=["target_promedio", "coverage"], ascending=[True, False]
        ).reset_index(drop=True)

    wanted_cols = ["indicators", "bin_labels", "coverage", "indices", "target_promedio", "regla"]
    return df_alcistas_ga[wanted_cols], df_bajistas_ga[wanted_cols]


def run_genetico_rules(data: pd.DataFrame, target_col: str = "Target", exclude_cols=None, **kwargs):
    """Alias para build_genetic_rules_sqx_multiseed (compatibilidad notebook)."""
    return build_genetic_rules_sqx_multiseed(
        data=data,
        target_col=target_col,
        exclude_cols=exclude_cols,
        **kwargs
    )
