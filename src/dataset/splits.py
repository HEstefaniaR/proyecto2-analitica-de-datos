"""Genera los splits fijos train / val / test por caso (un caso = un paciente).

Se ejecuta una sola vez y su salida (`splits/splits.csv`) se versiona; el resto del proyecto solo
lee ese CSV. No depende del EDA: el tipo de fractura se deriva directamente de las máscaras.

    uv run python -m src.dataset.splits

Estratificación (`config.SPLIT_STRATA`, semilla `config.SEED`, fracciones `config.SPLIT_FRACTIONS`):
la cantidad de casos de val y test sale de las fracciones y se reparte entre estratos de forma
proporcional (mayor residuo, empates en el orden de `SPLIT_STRATA`); dentro de cada estrato los
casos se eligen al azar con la semilla.
"""
import numpy as np
import pandas as pd
import SimpleITK as sitk

from src import config

# Código de fragmento -> región: 1-10 sacro, 11-20 coxal izquierdo, 21-30 coxal derecho
REGIONS = {"sacrum": range(1, 11), "left_hipbone": range(11, 21), "right_hipbone": range(21, 31)}


def fracture_type(codes) -> str:
    """Tipo de fractura a partir de los códigos de fragmento presentes en la máscara.

    Más de un fragmento en una región = región fracturada. PRD: sin fractura dentro de un hueso;
    UHF / BHF: coxal unilateral / bilateral; SF: sacro; SHF: sacro y coxal.
    """
    n = {r: sum(int(c) in cods for c in codes) for r, cods in REGIONS.items()}
    hips = int(n["left_hipbone"] > 1) + int(n["right_hipbone"] > 1)
    if n["sacrum"] > 1:
        return "SF" if hips == 0 else "SHF"
    return ("PRD", "UHF", "BHF")[hips]


def list_cases() -> pd.DataFrame:
    """Un caso por máscara del dataset, con su tipo de fractura."""
    rows = []
    for path in sorted(config.LABEL_DIR.glob("*.mha")):
        mask = sitk.GetArrayViewFromImage(sitk.ReadImage(str(path)))
        codes = np.unique(mask)
        rows.append({"case_id": path.stem, "fracture_type": fracture_type(codes[codes > 0])})
    return pd.DataFrame(rows)


def make_splits(cases: pd.DataFrame) -> pd.DataFrame:
    """Asigna cada caso a train / val / test, estratificando por `config.SPLIT_STRATA`."""
    fractions, strata = config.SPLIT_FRACTIONS, config.SPLIT_STRATA
    cases = cases[["case_id", "fracture_type"]].sort_values("case_id").reset_index(drop=True)
    type_to_stratum = {t: e for e, types in strata.items() for t in types}
    cases["stratum"] = cases.fracture_type.map(type_to_stratum)
    assert cases.stratum.notna().all(), "Hay tipos de fractura sin estrato en SPLIT_STRATA"

    n = len(cases)
    n_split = {"test": round(n * fractions["test"]), "val": round(n * fractions["val"])}
    stratum_size = {e: int((cases.stratum == e).sum()) for e in strata}

    # Cupos por estrato: primero test y luego val, con lo que queda de cada estrato
    quotas = {}
    for s in ("test", "val"):
        cap = {e: stratum_size[e] - sum(q[e] for q in quotas.values()) for e in strata}
        exact = {e: stratum_size[e] * fractions[s] for e in strata}
        quota = {e: min(int(np.floor(exact[e])), cap[e]) for e in strata}
        order = sorted(strata, key=lambda e: (-(exact[e] - np.floor(exact[e])), list(strata).index(e)))
        while sum(quota.values()) < n_split[s]:
            for e in order:
                if sum(quota.values()) == n_split[s]:
                    break
                if quota[e] < cap[e]:
                    quota[e] += 1
        quotas[s] = quota

    rng = np.random.default_rng(config.SEED)
    assignment = {}
    for e in strata:
        ids = cases.case_id[cases.stratum == e].to_numpy()
        ids = ids[rng.permutation(len(ids))]
        t, v = quotas["test"][e], quotas["val"][e]
        assignment.update({c: "test" for c in ids[:t]})
        assignment.update({c: "val" for c in ids[t:t + v]})
        assignment.update({c: "train" for c in ids[t + v:]})
    cases["split"] = cases.case_id.map(assignment)

    splits = cases[["case_id", "split", "stratum", "fracture_type"]]
    assert splits.case_id.is_unique and len(splits) == n
    return splits


def main() -> None:
    splits = make_splits(list_cases())
    config.SPLITS_DIR.mkdir(exist_ok=True)
    out = config.SPLITS_DIR / "splits.csv"
    splits.to_csv(out, index=False)
    print(f"Splits guardados en {out} (semilla {config.SEED}):", splits.split.value_counts().to_dict())
    print(pd.crosstab(splits.stratum, splits.split, margins=True))


if __name__ == "__main__":
    main()
