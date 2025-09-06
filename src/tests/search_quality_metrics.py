import os
import json
from datetime import datetime

import numpy as np
import pandas as pd
import psycopg2
from dotenv import load_dotenv
import argparse


# Minimal, single-purpose script:
# - Pull a sample of products with text embeddings
# - Compute Hybrid Recall@{1,5,10} using (country_code, city_code) as filters
#   and collection_section as the relevance proxy
# - Compute Filter-Separation Score for collection_section and store_name
# - Save a JSON report under src/tests/metrics with a timestamped filename


def _db_conn():
    load_dotenv()
    return psycopg2.connect(
        user=os.getenv("user"),
        password=os.getenv("password"),
        host=os.getenv("host"),
        port=os.getenv("port"),
        dbname=os.getenv("dbname"),
    )


def _parse_vec(cell):
    if cell is None:
        return None
    s = str(cell).strip()
    if not s or s == '[]':
        return None
    try:
        return np.asarray(json.loads(s), dtype=np.float32)
    except Exception:
        return None


def _normalize_rows(mat: np.ndarray) -> np.ndarray:
    if mat.size == 0:
        return mat
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8
    return mat / norms


def _vectors_from_series(series: pd.Series):
    parsed = series.map(_parse_vec).tolist()
    idx = np.array([i for i, v in enumerate(parsed) if v is not None], dtype=np.int64)
    if len(idx) == 0:
        return idx, np.zeros((0, 0), dtype=np.float32)
    dim = len(parsed[idx[0]])
    mat = np.zeros((len(idx), dim), dtype=np.float32)
    for j, i in enumerate(idx):
        mat[j] = parsed[i]
    return idx, mat


def _filter_separation(vectors: np.ndarray, labels: np.ndarray, seed: int | None = None):
    if len(vectors) < 3:
        return {}
    rs = np.random.RandomState(seed if seed is not None else 42)
    vectors = _normalize_rows(vectors)

    # Intra-category similarities (sampled pairs per label)
    intra_sims = []
    unique = np.unique(labels)
    for lab in unique:
        group_idx = np.where(labels == lab)[0]
        if len(group_idx) < 2:
            continue
        # Sample up to 300 pairs per label
        pairs = []
        for _ in range(min(300, len(group_idx) * (len(group_idx) - 1) // 2)):
            i, j = rs.choice(group_idx, size=2, replace=False)
            pairs.append((i, j))
        if not pairs:
            continue
        a = vectors[[i for i, _ in pairs]]
        b = vectors[[j for _, j in pairs]]
        sims = np.sum(a * b, axis=1)
        intra_sims.extend(sims.tolist())

    # Inter-category similarities (sampled pairs across different labels)
    inter_sims = []
    # Sample up to 5000 cross-label pairs
    for _ in range(5000):
        la, lb = rs.choice(unique, size=2, replace=False)
        ia = rs.choice(np.where(labels == la)[0])
        ib = rs.choice(np.where(labels == lb)[0])
        inter_sims.append(float(np.dot(vectors[ia], vectors[ib])))

    if not intra_sims or not inter_sims:
        return {}

    intra_mean = float(np.mean(intra_sims))
    inter_mean = float(np.mean(inter_sims))
    return {
        'intra_mean': intra_mean,
        'inter_mean': inter_mean,
        'separation': float(intra_mean - inter_mean),
    }


def _hybrid_recall(vectors: np.ndarray, country: np.ndarray, city: np.ndarray, section: np.ndarray):
    if len(vectors) < 3:
        return {}
    vectors = _normalize_rows(vectors)
    k_vals = [1, 5, 10]
    hits = {k: 0 for k in k_vals}
    totals = {k: 0 for k in k_vals}
    considered = 0
    pool_sizes = []

    for i in range(len(vectors)):
        mask = (country == country[i]) & (city == city[i])
        mask[i] = False
        cand_idx = np.where(mask)[0]
        if len(cand_idx) == 0:
            continue
        rel_mask = (section[cand_idx] == section[i])
        num_rel = int(np.sum(rel_mask))
        if num_rel == 0:
            continue
        sims = np.dot(vectors[cand_idx], vectors[i])
        order = np.argsort(-sims)
        pool_sizes.append(len(cand_idx))
        considered += 1
        for k in k_vals:
            topk = order[:min(k, len(order))]
            found = int(np.sum(rel_mask[topk]))
            hits[k] += found
            totals[k] += num_rel

    if considered == 0:
        return {}

    return {
        'recall@1': float(hits[1] / max(1, totals[1])),
        'recall@5': float(hits[5] / max(1, totals[5])),
        'recall@10': float(hits[10] / max(1, totals[10])),
        'queries_evaluated': int(considered),
        'avg_pool_size': float(np.mean(pool_sizes)) if pool_sizes else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Compute Hybrid Recall and Filter-Separation for text embeddings")
    parser.add_argument("--limit", type=int, default=1000, help="Max rows to fetch from DB")
    parser.add_argument("--out-dir", default="src/tests/metrics/search_quality_metrics", help="Directory to save JSON")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    limit = args.limit  # keep evaluation light and fast
    cols = [
        'id', 'country_code', 'city_code', 'store_name', 'collection_section',
        'text_emb_e5', 'text_emb_gte', 'text_emb_je3',
    ]
    sql = f"SELECT {', '.join(cols)} FROM glovo_ai.products ORDER BY id LIMIT %s"
    with _db_conn() as conn:
        df = pd.read_sql(sql, conn, params=(limit,))

    out = {
        'source': 'supabase_postgres: glovo_ai.products',
        'limit': limit,
        'embeddings': {},
    }

    emb_cols = {
        'text_emb_e5': 'e5',
        'text_emb_gte': 'gte',
        'text_emb_je3': 'je3',
    }

    for col_name, short in emb_cols.items():
        if col_name not in df.columns:
            continue
        idx, mat = _vectors_from_series(df[col_name])
        if len(idx) == 0:
            continue
        country = df.iloc[idx]['country_code'].astype(str).fillna('') .to_numpy()
        city = df.iloc[idx]['city_code'].astype(str).fillna('') .to_numpy()
        section = df.iloc[idx]['collection_section'].astype(str).fillna('') .to_numpy()
        store = df.iloc[idx]['store_name'].astype(str).fillna('') .to_numpy()

        # Hybrid Recall using (country_code, city_code) filters and collection_section relevance
        hybrid = _hybrid_recall(mat, country, city, section)

        # Filter-Separation for collection_section and store_name
        fs_section = _filter_separation(mat, section, seed=args.seed)
        fs_store = _filter_separation(mat, store, seed=args.seed)

        out['embeddings'][col_name] = {
            'hybrid_recall': hybrid,
            'filter_separation': {
                'collection_section': fs_section,
                'store_name': fs_store,
            },
        }

    stamp = datetime.utcnow().strftime('%Y%m%d-%H%M%S')
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"search_quality_metrics_{stamp}_{args.limit}.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print(json.dumps({'saved': out_path}, indent=2))


if __name__ == '__main__':
    main()


