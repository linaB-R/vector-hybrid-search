import os
import json
from datetime import datetime

import numpy as np
import pandas as pd
import psycopg2
from dotenv import load_dotenv
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.cluster import KMeans
import argparse


# Minimal script to evaluate proxy relevance/ranking quality for text embeddings
# Metrics per embedding (e5, gte, je3) using collection_section as proxy label:
# - 1-NN Accuracy (leave-one-out)
# - Silhouette Score (cosine)
# - Clustering Purity: ARI and NMI (KMeans vs labels)
# - Label Consistency @ K (K in {1,5,10})
# Saves a JSON report to src/tests/metrics.


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


def main():
    parser = argparse.ArgumentParser(description="Compute proxy relevance metrics for text embeddings")
    parser.add_argument("--limit", type=int, default=1000, help="Max rows to fetch from DB")
    parser.add_argument("--out-dir", default="src/tests/metrics", help="Directory to save JSON")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    limit = args.limit
    cols = [
        'id', 'collection_section',
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

    for col_name, _ in emb_cols.items():
        if col_name not in df.columns:
            continue
        idx, mat = _vectors_from_series(df[col_name])
        if len(idx) < 3:
            continue
        labels = df.iloc[idx]['collection_section'].astype(str).fillna('').to_numpy()
        if len(np.unique(labels)) < 2:
            continue

        mat = _normalize_rows(mat)

        # 1-NN leave-one-out accuracy (cosine)
        nn = NearestNeighbors(n_neighbors=2, metric='cosine')
        nn.fit(mat)
        _, nn_idx = nn.kneighbors(mat)
        nn1 = nn_idx[:, 1]
        knn_acc = float(np.mean(labels[nn1] == labels))

        # Silhouette score (cosine)
        sil = float(silhouette_score(mat, labels, metric='cosine'))

        # Clustering purity via ARI/NMI (KMeans with k = unique labels)
        k = len(np.unique(labels))
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=(args.seed if args.seed is not None else 42))
        pred = kmeans.fit_predict(mat)
        ari = float(adjusted_rand_score(labels, pred))
        nmi = float(normalized_mutual_info_score(labels, pred))

        # Label Consistency @ K (exclude self)
        k_max = 10
        nnk = NearestNeighbors(n_neighbors=min(k_max + 1, len(mat)), metric='cosine')
        nnk.fit(mat)
        _, idxs = nnk.kneighbors(mat)
        idxs = [row[row != i] for i, row in enumerate(idxs)]
        ks = [1, 5, 10]
        consistency = {}
        for k in ks:
            vals = []
            for i, row in enumerate(idxs):
                top = row[:min(k, len(row))]
                if len(top) == 0:
                    continue
                same = np.mean(labels[top] == labels[i])
                vals.append(float(same))
            if vals:
                consistency[f'label_consistency@{k}'] = float(np.mean(vals))

        out['embeddings'][col_name] = {
            'knn_1_acc': knn_acc,
            'silhouette_cosine': sil,
            'ari': ari,
            'nmi': nmi,
            'label_consistency': consistency,
            'samples': int(len(mat)),
            'unique_labels': int(k),
        }

    stamp = datetime.utcnow().strftime('%Y%m%d-%H%M%S')
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"relevance_quality_metrics_{stamp}.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print(json.dumps({'saved': out_path}, indent=2))


if __name__ == '__main__':
    main()


