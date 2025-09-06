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
    parser.add_argument("--out-dir", default="src/tests/metrics/relevance_quality_metrics", help="Directory to save JSON")
    parser.add_argument("--proxy-column", choices=["collection_section", "store_name", "product_name"], default="collection_section", help="Column to use as proxy labels")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    limit = args.limit
    cols = [
        'id', 'country_code', 'collection_section', 'store_name', 'product_name',
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
        labels_series = df.iloc[idx][args.proxy_column].astype(str).fillna('')
        # Base normalization for all proxies: lowercase, remove punctuation, collapse spaces
        labels_series = labels_series.str.lower().str.replace(r'[^\w\s]+', '', regex=True).str.replace(r'\s+', ' ', regex=True).str.strip()
        # Stronger normalization for product names: drop numeric/size tokens and keep first 3 tokens
        if args.proxy_column == 'product_name':
            labels_series = labels_series.str.split().map(
                lambda toks: ' '.join(
                    [t for t in toks if not t.isdigit() and t not in {'ml', 'g', 'kg', 'l', 'pack', 'pcs'}][:3]
                )
            )
        labels = labels_series.to_numpy()
        if len(np.unique(labels)) < 2:
            continue

        mat = _normalize_rows(mat)

        # Overall metrics (all countries)
        nn = NearestNeighbors(n_neighbors=2, metric='cosine')
        nn.fit(mat)
        _, nn_idx = nn.kneighbors(mat)
        nn1 = nn_idx[:, 1]
        knn_acc = float(np.mean(labels[nn1] == labels))

        uniq = len(np.unique(labels))
        if 2 <= uniq <= len(mat) - 1:
            sil = float(silhouette_score(mat, labels, metric='cosine'))
        else:
            sil = None

        k = len(np.unique(labels))
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=(args.seed if args.seed is not None else 42))
        pred = kmeans.fit_predict(mat)
        ari = float(adjusted_rand_score(labels, pred))
        nmi = float(normalized_mutual_info_score(labels, pred))

        k_max = 10
        nnk = NearestNeighbors(n_neighbors=min(k_max + 1, len(mat)), metric='cosine')
        nnk.fit(mat)
        _, idxs = nnk.kneighbors(mat)
        idxs = [row[row != i] for i, row in enumerate(idxs)]
        ks = [1, 5, 10]
        consistency = {}
        for k_val in ks:
            vals = []
            for i, row in enumerate(idxs):
                top = row[:min(k_val, len(row))]
                if len(top) == 0:
                    continue
                same = np.mean(labels[top] == labels[i])
                vals.append(float(same))
            if vals:
                consistency[f'label_consistency@{k_val}'] = float(np.mean(vals))

        # By-country breakdown only when proxy is store_name
        by_country = None
        if args.proxy_column == 'store_name':
            country = df.iloc[idx]['country_code'].astype(str).fillna('').to_numpy()
            by_country = {}
            for c in np.unique(country):
                mask = country == c
                if int(np.sum(mask)) < 3:
                    continue
                labels_c = labels[mask]
                uniq_c = len(np.unique(labels_c))
                if uniq_c < 2:
                    continue
                mat_c = mat[mask]

                nn_c = NearestNeighbors(n_neighbors=2, metric='cosine')
                nn_c.fit(mat_c)
                _, idx_c = nn_c.kneighbors(mat_c)
                nn1_c = idx_c[:, 1]
                knn_acc_c = float(np.mean(labels_c[nn1_c] == labels_c))

                if 2 <= uniq_c <= len(mat_c) - 1:
                    sil_c = float(silhouette_score(mat_c, labels_c, metric='cosine'))
                else:
                    sil_c = None

                k_c = uniq_c
                kmeans_c = KMeans(n_clusters=k_c, n_init=10, random_state=(args.seed if args.seed is not None else 42))
                pred_c = kmeans_c.fit_predict(mat_c)
                ari_c = float(adjusted_rand_score(labels_c, pred_c))
                nmi_c = float(normalized_mutual_info_score(labels_c, pred_c))

                nnk_c = NearestNeighbors(n_neighbors=min(k_max + 1, len(mat_c)), metric='cosine')
                nnk_c.fit(mat_c)
                _, idxs_c = nnk_c.kneighbors(mat_c)
                idxs_c = [row[row != i] for i, row in enumerate(idxs_c)]
                consistency_c = {}
                for k_val in ks:
                    vals = []
                    for i, row in enumerate(idxs_c):
                        top = row[:min(k_val, len(row))]
                        if len(top) == 0:
                            continue
                        same = np.mean(labels_c[top] == labels_c[i])
                        vals.append(float(same))
                    if vals:
                        consistency_c[f'label_consistency@{k_val}'] = float(np.mean(vals))

                by_country[c] = {
                    'knn_1_acc': knn_acc_c,
                    'silhouette_cosine': sil_c,
                    'ari': ari_c,
                    'nmi': nmi_c,
                    'label_consistency': consistency_c,
                    'samples': int(len(mat_c)),
                    'unique_labels': int(k_c),
                }

        entry = {
            'proxy': args.proxy_column,
            'overall': {
                'knn_1_acc': knn_acc,
                'silhouette_cosine': sil,
                'ari': ari,
                'nmi': nmi,
                'label_consistency': consistency,
                'samples': int(len(mat)),
                'unique_labels': int(k),
            }
        }
        if by_country is not None and len(by_country) > 0:
            entry['by_country'] = by_country
        out['embeddings'][col_name] = entry

    stamp = datetime.utcnow().strftime('%Y%m%d-%H%M%S')
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"relevance_quality_metrics_{args.proxy_column}_{stamp}_{args.limit}.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print(json.dumps({'saved': out_path}, indent=2))


if __name__ == '__main__':
    main()


