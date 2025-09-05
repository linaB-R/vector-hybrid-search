import os
import json
import time
from datetime import datetime
from typing import List, Dict, Optional, Tuple

import numpy as np
import psycopg2
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import umap
from dotenv import load_dotenv


# NOTE: This script connects directly to PostgreSQL (Supabase) using psycopg2, pulls a sample of
# rows from glovo_ai.products, computes selected metrics on available embedding columns, and saves
# results (JSON + optional plots) into src/tests/metrics. The code keeps a simple sequential flow
# with small helper functions and inline comments explaining each step.


def _db_conn():
    """Create a PostgreSQL connection using env vars: user, password, host, port, dbname."""
    # Load variables from a .env file located in the project root, if present.
    load_dotenv()
    return psycopg2.connect(
        user=os.getenv("user"),
        password=os.getenv("password"),
        host=os.getenv("host"),
        port=os.getenv("port"),
        dbname=os.getenv("dbname"),
    )


def _parse_vec(cell) -> Optional[np.ndarray]:
    """Parse pgvector text representation (e.g., "[0.1, -0.2, ...]") into a numpy array.
    Returns None if empty or malformed.
    """
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
    """L2-normalize rows for cosine computations."""
    if mat.size == 0:
        return mat
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8
    return mat / norms


def _select_text_col(cols: List[str]) -> Optional[str]:
    """Choose the first available text embedding column in preferred order.
    Prefer matched CLIP text first for cross‑modal pairing with CLIP image.
    """
    preferred = [
        'clip_text_emb',        # CLIP text (paired with image_emb_clip)
        'text_emb_clip_multi',  # CLIP multilingual (not jointly trained with ViT-B/32 image)
        'text_emb_gte',         # GTE base
        'text_emb_e5',          # E5 small
        'text_emb_je3',         # JE-3
    ]
    for c in preferred:
        if c in cols:
            return c
    return None


def _select_image_col(cols: List[str]) -> Optional[str]:
    """Choose the first available image embedding column in preferred order."""
    preferred = [
        'image_emb_clip',  # CLIP ViT-B/32 image
        'clip_image_emb',  # legacy name
    ]
    for c in preferred:
        if c in cols:
            return c
    return None


def _fetch_sample(limit: int, require_text: Optional[str] = None, require_image: Optional[str] = None) -> pd.DataFrame:
    """Pull a sample from glovo_ai.products.
    If require_text/image provided, enforce non-null for those columns to ensure pairs.
    """
    cols = [
        'id',
        'collection_section',
        'text_emb_je3', 'clip_text_emb', 'clip_image_emb',
        'text_emb_e5', 'text_emb_gte', 'image_emb_clip', 'text_emb_clip_multi',
    ]
    base = f"SELECT {', '.join(cols)} FROM glovo_ai.products"
    where = []
    if require_text:
        where.append(f"{require_text} IS NOT NULL")
    if require_image:
        where.append(f"{require_image} IS NOT NULL")
    if where:
        base += " WHERE " + " AND ".join(where)
    sql = base + " ORDER BY id LIMIT %s"
    with _db_conn() as conn:
        df = pd.read_sql(sql, conn, params=(limit,))
    return df


def _vectors_from_series(series: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    """Parse a pandas Series of vector strings into matrix and keep original indices for alignment."""
    parsed = series.map(_parse_vec).tolist()
    idx = np.array([i for i, v in enumerate(parsed) if v is not None], dtype=np.int64)
    if len(idx) == 0:
        return idx, np.zeros((0, 0), dtype=np.float32)
    dim = len(parsed[idx[0]])
    mat = np.zeros((len(idx), dim), dtype=np.float32)
    for j, i in enumerate(idx):
        mat[j] = parsed[i]
    return idx, mat


def cosine_and_euclidean_pair_stats(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    """Compute cosine and Euclidean stats on aligned pairs, plus negatives by permutation."""
    n = min(len(a), len(b))
    if n == 0:
        return {}
    a_n = _normalize_rows(a[:n])
    b_n = _normalize_rows(b[:n])

    cos_pos = np.sum(a_n * b_n, axis=1)
    euc_pos = np.linalg.norm(a[:n] - b[:n], axis=1)

    neg_stats = {}
    if n > 1:
        perm = np.roll(np.arange(n), 1)
        cos_neg = np.sum(a_n * b_n[perm], axis=1)
        euc_neg = np.linalg.norm(a[:n] - b[:n][perm], axis=1)
        neg_stats = {
            'cos_neg_mean': float(np.mean(cos_neg)),
            'cos_neg_p95': float(np.percentile(cos_neg, 95)),
            'euc_neg_mean': float(np.mean(euc_neg)),
            'euc_neg_p05': float(np.percentile(euc_neg, 5)),
        }

    return {
        'count': int(n),
        'cos_pos_mean': float(np.mean(cos_pos)),
        'cos_pos_median': float(np.median(cos_pos)),
        'cos_pos_p95': float(np.percentile(cos_pos, 95)),
        'euc_pos_mean': float(np.mean(euc_pos)),
        'euc_pos_median': float(np.median(euc_pos)),
        'euc_pos_p05': float(np.percentile(euc_pos, 5)),
        **neg_stats,
    }


def retrieval_metrics(queries: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """Single-relevant-per-query retrieval: Recall@K, nDCG@K, MRR using cosine similarity."""
    if len(queries) == 0 or len(targets) == 0:
        return {}
    q = _normalize_rows(queries)
    t = _normalize_rows(targets)
    sims = np.dot(q, t.T)
    ranks = np.argsort(-sims, axis=1)
    truth = np.arange(len(q))
    pos = np.empty(len(q), dtype=np.int32)
    for i in range(len(q)):
        pos[i] = int(np.where(ranks[i] == truth[i])[0][0]) + 1
    out = {}
    for k in [1, 5, 10]:
        rec = float(np.mean(pos <= k))
        ndcg = float(np.mean(np.where(pos <= k, 1.0 / np.log2(1.0 + pos), 0.0)))
        out[f'recall@{k}'] = rec
        out[f'ndcg@{k}'] = ndcg
    out['mrr'] = float(np.mean(1.0 / pos))
    return out


def knn_and_silhouette(vectors: np.ndarray, labels: List[str]) -> Dict[str, float]:
    """1-NN leave-one-out accuracy (cosine) and silhouette score (cosine) if >1 label."""
    if len(vectors) < 3:
        return {}
    labels_arr = np.array(labels)
    nn = NearestNeighbors(n_neighbors=2, metric='cosine')
    nn.fit(vectors)
    d, idx = nn.kneighbors(vectors)
    nn_idx = idx[:, 1]
    acc = float(np.mean(labels_arr[nn_idx] == labels_arr))
    out = {'knn_1_acc': acc}
    if len(np.unique(labels_arr)) > 1:
        sample = vectors
        labs = labels_arr
        if len(vectors) > 2000:
            rs = np.random.RandomState(42)
            sel = rs.choice(len(vectors), size=2000, replace=False)
            sample = vectors[sel]
            labs = labels_arr[sel]
        out['silhouette_cosine'] = float(silhouette_score(sample, labs, metric='cosine'))
    return out


def tsne_umap_plots(vectors: np.ndarray, labels: List[str], out_dir: str, prefix: str) -> Dict[str, str]:
    """Generate t-SNE and UMAP 2D scatter plots colored by labels; return file paths."""
    os.makedirs(out_dir, exist_ok=True)
    res = {}
    # t-SNE
    try:
        tsne2 = TSNE(n_components=2, init='pca', random_state=42, perplexity=min(30, max(5, len(vectors)//50)))
        xy = tsne2.fit_transform(vectors)
        fig = plt.figure(figsize=(6, 5))
        plt.scatter(xy[:, 0], xy[:, 1], c=pd.factorize(labels)[0], s=5, cmap='tab10', alpha=0.7)
        plt.title('t-SNE')
        path = os.path.join(out_dir, f"{prefix}_tsne.png")
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close(fig)
        res['tsne_png'] = path
    except Exception:
        pass
    # UMAP
    try:
        reducer = umap.UMAP(n_components=2, random_state=42)
        xy = reducer.fit_transform(vectors)
        fig = plt.figure(figsize=(6, 5))
        plt.scatter(xy[:, 0], xy[:, 1], c=pd.factorize(labels)[0], s=5, cmap='tab10', alpha=0.7)
        plt.title('UMAP')
        path = os.path.join(out_dir, f"{prefix}_umap.png")
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close(fig)
        res['umap_png'] = path
    except Exception:
        pass
    return res


def ari_nmi_from_kmeans(vectors: np.ndarray, labels: List[str]) -> Dict[str, float]:
    """Cluster with KMeans (k = unique labels) and compute ARI/NMI vs ground truth."""
    labs = np.array(labels)
    uniq = np.unique(labs)
    if len(uniq) < 2 or len(vectors) < len(uniq):
        return {}
    kmeans = KMeans(n_clusters=len(uniq), n_init=10, random_state=42)
    pred = kmeans.fit_predict(vectors)
    return {
        'ari': float(adjusted_rand_score(labs, pred)),
        'nmi': float(normalized_mutual_info_score(labs, pred)),
    }


def main():
    import argparse

    # CLI: choose sample size and which metrics to run.
    parser = argparse.ArgumentParser(description="Compute embedding metrics from Supabase (Postgres)")
    parser.add_argument("--limit", type=int, default=1000, help="Max rows to fetch from DB")
    parser.add_argument(
        "--metrics",
        default="all",
        help="Comma-separated: cosine,euclidean,retrieval,knn,silhouette,tsne,umap,ari,nmi or 'all'",
    )
    parser.add_argument(
        "--out-dir", default="src/tests/metrics", help="Directory to save JSON and plots"
    )
    parser.add_argument("--text-col", help="Override auto text column selection")
    parser.add_argument("--image-col", help="Override auto image column selection")
    parser.add_argument("--text-col-queries", help="Text column for queries (text→text retrieval)")
    parser.add_argument("--text-col-targets", help="Text column for targets (text→text retrieval)")
    parser.add_argument("--text-only", action='store_true', help="Disable any image-based metrics and compute text-only metrics")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    start = time.time()
    if args.seed is not None:
        np.random.seed(args.seed)

    # Determine metric set
    selected = set([m.strip().lower() for m in args.metrics.split(',')])
    if 'all' in selected:
        selected = {'cosine', 'euclidean', 'retrieval', 'knn', 'silhouette', 'tsne', 'umap', 'ari', 'nmi'}

    # Initial auto selection (may be overridden below)
    probe_df = _fetch_sample(min(args.limit, 100))
    cols = probe_df.columns.tolist()
    text_col = args.text_col if args.text_col else _select_text_col(cols)
    image_col = None if args.text_only else (args.image_col if args.image_col else _select_image_col(cols))

    # Build WHERE filters depending on selected metrics
    need_text = ('knn' in selected) or ('silhouette' in selected) or ('tsne' in selected) or ('umap' in selected) or ('ari' in selected) or ('nmi' in selected)
    need_pairs = ('retrieval' in selected) or ('cosine' in selected) or ('euclidean' in selected)
    # For text-only retrieval, we want two text columns
    t2t_queries = args.text_col_queries
    t2t_targets = args.text_col_targets
    text_pairs_mode = args.text_only and t2t_queries and t2t_targets

    req_text = (t2t_queries if text_pairs_mode else text_col) if (need_text or need_pairs) else None
    req_image = None if args.text_only else (image_col if need_pairs else None)

    # Fetch with requirements; if empty and CLIP text was requested but NULL in DB, fallback to multilingual CLIP text
    df = _fetch_sample(args.limit, require_text=req_text, require_image=req_image)
    if df.empty and (text_pairs_mode is False) and text_col == 'clip_text_emb':
        text_col = 'text_emb_clip_multi'
        req_text = text_col if (need_text or need_pairs) else None
        df = _fetch_sample(args.limit, require_text=req_text, require_image=req_image)

    # Parse chosen text embedding and labels for label-based metrics.
    label_col = 'collection_section' if 'collection_section' in cols else None
    label_metrics = {}
    vis_paths = {}
    cluster_metrics = {}

    # Build text vectors and labels if possible.
    text_idx, text_mat = _vectors_from_series(df[text_col]) if text_col else (np.array([], dtype=np.int64), np.zeros((0, 0), dtype=np.float32))
    labels = df.iloc[text_idx][label_col].astype(str).tolist() if label_col and len(text_idx) > 0 else []
    text_mat = _normalize_rows(text_mat)

    # Prepare cross-modal aligned pairs for pair/retrieval metrics.
    pair_stats = {}
    retrieval = {'text_to_text': {}, 'text_to_image': {}, 'image_to_text': {}}

    if text_pairs_mode:
        # Text→Text aligned pairs using two different text columns
        q_idx, q_mat_raw = _vectors_from_series(df[t2t_queries])
        t_idx, t_mat_raw = _vectors_from_series(df[t2t_targets])
        common = np.intersect1d(q_idx, t_idx)
        if len(common) > 0:
            pos_q = {orig: pos for pos, orig in enumerate(q_idx)}
            pos_targ = {orig: pos for pos, orig in enumerate(t_idx)}
            q_aligned = np.stack([q_mat_raw[pos_q[i]] for i in common])
            t_aligned_txt = np.stack([t_mat_raw[pos_targ[i]] for i in common])
            # Pair stats (cosine + euclidean) between text columns
            if ('cosine' in selected) or ('euclidean' in selected):
                pair_stats = cosine_and_euclidean_pair_stats(q_aligned, t_aligned_txt)
            # Retrieval text→text
            if 'retrieval' in selected:
                retrieval['text_to_text'] = retrieval_metrics(q_aligned, t_aligned_txt)
        else:
            q_aligned = np.zeros((0, 0), dtype=np.float32)
            t_aligned_txt = np.zeros((0, 0), dtype=np.float32)
    elif (not args.text_only) and text_col and image_col:
        # Cross‑modal path retained for backwards compatibility
        img_idx, img_mat = _vectors_from_series(df[image_col])
        common = np.intersect1d(text_idx, img_idx)
        if len(common) > 0:
            pos_t = {orig: pos for pos, orig in enumerate(text_idx)}
            pos_i = {orig: pos for pos, orig in enumerate(img_idx)}
            t_aligned = np.stack([text_mat[pos_t[i]] for i in common])
            v_aligned = np.stack([img_mat[pos_i[i]] for i in common])
        else:
            t_aligned = np.zeros((0, 0), dtype=np.float32)
            v_aligned = np.zeros((0, 0), dtype=np.float32)

    # Metric selection already computed above.

    # Cosine + Euclidean pair stats on aligned text↔image.
    if ('cosine' in selected) or ('euclidean' in selected):
        # Already computed in text_to_text branch; for cross-modal compute here
        if (not args.text_only) and 'text_to_text' not in retrieval:
            if 't_aligned' in locals() and 'v_aligned' in locals() and len(t_aligned) > 0 and len(v_aligned) > 0:
                pair_stats = cosine_and_euclidean_pair_stats(t_aligned, v_aligned)

    # Retrieval metrics (text→image and image→text) using aligned pairs.
    if 'retrieval' in selected and (not args.text_only):
        if 't_aligned' in locals() and 'v_aligned' in locals() and len(t_aligned) > 0 and len(v_aligned) > 0:
            retrieval['text_to_image'] = retrieval_metrics(t_aligned, v_aligned)
            retrieval['image_to_text'] = retrieval_metrics(v_aligned, t_aligned)

    # KNN and silhouette using text embeddings and labels.
    if ('knn' in selected) or ('silhouette' in selected):
        if len(text_mat) > 0 and len(labels) > 0:
            label_metrics = knn_and_silhouette(text_mat, labels)

    # t-SNE and UMAP plots from text embeddings colored by labels.
    if ('tsne' in selected) or ('umap' in selected):
        if len(text_mat) > 0 and len(labels) > 0:
            prefix = f"viz_{text_col or 'text'}_{args.limit}"
            vis_paths = tsne_umap_plots(text_mat, labels, args.out_dir, prefix)

    # ARI and NMI vs ground truth using KMeans over text embeddings.
    if ('ari' in selected) or ('nmi' in selected):
        if len(text_mat) > 0 and len(labels) > 0:
            cluster_metrics = ari_nmi_from_kmeans(text_mat, labels)

    # Build JSON with metrics and short interpretations for thesis-ready reporting.
    # Keep it concise and focused on explaining what higher/lower values mean.
    explain = {
        'cosine': 'Cosine similarity close to 1 means stronger text↔image alignment; negatives should be lower.',
        'euclidean': 'Smaller Euclidean distance means more similar vectors; positives should be lower than negatives.',
        'retrieval': 'Recall@K is fraction of queries retrieving the true pair in top-K; nDCG@K rewards higher ranks; MRR is average inverse rank.',
        'knn': '1-NN accuracy shows if nearest neighbor shares the same collection_section.',
        'silhouette': 'Silhouette (cosine) near 1 indicates well-separated clusters by collection_section; near 0 overlaps; negative indicates mis-clustering.',
        'tsne_umap': '2D projections visualize cluster structure; tighter, separated groups suggest coherent embeddings.',
        'ari_nmi': 'ARI and NMI compare KMeans clusters to collection_section; higher is better (perfect = 1).',
    }

    summary = {
        'source': 'supabase_postgres: glovo_ai.products',
        'limit': args.limit,
        'text_col': text_col,
        'image_col': image_col,
        'pair_stats': pair_stats,
        'retrieval': retrieval,
        'label_metrics': label_metrics,
        'cluster_metrics': cluster_metrics,
        'visualizations': vis_paths,
        'explanations': explain,
        'seconds': round(time.time() - start, 2),
    }

    stamp = datetime.utcnow().strftime('%Y%m%d-%H%M%S')
    out_json = os.path.join(args.out_dir, f"metrics_{stamp}_{args.limit}.json")
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    # Also print path to results so it is easy to find.
    print(json.dumps({'saved': out_json}, indent=2))


if __name__ == '__main__':
    main()


