import os
import json
from datetime import datetime

import numpy as np
import pandas as pd
import psycopg2
from dotenv import load_dotenv
import argparse


# Minimal script for multimodal alignment (text→image, image→text)
# Metrics using matching text/image embeddings (text_emb_clip_multi ↔ image_emb_clip):
# - Recall@K (K in {1,5,10})
# - Mean Reciprocal Rank (MRR)
# - Positive vs Negative Pair Separation (cosine and euclidean)


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
    parser = argparse.ArgumentParser(description="Compute multimodal alignment metrics (text↔image)")
    parser.add_argument("--limit", type=int, default=1000, help="Max rows to fetch from DB")
    parser.add_argument("--out-dir", default="src/tests/metrics", help="Directory to save JSON")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    cols = [
        'id',
        'text_emb_clip_multi', 'clip_text_emb',
        'image_emb_clip', 'clip_image_emb',
    ]
    sql = f"SELECT {', '.join(cols)} FROM glovo_ai.products ORDER BY id LIMIT %s"
    with _db_conn() as conn:
        df = pd.read_sql(sql, conn, params=(args.limit,))

    # Choose image column (prefer image_emb_clip, fallback to clip_image_emb)
    img_col = 'image_emb_clip' if 'image_emb_clip' in df.columns else ('clip_image_emb' if 'clip_image_emb' in df.columns else None)
    if img_col is None:
        print(json.dumps({'error': 'No image embedding column found'}, indent=2))
        return

    out = {
        'source': 'supabase_postgres: glovo_ai.products',
        'limit': args.limit,
        'image_col': img_col,
        'embeddings': {},
    }

    # Choose text column (prefer text_emb_clip_multi, fallback to clip_text_emb)
    text_cols = []
    if 'text_emb_clip_multi' in df.columns:
        text_cols.append('text_emb_clip_multi')
    elif 'clip_text_emb' in df.columns:
        text_cols.append('clip_text_emb')
    else:
        print(json.dumps({'error': 'No suitable text embedding column found (need text_emb_clip_multi or clip_text_emb)'}, indent=2))
        return
    k_list = [1, 5, 10]

    # Build image matrix once
    img_idx, img_mat = _vectors_from_series(df[img_col])

    for tcol in text_cols:
        if tcol not in df.columns:
            continue
        t_idx, t_mat = _vectors_from_series(df[tcol])
        if len(t_idx) == 0 or len(img_idx) == 0:
            continue

        # Align by original row indices
        common = np.intersect1d(t_idx, img_idx)
        if len(common) == 0:
            continue
        pos_t = {orig: pos for pos, orig in enumerate(t_idx)}
        pos_i = {orig: pos for pos, orig in enumerate(img_idx)}
        t_aligned = np.stack([t_mat[pos_t[i]] for i in common])
        v_aligned = np.stack([img_mat[pos_i[i]] for i in common])

        # Guard against dimensionality mismatch
        if t_aligned.shape[1] != v_aligned.shape[1]:
            out['embeddings'][tcol] = {
                'pairs': 0,
                'warning': f'dimension_mismatch: text_dim={t_aligned.shape[1]} image_dim={v_aligned.shape[1]}',
            }
            continue

        # Normalize for cosine
        t_norm = _normalize_rows(t_aligned)
        v_norm = _normalize_rows(v_aligned)

        # Positive vs Negative Pair Separation
        n = len(t_aligned)
        pos_cos = np.sum(t_norm * v_norm, axis=1).astype(float)
        pos_euc = np.linalg.norm(t_aligned - v_aligned, axis=1).astype(float)

        # Negative pairs via random permutation (seeded)
        rs = np.random.RandomState(args.seed if args.seed is not None else 42)
        perm = rs.permutation(n)
        # Avoid self-pairs
        fix = perm == np.arange(n)
        if np.any(fix):
            perm[fix] = (np.arange(n)[fix] + 1) % n
        neg_cos = np.sum(t_norm * v_norm[perm], axis=1).astype(float)
        neg_euc = np.linalg.norm(t_aligned - v_aligned[perm], axis=1).astype(float)

        sep = {
            'pos_cos_mean': float(np.mean(pos_cos)),
            'neg_cos_mean': float(np.mean(neg_cos)),
            'cos_separation': float(np.mean(pos_cos) - np.mean(neg_cos)),
            'pos_euc_mean': float(np.mean(pos_euc)),
            'neg_euc_mean': float(np.mean(neg_euc)),
            'euc_separation': float(np.mean(neg_euc) - np.mean(pos_euc)),
            'count': int(n),
        }

        # Retrieval: text→image
        sims_ti = np.dot(t_norm, v_norm.T)
        ranks_ti = np.argsort(-sims_ti, axis=1)
        truth = np.arange(n)
        pos_rank_ti = np.empty(n, dtype=np.int32)
        for i in range(n):
            pos_rank_ti[i] = int(np.where(ranks_ti[i] == truth[i])[0][0]) + 1
        recall_ti = {f'recall@{k}': float(np.mean(pos_rank_ti <= k)) for k in k_list}
        mrr_ti = float(np.mean(1.0 / pos_rank_ti))

        # Retrieval: image→text
        sims_it = sims_ti.T
        ranks_it = np.argsort(-sims_it, axis=1)
        pos_rank_it = np.empty(n, dtype=np.int32)
        for i in range(n):
            pos_rank_it[i] = int(np.where(ranks_it[i] == truth[i])[0][0]) + 1
        recall_it = {f'recall@{k}': float(np.mean(pos_rank_it <= k)) for k in k_list}
        mrr_it = float(np.mean(1.0 / pos_rank_it))

        out['embeddings'][tcol] = {
            'pairs': int(n),
            'pair_separation': sep,
            'text_to_image': {
                **recall_ti,
                'mrr': mrr_ti,
            },
            'image_to_text': {
                **recall_it,
                'mrr': mrr_it,
            },
        }

    stamp = datetime.utcnow().strftime('%Y%m%d-%H%M%S')
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"multimodal_alignment_metrics_{stamp}.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print(json.dumps({'saved': out_path}, indent=2))


if __name__ == '__main__':
    main()


