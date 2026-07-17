# perceptual_rdm.py
"""
Compute a perceptual RDM across identities using CLIP ViT-B/32 image embeddings.

Pipeline:
  1. Query photo_metadata.monkeysCoord to get monkey_id -> identity_name mapping
  2. For each identity, find all images in /OneMonkey/ matching that monkey_id
  3. Run each image through CLIP ViT-B/32 → 512-D embedding
  4. Average embeddings within identity → one vector per identity
  5. Compute pairwise cosine distance → perceptual RDM
  6. Save pickle + visualization

The script is self-diagnosing: it prints DB schema info, image counts, and
sanity checks before producing the final RDM. Run once with DRY_RUN=True to
verify everything before doing the full embedding pass.

Output (in OUTPUT_DIR):
  perceptual_rdm.pkl  — dict with keys: identities, embeddings, rdm, image_paths
  perceptual_rdm.png  — heatmap of the RDM
  diagnostics.txt     — per-identity image counts and any issues found
"""
import os
import sys
import pickle
import warnings
from pathlib import Path

import mysql.connector
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from scipy.spatial.distance import pdist, squareform

import torch
from transformers import CLIPModel, CLIPProcessor


# ═══════════════════════════════════════════════════════════════════════
# CONFIG — edit and run
# ═══════════════════════════════════════════════════════════════════════
STIM_DIR         = "/home/connorlab/Stimuli/OneMonkey"
DB_HOST          = "localhost"
DB_USER          = "xper_rw"
DB_PASS          = "up2nite"
METADATA_SCHEMA  = "photo_metadata"
METADATA_TABLE   = "monkeys"          # columns: monkey_id, jpg_id, monkey_name
MONKEY_INFO_CSV  = "/home/connorlab/Documents/GitHub/Julie/social_data/monkeyinfo.csv"

CLIP_MODEL_NAME  = "openai/clip-vit-base-patch32"
OUTPUT_DIR       = Path("./perceptual_rdm_output")

# If True, only print what would be done (no CLIP inference). Use this once
# to verify the DB lookup and image discovery before running the real pass.
DRY_RUN          = False

# Restrict to identities in monkeyinfo.csv (i.e., the ~33 used in RSA/state-space).
# If False, embeds all monkeys with photos in /OneMonkey/.
RESTRICT_TO_MONKEYINFO = True

# Identities to exclude entirely (e.g., too few images, or not in the RSA set).
# 46J has only 1 image on disk; 40J dropped to match the RSA/state-space set.
EXCLUDE_IDENTITIES = {'46J', '40J'}

# Warn if an identity has fewer than this many images (noisy averaged embedding).
MIN_IMAGES_WARN = 20
# ═══════════════════════════════════════════════════════════════════════


def connect_db():
    return mysql.connector.connect(host=DB_HOST, user=DB_USER, passwd=DB_PASS)


def get_monkey_lookup(conn, restrict_names=None):
    """
    Returns DataFrame with columns: monkey_id, name.

    photo_metadata.monkeys has one row per cropped image:
      monkey_id   = the cropped image filename stem ({monkey_id}.jpg in /OneMonkey/)
      jpg_id      = the ORIGINAL uncropped photo id (ignored here)
      monkey_name = the actual identity (e.g., 110E, 151J, 7124)

    One identity (monkey_name) maps to MANY monkey_ids — one per body-bbox crop
    it appears in. That is exactly the "multiple images per identity" we average.
    """
    cur = conn.cursor()
    cur.execute(
        f"SELECT monkey_id, monkey_name "
        f"FROM `{METADATA_SCHEMA}`.`{METADATA_TABLE}`"
    )
    rows = cur.fetchall()
    cur.close()

    records = [{'monkey_id': int(mid), 'name': str(nm).strip()}
               for mid, nm in rows if nm is not None]
    df = pd.DataFrame(records)
    print(f"\n  Found {len(df)} crop rows in {METADATA_TABLE} "
          f"({df['name'].nunique()} unique identities)")

    if restrict_names is not None:
        before_rows = len(df)
        before_ids = df['name'].nunique()
        df = df[df['name'].isin(restrict_names)].reset_index(drop=True)
        print(f"  Restricted to monkeyinfo.csv identities: "
              f"{df['name'].nunique()}/{before_ids} identities, "
              f"{len(df)}/{before_rows} crop rows")

    if EXCLUDE_IDENTITIES:
        before_ids = df['name'].nunique()
        df = df[~df['name'].isin(EXCLUDE_IDENTITIES)].reset_index(drop=True)
        print(f"  Excluded {EXCLUDE_IDENTITIES}: "
              f"{df['name'].nunique()}/{before_ids} identities remain")

    return df


def find_image_for_crop(monkey_id, stim_dir):
    """Each monkey_id corresponds to exactly one cropped file {monkey_id}.jpg."""
    stim_path = Path(stim_dir)
    # Try exact name with common extensions/cases
    for ext in ('.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG'):
        cand = stim_path / f"{monkey_id}{ext}"
        if cand.exists():
            return cand
    return None


def discover_images(lookup_df, stim_dir):
    """
    lookup_df has one row per crop (monkey_id, name). Collapse to one row per
    identity with the list of cropped-image paths.
    """
    lookup_df = lookup_df.copy()
    lookup_df['image_path'] = lookup_df['monkey_id'].apply(
        lambda mid: find_image_for_crop(mid, stim_dir))

    # Group by identity, collect all found image paths
    grouped = []
    for name, sub in lookup_df.groupby('name'):
        paths = [p for p in sub['image_path'].tolist() if p is not None]
        missing_ids = sub.loc[sub['image_path'].isna(), 'monkey_id'].tolist()
        grouped.append({
            'name': name,
            'monkey_ids': sub['monkey_id'].tolist(),
            'image_paths': paths,
            'n_images': len(paths),
            'n_missing': len(missing_ids),
            'missing_ids': missing_ids,
        })
    out = pd.DataFrame(grouped)
    return out


def print_image_discovery_summary(lookup_df):
    print(f"\n{'─' * 60}")
    print(f"  IMAGE DISCOVERY SUMMARY")
    print(f"{'─' * 60}")
    print(f"  Identities checked: {len(lookup_df)}")
    print(f"  Identities with >=1 image: {(lookup_df['n_images'] > 0).sum()}")
    print(f"  Identities with 0 images: {(lookup_df['n_images'] == 0).sum()}")
    print(f"  Total images found: {lookup_df['n_images'].sum()}")
    total_missing = lookup_df['n_missing'].sum()
    if total_missing > 0:
        print(f"  ⚠ Total crops with missing image files: {total_missing}")
    print(f"  Images per identity:")
    if lookup_df['n_images'].sum() > 0:
        print(f"    min={lookup_df['n_images'].min()}, "
              f"median={int(lookup_df['n_images'].median())}, "
              f"max={lookup_df['n_images'].max()}")

    missing = lookup_df[lookup_df['n_images'] == 0]
    if len(missing) > 0:
        print(f"\n  ⚠ {len(missing)} identities have NO images found:")
        for _, r in missing.iterrows():
            print(f"    name={r['name']}  (monkey_ids={r['monkey_ids']})")

    low = lookup_df[(lookup_df['n_images'] > 0) &
                    (lookup_df['n_images'] < MIN_IMAGES_WARN)]
    if len(low) > 0:
        print(f"\n  ⚠ {len(low)} identities have < {MIN_IMAGES_WARN} images "
              f"(noisy averaged embedding — interpret with caution):")
        for _, r in low.sort_values('n_images').iterrows():
            print(f"    {r['name']:<8s}: {r['n_images']} images")

    print(f"\n  Per-identity image counts (sorted by name):")
    sub = lookup_df.sort_values('name')
    for _, r in sub.iterrows():
        flag = '  ⚠' if r['n_missing'] > 0 else ''
        print(f"    {r['name']:<8s}: {r['n_images']:>2d} images "
              f"({len(r['monkey_ids'])} crops in DB){flag}")


def load_clip(model_name=CLIP_MODEL_NAME, device='cuda'):
    print(f"\n  Loading CLIP: {model_name}")
    # use_safetensors=True avoids the torch.load security gate (CVE-2025-32434)
    # that newer transformers enforce against the older .bin checkpoint format.
    try:
        model = CLIPModel.from_pretrained(
            model_name, use_safetensors=True).to(device).eval()
    except Exception as e:
        print(f"  safetensors load failed ({e}); retrying with default...")
        model = CLIPModel.from_pretrained(model_name).to(device).eval()
    processor = CLIPProcessor.from_pretrained(model_name)
    print(f"  Device: {device}")
    return model, processor


def _extract_image_embeds(feats, model, inputs):
    """
    Return the (batch, embed_dim) projected CLIP image embedding as a tensor,
    robust to transformers-version differences in get_image_features output.

    The projected CLIP image embedding has dimension model.config.projection_dim
    (512 for ViT-B/32). The pre-projection vision pooled output has dimension
    model.config.vision_config.hidden_size (768 for ViT-B/32). We use the
    projected dimension to decide whether a tensor still needs projecting.
    """
    proj_dim = model.config.projection_dim          # 512 for ViT-B/32

    def _maybe_project(t):
        # If already at projection_dim, it's the final embedding — return as is.
        # If at the vision hidden size, apply the visual projection.
        if t.shape[-1] == proj_dim:
            return t
        return model.visual_projection(t)

    # Case 1: already a tensor
    if isinstance(feats, torch.Tensor):
        return _maybe_project(feats)

    # Case 2: ModelOutput exposing image_embeds (already projected)
    v = getattr(feats, 'image_embeds', None)
    if isinstance(v, torch.Tensor):
        return v

    # Case 3: ModelOutput exposing pooler_output. On some versions this is the
    # final projected embedding (dim 512); on others it's the pre-projection
    # pooled vision output (dim 768). _maybe_project handles both.
    pooled = getattr(feats, 'pooler_output', None)
    if isinstance(pooled, torch.Tensor):
        return _maybe_project(pooled)

    # Case 4: last resort — recompute from the vision tower explicitly
    vision_out = model.vision_model(pixel_values=inputs['pixel_values'])
    return _maybe_project(vision_out.pooler_output)


@torch.no_grad()
def embed_images(image_paths, model, processor, device='cuda', batch_size=64):
    """Run a list of image paths through CLIP in batches. Returns (feats, valid_paths)."""
    if len(image_paths) == 0:
        return None, []

    all_feats = []
    valid_paths = []
    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start:start + batch_size]
        images = []
        batch_valid = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert('RGB')
                images.append(img)
                batch_valid.append(p)
            except Exception as e:
                print(f"    ⚠ could not load {p.name}: {e}")
        if len(images) == 0:
            continue

        inputs = processor(images=images, return_tensors='pt').to(device)
        feats = model.get_image_features(**inputs)
        feats = _extract_image_embeds(feats, model, inputs)
        # L2-normalize each image embedding
        feats = feats / feats.norm(dim=-1, keepdim=True)
        all_feats.append(feats.cpu().numpy())
        valid_paths.extend(batch_valid)

    if len(all_feats) == 0:
        return None, []
    return np.concatenate(all_feats, axis=0), valid_paths


def compute_perceptual_rdm(lookup_df, model, processor, device='cuda'):
    """
    Run CLIP on each identity's images, average embeddings per identity,
    compute pairwise cosine distance → perceptual RDM.
    Returns: identities (list), embeddings (n_ids × dim), rdm (n_ids × n_ids),
             used_paths (dict name -> list of Paths).
    """
    sub = lookup_df[lookup_df['n_images'] > 0].sort_values('name').reset_index(drop=True)
    identities = sub['name'].tolist()
    n = len(identities)
    used_paths = {}

    print(f"\n  Embedding {n} identities ({sub['n_images'].sum()} images)...")

    # Collect one averaged vector per identity. We don't hardcode the embedding
    # dim — it's taken from whatever CLIP returns. Identities with no valid
    # images get a placeholder filled in after we know the dim.
    per_identity_vecs = [None] * n
    embed_dim = None
    for i, row in sub.iterrows():
        feats, valid = embed_images(row['image_paths'], model, processor, device)
        if feats is None or len(feats) == 0:
            warnings.warn(f"No valid images for {row['name']}")
            used_paths[row['name']] = []
            continue
        if embed_dim is None:
            embed_dim = feats.shape[1]
        vec = feats.mean(axis=0)
        # Re-normalize the averaged vector so cosine distance is bounded [0, 2]
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        per_identity_vecs[i] = vec.astype(np.float32)
        used_paths[row['name']] = valid
        print(f"    [{i+1:>2d}/{n}] {row['name']:<8s}: {len(valid):>2d} images embedded")

    if embed_dim is None:
        raise RuntimeError("No identity produced a valid embedding — check image paths.")

    # Assemble matrix; any identity that had no images becomes a NaN row
    embeddings = np.full((n, embed_dim), np.nan, dtype=np.float32)
    for i, v in enumerate(per_identity_vecs):
        if v is not None:
            embeddings[i] = v

    # Warn if any NaN rows remain (they would propagate into the RDM)
    nan_rows = np.where(np.isnan(embeddings).any(axis=1))[0]
    if len(nan_rows) > 0:
        bad = [identities[i] for i in nan_rows]
        warnings.warn(f"{len(bad)} identities have no embedding (NaN rows): {bad}. "
                      f"Their RDM entries will be NaN.")

    # Pairwise cosine distance — for L2-normalized vectors, cosine_dist = 1 - dot
    rdm = squareform(pdist(embeddings, metric='cosine'))
    return identities, embeddings, rdm, used_paths


def plot_rdm(identities, rdm, group_lookup, output_path):
    """Plot the perceptual RDM with identities sorted by group for readability."""
    # Sort identities by group, then alphabetically
    order_idx = sorted(range(len(identities)),
                       key=lambda i: (group_lookup.get(identities[i], 'ZZZ'),
                                      identities[i]))
    sorted_ids = [identities[i] for i in order_idx]
    sorted_rdm = rdm[np.ix_(order_idx, order_idx)]

    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(sorted_rdm, cmap='inferno', aspect='equal')
    ax.set_xticks(range(len(sorted_ids)))
    ax.set_yticks(range(len(sorted_ids)))
    ax.set_xticklabels(sorted_ids, rotation=90, fontsize=8)
    ax.set_yticklabels(sorted_ids, fontsize=8)

    # Draw thick lines between groups
    last_group = None
    for i, ident in enumerate(sorted_ids):
        g = group_lookup.get(ident, 'unknown')
        if g != last_group and i > 0:
            ax.axhline(i - 0.5, color='cyan', lw=1.5)
            ax.axvline(i - 0.5, color='cyan', lw=1.5)
        last_group = g

    plt.colorbar(im, ax=ax, label='cosine distance')
    ax.set_title('Perceptual RDM (CLIP ViT-B/32, mean over images per identity)\n'
                 'identities sorted by group')
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  RDM plot saved: {output_path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load monkeyinfo to optionally restrict to known identities
    print("Loading monkeyinfo.csv...")
    info = pd.read_csv(MONKEY_INFO_CSV)
    info['Name'] = info['Name'].astype(str).str.strip()
    restrict_names = set(info['Name']) if RESTRICT_TO_MONKEYINFO else None
    if restrict_names:
        print(f"  {len(restrict_names)} identities in monkeyinfo.csv")
    group_lookup = dict(zip(info['Name'], info['Group Name']))

    # 2. Connect + look up monkey_id ↔ name mapping
    print("\nConnecting to DB...")
    conn = connect_db()
    lookup_df = get_monkey_lookup(conn, restrict_names=restrict_names)
    conn.close()

    # 3. Find images for each identity
    print(f"\nSearching for images in {STIM_DIR}...")
    lookup_df = discover_images(lookup_df, STIM_DIR)
    print_image_discovery_summary(lookup_df)

    # Save diagnostics regardless of DRY_RUN
    diag_path = OUTPUT_DIR / 'diagnostics.txt'
    with open(diag_path, 'w') as f:
        f.write(f"Identity image discovery summary\n{'=' * 60}\n")
        for _, r in lookup_df.sort_values('name').iterrows():
            f.write(f"{r['name']:<8s}: {r['n_images']:>2d} images "
                    f"({len(r['monkey_ids'])} crops in DB)")
            if r['n_missing'] > 0:
                f.write(f"  [MISSING {r['n_missing']}: ids {r['missing_ids']}]")
            f.write("\n")
            for p in r['image_paths']:
                f.write(f"    {p}\n")
    print(f"\n  Diagnostics written: {diag_path}")

    if DRY_RUN:
        print("\n  DRY_RUN=True — stopping before CLIP inference.")
        print("  Verify the per-identity image counts above. If they look right,")
        print("  set DRY_RUN=False and rerun.")
        return

    # 4. Load CLIP and embed
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, processor = load_clip(CLIP_MODEL_NAME, device)

    identities, embeddings, rdm, used_paths = compute_perceptual_rdm(
        lookup_df, model, processor, device)

    # 5. Save
    out_pkl = OUTPUT_DIR / 'perceptual_rdm.pkl'
    with open(out_pkl, 'wb') as f:
        pickle.dump(dict(
            identities=identities,
            embeddings=embeddings,
            rdm=rdm,
            used_image_paths={k: [str(p) for p in v] for k, v in used_paths.items()},
            model=CLIP_MODEL_NAME,
            distance_metric='cosine',
        ), f)
    print(f"\n  Pickle saved: {out_pkl}")

    out_png = OUTPUT_DIR / 'perceptual_rdm.png'
    plot_rdm(identities, rdm, group_lookup, out_png)

    # Also write as CSV for easy inspection
    out_csv = OUTPUT_DIR / 'perceptual_rdm.csv'
    pd.DataFrame(rdm, index=identities, columns=identities).to_csv(out_csv)
    print(f"  CSV saved: {out_csv}")

    print(f"\n{'═' * 60}")
    print("  DONE.")
    print(f"{'═' * 60}")
    print(f"  Identities embedded: {len(identities)}")
    print(f"  Embedding dim: {embeddings.shape[1]}")
    print(f"  RDM shape: {rdm.shape}")
    print(f"  RDM range: [{rdm.min():.3f}, {rdm.max():.3f}]")
    print(f"  Output dir: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
