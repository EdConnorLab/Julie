#!/usr/bin/env python3
"""
Compute low-level image statistics for monkey crop photos.
Stores results in MySQL table: monkeys_image_stats

Computed features (all on the full crop):
  - mean_luminance        : mean of grayscale image (0-255)
  - rms_contrast          : std of grayscale image
  - mean_r, mean_g, mean_b: mean RGB channel values
  - mean_L, mean_a, mean_b_lab : mean CIE-Lab values
  - hf_energy             : fraction of spectral energy above Nyquist/4
                            (high = sharp, low = blurry/hazy)
  - img_width, img_height : crop dimensions in pixels

Computed features (on the head region only, using monkeysCoord):
  - head_mean_luminance
  - head_rms_contrast
  - head_hf_energy

Usage:
  python compute_image_stats.py              # process all
  python compute_image_stats.py --force      # recompute already-processed
"""

import argparse
import os
import re
import sys
import numpy as np
from PIL import Image
from tqdm import tqdm
import mysql.connector

# ============================================================
# CONFIGURATION
# ============================================================
DB_CONFIG = {
    'host': '172.30.6.59',
    'port': 3306,
    'user': 'xper_rw',
    'password': 'up2nite',
    'database': 'photo_metadata',
}
PHOTO_DIR = '/home/connorlab/Stimuli/OneMonkey'
TABLE_NAME = 'monkeys_image_stats'


def parse_coords(coord_str):
    """Parse '(x1, y1, x2, y2)' from DB into a tuple of ints."""
    if not coord_str:
        return None
    nums = re.findall(r'-?\d+', str(coord_str))
    if len(nums) == 4:
        return tuple(int(n) for n in nums)
    return None


def compute_hf_energy(gray_arr):
    """
    Fraction of 2D FFT energy in high-frequency band.
    High value = sharp image, low value = blurry.
    """
    f = np.fft.fft2(gray_arr.astype(np.float64))
    fshift = np.fft.fftshift(f)
    mag = np.abs(fshift) ** 2

    h, w = gray_arr.shape
    cy, cx = h // 2, w // 2
    # threshold at 1/4 of Nyquist
    radius = min(h, w) // 4

    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)

    total = mag.sum()
    if total == 0:
        return 0.0
    hf = mag[dist > radius].sum()
    return float(hf / total)


def rgb_to_lab(rgb_arr):
    """
    Convert RGB array (H,W,3) uint8 to CIE-Lab.
    Uses the sRGB -> XYZ -> Lab conversion with D65 illuminant.
    """
    # normalize to [0,1]
    rgb = rgb_arr.astype(np.float64) / 255.0

    # linearize sRGB
    mask = rgb > 0.04045
    rgb[mask] = ((rgb[mask] + 0.055) / 1.055) ** 2.4
    rgb[~mask] = rgb[~mask] / 12.92

    # sRGB -> XYZ (D65)
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041

    # normalize by D65 white point
    x /= 0.95047
    y /= 1.00000
    z /= 1.08883

    # XYZ -> Lab
    def f(t):
        delta = 6.0 / 29.0
        mask = t > delta ** 3
        out = np.empty_like(t)
        out[mask] = np.cbrt(t[mask])
        out[~mask] = t[~mask] / (3 * delta ** 2) + 4.0 / 29.0
        return out

    fx, fy, fz = f(x), f(y), f(z)
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b_lab = 200 * (fy - fz)

    return L, a, b_lab


def compute_stats(img_rgb, gray):
    """Compute stats for a given RGB image and its grayscale version."""
    gray_arr = np.array(gray, dtype=np.float64)
    rgb_arr = np.array(img_rgb, dtype=np.uint8)

    mean_lum = float(gray_arr.mean())
    rms_con = float(gray_arr.std())
    mean_r = float(rgb_arr[:, :, 0].mean())
    mean_g = float(rgb_arr[:, :, 1].mean())
    mean_b = float(rgb_arr[:, :, 2].mean())

    L, a, b_lab = rgb_to_lab(rgb_arr)
    mean_L = float(L.mean())
    mean_a = float(a.mean())
    mean_b_lab = float(b_lab.mean())

    hf = compute_hf_energy(gray_arr)

    return {
        'mean_luminance': round(mean_lum, 4),
        'rms_contrast': round(rms_con, 4),
        'mean_r': round(mean_r, 4),
        'mean_g': round(mean_g, 4),
        'mean_b': round(mean_b, 4),
        'mean_L': round(mean_L, 4),
        'mean_a': round(mean_a, 4),
        'mean_b_lab': round(mean_b_lab, 4),
        'hf_energy': round(hf, 6),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', action='store_true',
                        help='Recompute stats for already-processed photos')
    args = parser.parse_args()

    # connect
    db = mysql.connector.connect(**DB_CONFIG)
    cur = db.cursor(dictionary=True)

    # create table
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS `{TABLE_NAME}` (
            monkey_id         INT PRIMARY KEY,
            mean_luminance    FLOAT,
            rms_contrast      FLOAT,
            mean_r            FLOAT,
            mean_g            FLOAT,
            mean_b            FLOAT,
            mean_L            FLOAT,
            mean_a            FLOAT,
            mean_b_lab        FLOAT,
            hf_energy         FLOAT,
            img_width         INT,
            img_height        INT,
            head_mean_luminance FLOAT,
            head_rms_contrast   FLOAT,
            head_hf_energy      FLOAT,
            computed_at       DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    db.commit()

    # load coords for head region
    cur.execute('SELECT monkey_id, body, face FROM monkeysCoord')
    coords = {}
    for r in cur.fetchall():
        coords[r['monkey_id']] = {
            'body': parse_coords(r['body']),
            'face': parse_coords(r['face']),
        }

    # find already-processed
    already_done = set()
    if not args.force:
        cur.execute(f'SELECT monkey_id FROM `{TABLE_NAME}`')
        already_done = {r['monkey_id'] for r in cur.fetchall()}

    # scan photos
    avail = {}
    for f in os.listdir(PHOTO_DIR):
        m = re.match(r'^(\d+)\.[Jj][Pp][Gg]$', f)
        if m:
            avail[int(m.group(1))] = os.path.join(PHOTO_DIR, f)

    to_process = sorted(set(avail.keys()) - already_done)
    total = len(to_process)
    print(f'Photos to process: {total}  (already done: {len(already_done)}, '
          f'force={args.force})')

    if total == 0:
        print('Nothing to do.')
        cur.close()
        db.close()
        return

    insert_sql = f"""
        INSERT INTO `{TABLE_NAME}`
            (monkey_id, mean_luminance, rms_contrast,
             mean_r, mean_g, mean_b,
             mean_L, mean_a, mean_b_lab,
             hf_energy, img_width, img_height,
             head_mean_luminance, head_rms_contrast, head_hf_energy)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            mean_luminance  = VALUES(mean_luminance),
            rms_contrast    = VALUES(rms_contrast),
            mean_r          = VALUES(mean_r),
            mean_g          = VALUES(mean_g),
            mean_b          = VALUES(mean_b),
            mean_L          = VALUES(mean_L),
            mean_a          = VALUES(mean_a),
            mean_b_lab      = VALUES(mean_b_lab),
            hf_energy       = VALUES(hf_energy),
            img_width       = VALUES(img_width),
            img_height      = VALUES(img_height),
            head_mean_luminance = VALUES(head_mean_luminance),
            head_rms_contrast   = VALUES(head_rms_contrast),
            head_hf_energy      = VALUES(head_hf_energy),
            computed_at     = CURRENT_TIMESTAMP
    """

    errors = []
    for i, mid in enumerate(tqdm(to_process, desc='Computing stats', unit='img')):
        path = avail[mid]
        try:
            img = Image.open(path).convert('RGB')
            gray = img.convert('L')
            w, h = img.size

            # full-image stats
            s = compute_stats(img, gray)

            # head region stats
            cd = coords.get(mid)
            head_lum = None
            head_con = None
            head_hf = None
            if cd and cd['body'] and cd['face']:
                bx1, by1, _, _ = cd['body']
                fx1, fy1, fx2, fy2 = cd['face']
                # offset to crop coords
                hx1 = max(0, fx1 - bx1)
                hy1 = max(0, fy1 - by1)
                hx2 = min(w, fx2 - bx1)
                hy2 = min(h, fy2 - by1)
                if hx2 > hx1 and hy2 > hy1:
                    head_crop = img.crop((hx1, hy1, hx2, hy2))
                    head_gray = head_crop.convert('L')
                    head_arr = np.array(head_gray, dtype=np.float64)
                    head_lum = round(float(head_arr.mean()), 4)
                    head_con = round(float(head_arr.std()), 4)
                    if head_arr.shape[0] > 4 and head_arr.shape[1] > 4:
                        head_hf = round(compute_hf_energy(head_arr), 6)

            cur.execute(insert_sql, (
                mid,
                s['mean_luminance'], s['rms_contrast'],
                s['mean_r'], s['mean_g'], s['mean_b'],
                s['mean_L'], s['mean_a'], s['mean_b_lab'],
                s['hf_energy'], w, h,
                head_lum, head_con, head_hf,
            ))

            # commit in batches
            if (i + 1) % 50 == 0:
                db.commit()

        except Exception as e:
            errors.append((mid, str(e)))
            tqdm.write(f'  ERROR monkey_id {mid}: {e}')

    db.commit()
    print(f'\nDone. Processed {total - len(errors)}/{total} successfully.')
    if errors:
        print(f'{len(errors)} errors:')
        for mid, msg in errors[:10]:
            print(f'  monkey_id {mid}: {msg}')

    cur.close()
    db.close()


if __name__ == '__main__':
    main()
