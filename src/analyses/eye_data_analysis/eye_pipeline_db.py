"""
Eye-tracking analysis pipeline for the macaque social-photo experiment.

Reads directly from the recording schema in MySQL.

To use in PyCharm: just hit Run. Edit the CONFIG block below to change schema,
number of trials, which eye, etc.
"""

import ast
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import mysql.connector
import numpy as np
import pandas as pd


# ============================================================================
# CONFIG — edit these and hit Run
# ============================================================================
SCHEMA           = None                    # set to a specific schema like "20231009_recording"
                                           # to run just one day, or leave as None to auto-discover
                                           # from PKL_DIR below

# Folder containing pkl files like "2023-09-26_round_1.pkl".
# Dates are extracted from filenames and converted to schema names.
PKL_DIR          = "/home/connorlab/Documents/GitHub/Julie/Cortana/sorted_spike_cache_filtered"
N_TRIALS_TO_PLOT = 9                       # how many real trials to plot
EYE_TO_USE       = "leftIscan"             # 'leftIscan' or 'rightIscan'
OUTPUT_DIR       = Path("./eye_track_plots")

# Metadata schema (face / body bboxes per monkey_id, in NATIVE photo pixels).
# monkey_id matches the number in the stimulus filename, e.g. 2307.JPG -> 2307.
METADATA_SCHEMA  = "photo_metadata"
METADATA_TABLE   = "monkeysCoord"

# Bbox is (x1, y1, x2, y2) in native photo pixels. We need to know which corner
# the bbox is measured from. Standard image convention: (0,0) = TOP-LEFT, y goes
# DOWN. If face boxes end up upside-down on the plot, flip this:
BBOX_Y_FROM_TOP  = True   # True = y=0 at top of photo (standard), False = y=0 at bottom

DB_HOST = "localhost"
DB_USER = "xper_rw"
DB_PASS = "up2nite"

SKIP_MOCK_PATHS = ("animal-images",)       # substrings of paths to ignore
MAX_TRIAL_DUR_S = 10.0                     # discard trials longer than this (bad SlideOff match)


# ============================================================================
# DATABASE
# ============================================================================
def connect():
    return mysql.connector.connect(host=DB_HOST, user=DB_USER, passwd=DB_PASS)


def read_system_vars(conn, schema):
    """Read screen-geometry params from the schema's SystemVar table."""
    needed = [
        "xper_monkey_screen_distance",
        "xper_monkey_screen_width",
        "xper_monkey_screen_height",
    ]
    # Some setups also have resolution vars; we'll try a few common names.
    optional = [
        "xper_monkey_screen_resolution_width",
        "xper_monkey_screen_resolution_height",
        "xper_monkey_screen_pixel_width",
        "xper_monkey_screen_pixel_height",
    ]

    cur = conn.cursor()
    cur.execute(f"SELECT name, val FROM `{schema}`.SystemVar")
    rows = dict(cur.fetchall())
    cur.close()

    out = {}
    for k in needed:
        if k not in rows:
            raise RuntimeError(
                f"Missing required SystemVar '{k}' in {schema}.SystemVar.\n"
                f"Available names: {sorted(rows.keys())}"
            )
        out[k] = float(rows[k])

    for k in optional:
        if k in rows:
            out[k] = float(rows[k])

    # Try to pick resolution out of whatever variant the DB uses
    width_px = (out.get("xper_monkey_screen_resolution_width")
                or out.get("xper_monkey_screen_pixel_width"))
    height_px = (out.get("xper_monkey_screen_resolution_height")
                 or out.get("xper_monkey_screen_pixel_height"))

    if width_px is None or height_px is None:
        print("WARNING: screen resolution not found in SystemVar — defaulting to 3840x2160.")
        print(f"  Available SystemVar names: {sorted(rows.keys())}")
        width_px, height_px = 3840.0, 2160.0

    return {
        "screen_distance_mm": out["xper_monkey_screen_distance"],
        "screen_width_mm":    out["xper_monkey_screen_width"],
        "screen_height_mm":   out["xper_monkey_screen_height"],
        "screen_width_px":    width_px,
        "screen_height_px":   height_px,
    }


def read_behmsg(conn, schema):
    """Pull only the event types we care about — keeps memory low."""
    cur = conn.cursor()
    cur.execute(f"""
        SELECT tstamp, type, msg
        FROM `{schema}`.BehMsg
        WHERE type IN ('SlideOn','SlideOff','TrialStart','TrialStop',
                       'TrialComplete','FixationSucceed','InitialEyeInSucceed')
        ORDER BY tstamp
    """)
    rows = cur.fetchall()
    cur.close()
    return pd.DataFrame(rows, columns=['tstamp', 'type', 'msg'])


def read_behmsgeye(conn, schema, t_start, t_end):
    """Pull eye-device samples for one time window (microsecond tstamps)."""
    cur = conn.cursor()
    cur.execute(f"""
        SELECT tstamp, msg
        FROM `{schema}`.BehMsgEye
        WHERE type = 'EyeDeviceMessage'
          AND tstamp BETWEEN %s AND %s
        ORDER BY tstamp
    """, (int(t_start), int(t_end)))
    rows = cur.fetchall()
    cur.close()
    return pd.DataFrame(rows, columns=['tstamp', 'msg'])


# ============================================================================
# PARSING
# ============================================================================
def parse_slide_on(msg):
    if msg is None: return None
    try:
        root = ET.fromstring(msg)
        spec = root.find('spec')
        return {
            'taskId': root.findtext('taskId'),
            'filePath': spec.findtext('filePath'),
            'spec_width_px': float(spec.findtext('width')),
            'spec_height_px': float(spec.findtext('height')),
            'headHeight': float(spec.findtext('headHeight') or 0),
        }
    except Exception:
        return None


def parse_eye_msg(msg):
    try:
        root = ET.fromstring(msg)
        return {
            'eye_id': root.findtext('id'),
            'deg_x': float(root.find('degree/x').text),
            'deg_y': float(root.find('degree/y').text),
        }
    except Exception:
        return None


def get_taskid(msg):
    if msg is None: return None
    try:
        return ET.fromstring(msg).findtext('taskId')
    except Exception:
        return None


def build_slide_events(behmsg):
    """Each row = one trial slide, with on/off timestamps + stim spec."""
    so = behmsg[behmsg['type'] == 'SlideOn'].copy()
    parsed = pd.DataFrame(so['msg'].apply(parse_slide_on).tolist())
    so = pd.concat([so.reset_index(drop=True), parsed], axis=1)
    so = so.rename(columns={'tstamp': 'slide_on_ts'}).drop(columns=['type', 'msg'])

    off = behmsg[behmsg['type'] == 'SlideOff'].copy()
    off['taskId'] = off['msg'].apply(get_taskid)
    off = off.rename(columns={'tstamp': 'slide_off_ts'})[['taskId', 'slide_off_ts']]

    merged = so.merge(off, on='taskId', how='left')
    merged['slide_on_ts'] = pd.to_numeric(merged['slide_on_ts'], errors='coerce')
    merged['slide_off_ts'] = pd.to_numeric(merged['slide_off_ts'], errors='coerce')
    return merged


def build_eye_df(behmsgeye):
    parsed = pd.DataFrame(behmsgeye['msg'].apply(parse_eye_msg).tolist())
    df = pd.concat([behmsgeye[['tstamp']].reset_index(drop=True), parsed], axis=1)
    return df.dropna()


# ============================================================================
# METADATA (face / body bboxes per monkey_id)
# ============================================================================
def load_metadata(conn):
    """Read face/body bboxes from photo_metadata.monkeysCoord.
    Returns dict: monkey_id -> {'body': (x1,y1,x2,y2) or None,
                                'face': (x1,y1,x2,y2) or None}."""
    cur = conn.cursor()
    cur.execute(f"SELECT monkey_id, body, face FROM `{METADATA_SCHEMA}`.`{METADATA_TABLE}`")
    rows = cur.fetchall()
    cur.close()

    def safe_parse(v):
        if v is None: return None
        try: return tuple(ast.literal_eval(str(v)))
        except Exception: return None

    return {int(mid): {'body': safe_parse(body), 'face': safe_parse(face)}
            for mid, body, face in rows}


def monkey_id_from_path(path):
    """Extract monkey_id from filename like '/.../2307.JPG' -> 2307."""
    if not path: return None
    stem = Path(path).stem
    m = re.match(r'(\d+)', stem)
    return int(m.group(1)) if m else None


def bbox_native_to_screen(bbox, native_wh, display_wh, y_from_top=True):
    """
    Convert a bbox (already in cropped-image pixel coords) to screen-centered coords.

    bbox       : (x1, y1, x2, y2) in cropped image pixels
    native_wh  : (width, height) of the cropped stimulus image file
    display_wh : (width, height) of the image as displayed on screen
    y_from_top : True = y=0 at top of image (standard), False = y=0 at bottom

    Returns (x_left, y_bottom, rect_width, rect_height) for matplotlib Rectangle.
    """
    if bbox is None or native_wh is None or display_wh is None:
        return None
    x1, y1, x2, y2 = bbox
    nw, nh = native_wh
    dw, dh = display_wh
    sx, sy = dw / nw, dh / nh

    # Scale to display pixels
    x1s, x2s = x1 * sx, x2 * sx
    y1s, y2s = y1 * sy, y2 * sy

    # Center on origin; higher y = visually up in matplotlib
    left = -dw / 2 + x1s
    width = x2s - x1s
    if y_from_top:
        top    =  dh / 2 - y1s
        bottom =  dh / 2 - y2s
    else:
        bottom = -dh / 2 + y1s
        top    = -dh / 2 + y2s

    return (left, bottom, width, top - bottom)


def face_bbox_in_crop(body_bbox, face_bbox):
    """
    Convert face bbox from original-photo coords to cropped-image coords.
    The crop is defined by body_bbox.

    body_bbox : (x1, y1, x2, y2) in original photo pixels — defines the crop
    face_bbox : (x1, y1, x2, y2) in original photo pixels
    Returns   : (x1, y1, x2, y2) in cropped image pixels, or None
    """
    if body_bbox is None or face_bbox is None:
        return None
    bx1, by1, bx2, by2 = body_bbox
    fx1, fy1, fx2, fy2 = face_bbox
    return (fx1 - bx1, fy1 - by1, fx2 - bx1, fy2 - by1)


def compute_display_size(native_wh, screen_params):
    """
    Compute how large the image appears on screen (in screen pixels),
    assuming xper scales to fit the screen while maintaining aspect ratio.
    """
    nw, nh = native_wh
    sw = screen_params['screen_width_px']
    sh = screen_params['screen_height_px']
    scale = min(sw / nw, sh / nh)
    return (nw * scale, nh * scale)


# ============================================================================
# COORDINATE CONVERSION (deg of visual angle -> screen pixels)
# ============================================================================
def deg2mm(deg, distance_mm):
    return math.tan((deg / 2.0) * (math.pi / 180.0)) * 2.0 * distance_mm


def deg_to_screen_pixels(x_deg, y_deg, sp):
    x_mm = np.array([deg2mm(d, sp['screen_distance_mm']) for d in x_deg])
    y_mm = np.array([deg2mm(d, sp['screen_distance_mm']) for d in y_deg])
    return (x_mm / (sp['screen_width_mm']  / sp['screen_width_px']),
            y_mm / (sp['screen_height_mm'] / sp['screen_height_px']))


# ============================================================================
# PLOTTING
# ============================================================================
def plot_trial(trial, eye_df, screen_params, metadata,
               eye_to_use=EYE_TO_USE, ax=None):
    """Overlay eye gaze + face/body bboxes on the stimulus image."""
    mask = ((eye_df['tstamp'] >= trial['slide_on_ts']) &
            (eye_df['tstamp'] <= trial['slide_off_ts']) &
            (eye_df['eye_id'] == eye_to_use))
    eye_trial = eye_df.loc[mask].sort_values('tstamp')
    if len(eye_trial) == 0:
        print(f"  no eye samples for taskId={trial['taskId']}")
        return None

    x_px, y_px = deg_to_screen_pixels(eye_trial['deg_x'].values,
                                      eye_trial['deg_y'].values,
                                      screen_params)

    img_path = trial['filePath']
    native_wh = None
    display_wh = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # Load actual stimulus and compute displayed size on screen
    try:
        im = plt.imread(img_path)
        native_wh = (im.shape[1], im.shape[0])  # (width, height)
        display_wh = compute_display_size(native_wh, screen_params)
        dw, dh = display_wh
        print(f"  {Path(img_path).name}: native={native_wh}, display=({dw:.0f}, {dh:.0f})")
        extent = [-dw/2, dw/2, -dh/2, dh/2]
        ax.imshow(im, extent=extent, aspect='auto')
    except Exception as e:
        print(f"  could not load {img_path}: {e}")
        # Fallback: use spec dims as rough extent
        dw, dh = trial['spec_width_px'], trial['spec_height_px']
        extent = [-dw/2, dw/2, -dh/2, dh/2]
        ax.add_patch(patches.Rectangle((-dw/2, -dh/2), dw, dh, fill=False,
                                       edgecolor='black'))

    # Eye track
    ax.plot(x_px, y_px, '-', color='red', linewidth=0.6, alpha=0.4)
    ax.scatter(x_px, y_px, c=np.arange(len(x_px)), cmap='autumn',
               s=10, edgecolors='none')

    # Face / body bbox overlays from metadata
    mid = monkey_id_from_path(img_path)
    meta = metadata.get(mid) if mid is not None else None
    if meta and native_wh and display_wh:
        body_bb = meta.get('body')
        face_bb = meta.get('face')

        # Face bbox is in ORIGINAL photo coords; stimulus is cropped to body bbox.
        # Convert face coords to cropped-image coords.
        face_in_crop = face_bbox_in_crop(body_bb, face_bb)

        # Sanity check: native image dims should roughly match body crop size
        if body_bb:
            crop_w = body_bb[2] - body_bb[0]
            crop_h = body_bb[3] - body_bb[1]
            nw, nh = native_wh
            if abs(crop_w - nw) > 50 or abs(crop_h - nh) > 50:
                print(f"  WARNING {mid}: native=({nw},{nh}) vs body crop=({crop_w},{crop_h}) — mismatch!")
            body_in_crop = (0, 0, crop_w, crop_h)
            rect = bbox_native_to_screen(body_in_crop, native_wh, display_wh,
                                         y_from_top=BBOX_Y_FROM_TOP)
            if rect:
                lx, by, bw, bh = rect
                ax.add_patch(patches.Rectangle(
                    (lx, by), bw, bh, fill=False, edgecolor='lime',
                    linewidth=2, linestyle='--', label='body (full crop)'))

        if face_in_crop:
            rect = bbox_native_to_screen(face_in_crop, native_wh, display_wh,
                                         y_from_top=BBOX_Y_FROM_TOP)
            if rect:
                lx, by, bw, bh = rect
                ax.add_patch(patches.Rectangle(
                    (lx, by), bw, bh, fill=False, edgecolor='cyan',
                    linewidth=2, linestyle='--', label='face'))

        ax.legend(loc='lower right', fontsize=8)

    ax.set_xlim(-dw/2 * 1.3, dw/2 * 1.3)
    ax.set_ylim(-dh/2 * 1.3, dh/2 * 1.3)
    ax.set_aspect('equal')
    ax.set_title(f"{Path(img_path).name}  id={mid}  N={len(eye_trial)}  "
                 f"dur={(trial['slide_off_ts']-trial['slide_on_ts'])/1e6:.2f}s",
                 fontsize=9)
    return ax


# ============================================================================
# SCHEMA DISCOVERY
# ============================================================================
def discover_schemas(pkl_dir):
    """
    Extract recording dates from pkl filenames like '2023-09-26_round_1.pkl'
    and return sorted list of unique schema names like '20230926_recording'.
    """
    schemas = set()
    pkl_path = Path(pkl_dir)
    if not pkl_path.exists():
        print(f"WARNING: {pkl_dir} does not exist")
        return []
    for f in pkl_path.glob("*.pkl"):
        m = re.match(r'(\d{4})-(\d{2})-(\d{2})_', f.name)
        if m:
            date_str = m.group(1) + m.group(2) + m.group(3)
            schemas.add(f"{date_str}_recording")
    return sorted(schemas)


# ============================================================================
# MAIN
# ============================================================================
def run_one_schema(conn, schema, metadata):
    """Run the pipeline for a single recording schema."""
    print(f"\n{'='*60}")
    print(f"  Processing: {schema}")
    print(f"{'='*60}")

    print("Reading SystemVar...")
    screen_params = read_system_vars(conn, schema)
    print("  screen params:", screen_params)

    print("Reading BehMsg (events only)...")
    behmsg = read_behmsg(conn, schema)
    print(f"  {len(behmsg)} rows")

    slides = build_slide_events(behmsg)
    real = slides[~slides['filePath'].str.contains('|'.join(SKIP_MOCK_PATHS),
                                                   na=False)].copy()
    real = real.dropna(subset=['slide_off_ts'])

    # Filter out trials with unreasonable duration (bad SlideOff match)
    real['dur_s'] = (real['slide_off_ts'] - real['slide_on_ts']) / 1e6
    n_before = len(real)
    real = real[real['dur_s'] <= MAX_TRIAL_DUR_S]
    n_dropped = n_before - len(real)
    print(f"  {len(slides)} total slides; {len(real)} real (non-mock)"
          + (f" ({n_dropped} dropped for dur > {MAX_TRIAL_DUR_S}s)" if n_dropped else ""))

    if len(real) == 0:
        print("  No real trials found — skipping.")
        return

    # Pick evenly spaced trials across the session
    n_to_plot = min(N_TRIALS_TO_PLOT, len(real))
    indices = np.linspace(0, len(real) - 1, n_to_plot, dtype=int)
    demo = real.iloc[indices]
    print(f"  Selected {len(demo)} evenly spaced trials "
          f"(indices: {indices.tolist()} out of {len(real)})")

    # Read eye data only for the window covering those demo trials
    t_start = int(demo['slide_on_ts'].min()) - 100_000
    t_end   = int(demo['slide_off_ts'].max()) + 100_000
    print(f"Reading BehMsgEye for window {t_start}..{t_end} "
          f"({(t_end-t_start)/1e6:.1f} s)...")
    behmsgeye = read_behmsgeye(conn, schema, t_start, t_end)
    print(f"  {len(behmsgeye)} eye samples")

    eye_df = build_eye_df(behmsgeye)
    print(f"  {len(eye_df)} parsed eye rows; "
          f"eyes: {eye_df['eye_id'].unique().tolist()}")

    # Plot in a grid
    n = len(demo)
    cols = min(3, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
    axes = np.atleast_1d(axes).flatten()
    for ax, (_, trial) in zip(axes, demo.iterrows()):
        plot_trial(trial, eye_df, screen_params, metadata, ax=ax)
    for ax in axes[len(demo):]:
        ax.axis('off')

    plt.tight_layout()
    out = OUTPUT_DIR / f"{schema}_eye_tracks.png"
    plt.savefig(out, dpi=120, bbox_inches='tight')
    print(f"Saved: {out}")
    plt.show()
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    conn = connect()

    print("Loading metadata from DB...")
    try:
        metadata = load_metadata(conn)
        print(f"  {len(metadata)} monkey_id entries")
    except Exception as e:
        print(f"  WARNING: could not read metadata — bboxes will be skipped: {e}")
        metadata = {}

    # Determine which schemas to run
    if SCHEMA:
        schemas = [SCHEMA]
    else:
        schemas = discover_schemas(PKL_DIR)
        print(f"\nDiscovered {len(schemas)} schemas from {PKL_DIR}:")
        for s in schemas:
            print(f"  {s}")

    for schema in schemas:
        try:
            run_one_schema(conn, schema, metadata)
        except Exception as e:
            print(f"  ERROR on {schema}: {e}")
            continue

    conn.close()
    print(f"\nDone. All figures saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
