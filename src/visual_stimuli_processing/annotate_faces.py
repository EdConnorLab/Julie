#!/usr/bin/env python3
"""
Monkey Face Annotation Tool
----------------------------
Annotate head pose, face direction, occlusion, and image quality
for monkey photos. Saves annotations to MySQL table: monkeys_face_annotations

Keyboard shortcuts:
  1-5    Head pose (Frontal / 3/4 / Profile / 3/4 Away / Back)
  Q/W/E  Direction (Left / Center / Right)
  A/S/D  Occlusion (None / Partial / Heavy)
  Z/X/C  Image quality (Clear / Out of Focus / Mesh Haze)
  B      Quick "No Face / Back" (auto-saves & advances)
  Enter  Save & Next
  Left   Previous photo
  Right  Skip / Next
"""

import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk, ImageDraw, ImageFont
import mysql.connector
import os
import re

# ============================================================
# CONFIGURATION — update password before running
# ============================================================
DB_CONFIG = {
    'host': '172.30.6.59',
    'port': 3306,
    'user': 'xper_rw',
    'password': 'YOUR_PASSWORD_HERE',  # <-- UPDATE THIS
    'database': 'photo_metadata',
}
PHOTO_DIR = '/home/connorlab/Stimuli/OneMonkey'
TABLE_NAME = 'monkeys_face_annotations'
DISPLAY_MAX = (700, 700)

# ============================================================
# CATEGORY DEFINITIONS
# ============================================================
HEAD_POSES = [
    ('Frontal (~0°)',           'frontal'),
    ('Three-Qtr (~45°)',       'three_quarter'),
    ('Profile (~90°)',          'profile'),
    ('Three-Qtr Away (~135°)', 'three_quarter_away'),
    ('Back (~180°)',            'back'),
]
DIRECTIONS = [
    ('Left',   'left'),
    ('Center', 'center'),
    ('Right',  'right'),
]
OCCLUSIONS = [
    ('None',              'none'),
    ('Partial (cage/obj)', 'partial'),
    ('Heavy',              'heavy'),
]
IMG_QUALITIES = [
    ('Clear',          'clear'),
    ('Out of Focus',   'out_of_focus'),
    ('Mesh Haze',      'mesh_haze'),
]


def parse_coords(coord_str):
    """Parse '(x1, y1, x2, y2)' from DB into a tuple of ints."""
    if not coord_str:
        return None
    nums = re.findall(r'-?\d+', str(coord_str))
    if len(nums) == 4:
        return tuple(int(n) for n in nums)
    return None


class AnnotationTool:
    def __init__(self, root):
        self.root = root
        self.root.title('Monkey Face Annotation Tool')
        self.root.geometry('1200x980')
        self.root.configure(bg='#1e1e1e')

        # state
        self.idx = 0
        self.id_list = []          # monkey_ids in current view
        self.all_ids = []          # every valid monkey_id
        self.unannotated = []      # ids not yet labeled
        self.coords = {}           # mid -> {body, face}
        self.names = {}            # mid -> monkey_name
        self.saved = {}            # mid -> annotation dict
        self.photo_tk = None       # prevent GC

        # current selections
        self.sel_pose = tk.StringVar(value='')
        self.sel_dir  = tk.StringVar(value='')
        self.sel_occ  = tk.StringVar(value='')
        self.sel_qual = tk.StringVar(value='clear')

        # DB
        self._connect_db()
        self._ensure_table()
        self._load_data()

        # GUI
        self._build_ui()
        self._bind_keys()

        if self.id_list:
            self._show()
        else:
            messagebox.showinfo('Done', 'No photos to annotate.')

    # ── database ────────────────────────────────────────────
    def _connect_db(self):
        try:
            self.db = mysql.connector.connect(**DB_CONFIG)
            self.cur = self.db.cursor(dictionary=True)
        except Exception as e:
            messagebox.showerror('DB Error', f'Cannot connect:\n{e}')
            raise

    def _ensure_table(self):
        self.cur.execute(f"""
            CREATE TABLE IF NOT EXISTS `{TABLE_NAME}` (
                monkey_id      INT PRIMARY KEY,
                head_pose      ENUM('frontal','three_quarter','profile',
                                    'three_quarter_away','back') NOT NULL,
                face_direction ENUM('left','right','center','na') NOT NULL,
                occlusion      ENUM('none','partial','heavy') NOT NULL,
                image_quality  ENUM('clear','out_of_focus','mesh_haze')
                               NOT NULL DEFAULT 'clear',
                annotated_at   DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at     DATETIME DEFAULT CURRENT_TIMESTAMP
                               ON UPDATE CURRENT_TIMESTAMP
            )
        """)
        self.db.commit()
        # add column if table already existed without it
        try:
            self.cur.execute(f"""
                ALTER TABLE `{TABLE_NAME}`
                ADD COLUMN image_quality ENUM('clear','out_of_focus','mesh_haze')
                NOT NULL DEFAULT 'clear'
                AFTER occlusion
            """)
            self.db.commit()
            print('Added image_quality column to existing table')
        except mysql.connector.errors.ProgrammingError:
            pass  # column already exists

    def _load_data(self):
        # coordinates
        self.cur.execute('SELECT monkey_id, body, face FROM monkeysCoord')
        for r in self.cur.fetchall():
            self.coords[r['monkey_id']] = {
                'body': parse_coords(r['body']),
                'face': parse_coords(r['face']),
            }

        # names
        self.cur.execute('SELECT monkey_id, monkey_name FROM monkeys')
        for r in self.cur.fetchall():
            self.names[r['monkey_id']] = r['monkey_name']

        # existing annotations
        self.cur.execute(
            f'SELECT monkey_id, head_pose, face_direction, occlusion, '
            f'image_quality FROM `{TABLE_NAME}`')
        for r in self.cur.fetchall():
            self.saved[r['monkey_id']] = {
                'head_pose': r['head_pose'],
                'direction': r['face_direction'],
                'occlusion': r['occlusion'],
                'image_quality': r['image_quality'],
            }

        # scan folder for available JPGs
        avail = set()
        for f in os.listdir(PHOTO_DIR):
            m = re.match(r'^(\d+)\.[Jj][Pp][Gg]$', f)
            if m:
                avail.add(int(m.group(1)))

        valid = sorted(avail & set(self.coords.keys()))
        self.all_ids = valid
        self.unannotated = [m for m in valid if m not in self.saved]
        self.id_list = list(self.unannotated) if self.unannotated else list(valid)

        print(f'Photos: {len(valid)} total, '
              f'{len(self.saved)} annotated, '
              f'{len(self.unannotated)} remaining')

    # ── GUI construction ────────────────────────────────────
    def _build_ui(self):
        style_bg = '#1e1e1e'
        style_fg = '#e0e0e0'
        btn_font = ('Segoe UI', 10)
        lbl_font = ('Segoe UI', 11)

        # --- top bar ---
        top = tk.Frame(self.root, bg=style_bg)
        top.pack(fill=tk.X, padx=12, pady=(8, 2))

        self.lbl_progress = tk.Label(top, text='', font=lbl_font,
                                     bg=style_bg, fg='#aaa')
        self.lbl_progress.pack(side=tk.LEFT)

        self.lbl_info = tk.Label(top, text='', font=('Segoe UI', 11, 'bold'),
                                 bg=style_bg, fg='#fff')
        self.lbl_info.pack(side=tk.RIGHT)

        # --- mode / jump ---
        mode_fr = tk.Frame(self.root, bg=style_bg)
        mode_fr.pack(fill=tk.X, padx=12, pady=2)

        self.mode_var = tk.StringVar(value='unannotated')
        for text, val in [('Unannotated', 'unannotated'), ('Review all', 'all')]:
            tk.Radiobutton(mode_fr, text=text, variable=self.mode_var,
                           value=val, command=self._switch_mode,
                           bg=style_bg, fg=style_fg, selectcolor='#333',
                           activebackground=style_bg,
                           font=btn_font).pack(side=tk.LEFT, padx=(0, 10))

        tk.Label(mode_fr, text='Jump to ID:', bg=style_bg, fg=style_fg,
                 font=btn_font).pack(side=tk.LEFT, padx=(20, 4))
        self.jump_entry = tk.Entry(mode_fr, width=7, font=btn_font)
        self.jump_entry.pack(side=tk.LEFT)
        tk.Button(mode_fr, text='Go', command=self._jump,
                  font=btn_font, width=4).pack(side=tk.LEFT, padx=4)

        # --- live counter ---
        ctr_fr = tk.Frame(self.root, bg=style_bg)
        ctr_fr.pack(fill=tk.X, padx=12, pady=(2, 0))
        self.ctr_labels = {}
        # pose counts
        tk.Label(ctr_fr, text='Pose:', font=('Segoe UI', 9, 'bold'),
                 bg=style_bg, fg='#888').pack(side=tk.LEFT)
        for _, val in HEAD_POSES:
            lbl = tk.Label(ctr_fr, text=f'{val}: 0', font=('Segoe UI', 9),
                           bg=style_bg, fg='#aaa', padx=4)
            lbl.pack(side=tk.LEFT)
            self.ctr_labels[f'pose_{val}'] = lbl
        # separator
        tk.Label(ctr_fr, text='  |  ', bg=style_bg, fg='#555',
                 font=('Segoe UI', 9)).pack(side=tk.LEFT)
        # quality counts
        tk.Label(ctr_fr, text='Quality:', font=('Segoe UI', 9, 'bold'),
                 bg=style_bg, fg='#888').pack(side=tk.LEFT)
        for _, val in IMG_QUALITIES:
            lbl = tk.Label(ctr_fr, text=f'{val}: 0', font=('Segoe UI', 9),
                           bg=style_bg, fg='#aaa', padx=4)
            lbl.pack(side=tk.LEFT)
            self.ctr_labels[f'qual_{val}'] = lbl

        # --- canvas ---
        self.canvas = tk.Canvas(self.root, bg='#2b2b2b',
                                width=DISPLAY_MAX[0], height=DISPLAY_MAX[1],
                                highlightthickness=0)
        self.canvas.pack(padx=12, pady=6)

        # --- annotation panel ---
        ann = tk.LabelFrame(self.root, text=' Annotation ',
                            font=('Segoe UI', 10, 'bold'),
                            bg=style_bg, fg=style_fg, bd=1)
        ann.pack(fill=tk.X, padx=12, pady=4)

        # head pose row
        r1 = tk.Frame(ann, bg=style_bg)
        r1.pack(fill=tk.X, padx=6, pady=3)
        tk.Label(r1, text='Head Pose:', width=11, anchor='w',
                 font=('Segoe UI', 10, 'bold'),
                 bg=style_bg, fg=style_fg).pack(side=tk.LEFT)
        for i, (label, val) in enumerate(HEAD_POSES):
            tk.Radiobutton(r1, text=f'[{i+1}] {label}',
                           variable=self.sel_pose, value=val,
                           indicatoron=0, width=22, pady=4,
                           font=btn_font, selectcolor='#2e7d32',
                           bg='#333', fg=style_fg,
                           activebackground='#444',
                           ).pack(side=tk.LEFT, padx=2)

        # direction row
        r2 = tk.Frame(ann, bg=style_bg)
        r2.pack(fill=tk.X, padx=6, pady=3)
        tk.Label(r2, text='Direction:', width=11, anchor='w',
                 font=('Segoe UI', 10, 'bold'),
                 bg=style_bg, fg=style_fg).pack(side=tk.LEFT)
        keys_d = ['Q', 'W', 'E']
        for i, (label, val) in enumerate(DIRECTIONS):
            tk.Radiobutton(r2, text=f'[{keys_d[i]}] {label}',
                           variable=self.sel_dir, value=val,
                           indicatoron=0, width=14, pady=4,
                           font=btn_font, selectcolor='#1565c0',
                           bg='#333', fg=style_fg,
                           activebackground='#444',
                           ).pack(side=tk.LEFT, padx=2)

        # occlusion row
        r3 = tk.Frame(ann, bg=style_bg)
        r3.pack(fill=tk.X, padx=6, pady=3)
        tk.Label(r3, text='Occlusion:', width=11, anchor='w',
                 font=('Segoe UI', 10, 'bold'),
                 bg=style_bg, fg=style_fg).pack(side=tk.LEFT)
        keys_o = ['A', 'S', 'D']
        for i, (label, val) in enumerate(OCCLUSIONS):
            tk.Radiobutton(r3, text=f'[{keys_o[i]}] {label}',
                           variable=self.sel_occ, value=val,
                           indicatoron=0, width=18, pady=4,
                           font=btn_font, selectcolor='#e65100',
                           bg='#333', fg=style_fg,
                           activebackground='#444',
                           ).pack(side=tk.LEFT, padx=2)

        # image quality row
        r4 = tk.Frame(ann, bg=style_bg)
        r4.pack(fill=tk.X, padx=6, pady=3)
        tk.Label(r4, text='Img Quality:', width=11, anchor='w',
                 font=('Segoe UI', 10, 'bold'),
                 bg=style_bg, fg=style_fg).pack(side=tk.LEFT)
        keys_q = ['Z', 'X', 'C']
        for i, (label, val) in enumerate(IMG_QUALITIES):
            tk.Radiobutton(r4, text=f'[{keys_q[i]}] {label}',
                           variable=self.sel_qual, value=val,
                           indicatoron=0, width=18, pady=4,
                           font=btn_font, selectcolor='#6a1b9a',
                           bg='#333', fg=style_fg,
                           activebackground='#444',
                           ).pack(side=tk.LEFT, padx=2)

        # --- nav buttons ---
        nav = tk.Frame(self.root, bg=style_bg)
        nav.pack(fill=tk.X, padx=12, pady=(4, 8))

        tk.Button(nav, text='← Prev  (Left)', command=self._prev,
                  font=btn_font, width=16, height=2).pack(side=tk.LEFT, padx=4)
        tk.Button(nav, text='Skip  (Right →)', command=self._skip,
                  font=btn_font, width=16, height=2).pack(side=tk.LEFT, padx=4)
        tk.Button(nav, text='No Face / Back  (B)', command=self._quick_back,
                  font=btn_font, width=20, height=2,
                  bg='#555', fg='white').pack(side=tk.LEFT, padx=4)
        tk.Button(nav, text='Save & Next  (Enter)', command=self._save_next,
                  font=('Segoe UI', 11, 'bold'), width=22, height=2,
                  bg='#2e7d32', fg='white',
                  activebackground='#388e3c').pack(side=tk.RIGHT, padx=4)

        # status
        self.lbl_status = tk.Label(self.root, text='', font=('Segoe UI', 9),
                                    bg=style_bg, fg='#888', anchor='w')
        self.lbl_status.pack(fill=tk.X, padx=12, pady=(0, 4))

    def _bind_keys(self):
        for widget in [self.root]:
            widget.bind('1', lambda e: self.sel_pose.set('frontal'))
            widget.bind('2', lambda e: self.sel_pose.set('three_quarter'))
            widget.bind('3', lambda e: self.sel_pose.set('profile'))
            widget.bind('4', lambda e: self.sel_pose.set('three_quarter_away'))
            widget.bind('5', lambda e: self.sel_pose.set('back'))

            widget.bind('q', lambda e: self.sel_dir.set('left'))
            widget.bind('w', lambda e: self.sel_dir.set('center'))
            widget.bind('e', lambda e: self.sel_dir.set('right'))

            widget.bind('a', lambda e: self.sel_occ.set('none'))
            widget.bind('s', lambda e: self.sel_occ.set('partial'))
            widget.bind('d', lambda e: self.sel_occ.set('heavy'))

            widget.bind('z', lambda e: self.sel_qual.set('clear'))
            widget.bind('x', lambda e: self.sel_qual.set('out_of_focus'))
            widget.bind('c', lambda e: self.sel_qual.set('mesh_haze'))

            widget.bind('<Return>', lambda e: self._save_next())
            widget.bind('<Left>',   lambda e: self._prev())
            widget.bind('<Right>',  lambda e: self._skip())
            widget.bind('b',        lambda e: self._quick_back())

    # ── display ─────────────────────────────────────────────
    def _update_counters(self):
        pose_counts = {}
        qual_counts = {}
        for a in self.saved.values():
            p = a['head_pose']
            pose_counts[p] = pose_counts.get(p, 0) + 1
            q = a.get('image_quality', 'clear')
            qual_counts[q] = qual_counts.get(q, 0) + 1
        for _, val in HEAD_POSES:
            n = pose_counts.get(val, 0)
            lbl = self.ctr_labels.get(f'pose_{val}')
            if lbl:
                color = '#4caf50' if n >= 40 else '#ff9800' if n >= 20 else '#aaa'
                lbl.config(text=f'{val}: {n}', fg=color)
        for _, val in IMG_QUALITIES:
            n = qual_counts.get(val, 0)
            lbl = self.ctr_labels.get(f'qual_{val}')
            if lbl:
                lbl.config(text=f'{val}: {n}')

    def _show(self):
        if not self.id_list or self.idx >= len(self.id_list):
            messagebox.showinfo('Done', 'No more photos in this view.')
            return

        self._update_counters()

        mid = self.id_list[self.idx]
        name = self.names.get(mid, '?')
        n_done = len(self.saved)

        self.lbl_progress.config(
            text=f'Done: {n_done}/{len(self.all_ids)}   |   '
                 f'View: {self.idx + 1}/{len(self.id_list)}')
        self.lbl_info.config(text=f'monkey_id: {mid}   name: {name}')

        # load image
        path = os.path.join(PHOTO_DIR, f'{mid}.JPG')
        if not os.path.exists(path):
            path = os.path.join(PHOTO_DIR, f'{mid}.jpg')
        try:
            img = Image.open(path).convert('RGB')
        except Exception as e:
            self.lbl_status.config(text=f'Error: {e}')
            return

        # compute scale after thumbnail
        orig_w, orig_h = img.size
        img.thumbnail(DISPLAY_MAX, Image.LANCZOS)
        sx = img.width / orig_w
        sy = img.height / orig_h

        # draw head bbox (green) on the resized image
        cd = self.coords.get(mid)
        if cd and cd['body'] and cd['face']:
            bx1, by1, bx2, by2 = cd['body']
            fx1, fy1, fx2, fy2 = cd['face']

            # offset to crop space, then scale to display
            dx1 = (fx1 - bx1) * sx
            dy1 = (fy1 - by1) * sy
            dx2 = (fx2 - bx1) * sx
            dy2 = (fy2 - by1) * sy

            # clamp
            dx1 = max(0, dx1)
            dy1 = max(0, dy1)
            dx2 = min(img.width, dx2)
            dy2 = min(img.height, dy2)

            draw = ImageDraw.Draw(img)
            for t in range(3):
                draw.rectangle([dx1 - t, dy1 - t, dx2 + t, dy2 + t],
                               outline='#00ff00')
            try:
                fnt = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 14)
            except Exception:
                fnt = ImageFont.load_default()
            draw.text((dx1 + 4, dy1 + 4), 'HEAD', fill='#00ff00', font=fnt)

        self.photo_tk = ImageTk.PhotoImage(img)
        self.canvas.delete('all')
        self.canvas.config(width=img.width, height=img.height)
        self.canvas.create_image(img.width // 2, img.height // 2,
                                 image=self.photo_tk)

        # pre-fill selections if already annotated (review mode)
        if mid in self.saved:
            a = self.saved[mid]
            self.sel_pose.set(a['head_pose'])
            self.sel_dir.set(a['direction'])
            self.sel_occ.set(a['occlusion'])
            self.sel_qual.set(a.get('image_quality', 'clear'))
            self.lbl_status.config(text='Already annotated — will overwrite on save')
        else:
            self.sel_pose.set('')
            self.sel_dir.set('')
            self.sel_occ.set('')
            self.sel_qual.set('clear')
            self.lbl_status.config(text='Keys: 1-5 pose  |  Q/W/E dir  |  '
                                        'A/S/D occl  |  Z/X/C quality  |  '
                                        'Enter save  |  B no-face')

    # ── actions ─────────────────────────────────────────────
    def _save(self, mid):
        pose = self.sel_pose.get()
        dirn = self.sel_dir.get()
        occ  = self.sel_occ.get()
        qual = self.sel_qual.get() or 'clear'

        if not pose:
            messagebox.showwarning('Missing', 'Select a head pose (keys 1-5)')
            return False
        if pose in ('back', 'three_quarter_away'):
            dirn = 'na'
        elif not dirn:
            messagebox.showwarning('Missing', 'Select a direction (keys Q/W/E)')
            return False
        if not occ:
            messagebox.showwarning('Missing', 'Select occlusion level (keys A/S/D)')
            return False

        self.cur.execute(f"""
            INSERT INTO `{TABLE_NAME}`
                (monkey_id, head_pose, face_direction, occlusion, image_quality)
            VALUES (%s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                head_pose      = VALUES(head_pose),
                face_direction = VALUES(face_direction),
                occlusion      = VALUES(occlusion),
                image_quality  = VALUES(image_quality),
                updated_at     = CURRENT_TIMESTAMP
        """, (mid, pose, dirn, occ, qual))
        self.db.commit()

        self.saved[mid] = {'head_pose': pose, 'direction': dirn,
                           'occlusion': occ, 'image_quality': qual}
        if mid in self.unannotated:
            self.unannotated.remove(mid)
        return True

    def _save_next(self):
        if not self.id_list:
            return
        mid = self.id_list[self.idx]
        if self._save(mid):
            self.lbl_status.config(text=f'Saved monkey_id {mid}')
            self._next()

    def _quick_back(self):
        self.sel_pose.set('back')
        self.sel_dir.set('na')
        self.sel_occ.set('none')
        self.sel_qual.set('clear')
        self._save_next()

    def _next(self):
        if self.idx < len(self.id_list) - 1:
            self.idx += 1
            self._show()
        else:
            messagebox.showinfo('End', 'Reached end of list.')

    def _prev(self):
        if self.idx > 0:
            self.idx -= 1
            self._show()

    def _skip(self):
        self._next()

    def _switch_mode(self):
        if self.mode_var.get() == 'unannotated':
            self.id_list = list(self.unannotated)
        else:
            self.id_list = list(self.all_ids)
        self.idx = 0
        if self.id_list:
            self._show()
        else:
            self.canvas.delete('all')
            self.lbl_status.config(text='No photos in this view')

    def _jump(self):
        try:
            target = int(self.jump_entry.get())
        except ValueError:
            return
        if target in self.id_list:
            self.idx = self.id_list.index(target)
            self._show()
        else:
            messagebox.showwarning('Not found',
                                   f'monkey_id {target} not in current list')

    def _on_close(self):
        try:
            self.cur.close()
            self.db.close()
        except Exception:
            pass
        self.root.destroy()


def main():
    root = tk.Tk()
    app = AnnotationTool(root)
    root.protocol('WM_DELETE_WINDOW', app._on_close)
    root.mainloop()


if __name__ == '__main__':
    main()
