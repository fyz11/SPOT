# -*- coding: utf-8 -*-
"""
SPOT GUI: run the full SPOT pipeline on a folder of organoid videos and browse the analysis, without writing code.

    python spot_gui.py [path/to/spot_project.json]

The heavy computation runs in a separate Python process (spot_pipeline.py), so the window stays responsive and a run
can be stopped at any time. All results are written to the chosen results folder; the project settings are saved as
spot_project.json in that folder and can be re-opened later to resume.
"""
import os
import sys
import re
import json
import queue
import threading
import subprocess
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from tkinter.scrolledtext import ScrolledText

import numpy as np

GUI_DIR = os.path.dirname(os.path.abspath(__file__))
if GUI_DIR not in sys.path:
    sys.path.insert(0, GUI_DIR)
import spot_pipeline as sp  # noqa: E402  (light weight: SPOT itself is only imported in the worker process)

APP_TITLE = 'SAM-SPOT - organoid phenotyping'
ASSETS_DIR = os.path.join(GUI_DIR, 'assets')
SETTINGS_FILE = os.path.join(os.path.expanduser('~'), '.spot_gui.json')

CHANNELS = ['auto (brightest)', '1 (red / gray)', '2 (green)', '3 (blue)']  # index - 1 = channel, -1 = auto
STATUS_SYMBOL = {'done': '✓', 'todo': '·', 'none': '0 org.', 'n/a': '', 'fail': '✗ failed',
                 'start': '▶ running', 'skip': '✓', 'stale': '⟳'}
SHORT_STEP = {'prepare': 'Prepare', 'detect': 'Detect', 'track': 'Track', 'segment': 'Segment',
              'postprocess': 'Clean up', 'features': 'SAM features', 'markers': 'Marker', 'compile': 'Compile',
              'analysis': 'Analysis'}
# colour themes: bg = window, panel = buttons / tabs, field = entries / tables, accent = selected tab and run buttons
THEMES_STANDARD = {'bg': '#f0f0f0', 'fg': '#000000', 'muted': '#555555', 'accent_dark': '#0b5394', 'tip': '#ffffe0',
                   'log_bg': '#ffffff', 'dataset_row': '#eef3fb'}
THEMES = {
    'Standard': None,
    'Pink Panther': {'bg': '#fbe3ee', 'panel': '#f6c6dc', 'field': '#fff6fa', 'fg': '#4a1530', 'muted': '#8a4a6a',
                     'accent': '#e0218a', 'accent_dark': '#a3125f', 'select': '#f06aa8', 'heading': '#efa3c7',
                     'border': '#d98ab0', 'tip': '#ffe9f3', 'log_bg': '#fff6fa', 'dataset_row': '#f9d2e4'},
    'Forest Green': {'bg': '#e7efe2', 'panel': '#cddfc2', 'field': '#f7fbf4', 'fg': '#1d3320', 'muted': '#4f6b4f',
                     'accent': '#2e6b30', 'accent_dark': '#1f4d22', 'select': '#4f8a3c', 'heading': '#b5cfa6',
                     'border': '#8fae84', 'tip': '#eef6e8', 'log_bg': '#f7fbf4', 'dataset_row': '#d7e8cc'},
}
TOOLTIP_BG = THEMES_STANDARD['tip']

INPUT_TYPES = [
    # value, label, help
    ('videos', 'Videos (SPOT finds, tracks and outlines the organoids)',
     'A folder of video files (.avi, .tif, ...). The pretrained networks detect and outline the organoids.'),
    ('ctc', 'Cell Tracking Challenge dataset (annotated cells)',
     'A dataset from celltrackingchallenge.net (or your tracker\'s results in the same format): the cells are '
     'outlined and tracked from its masks and track table.'),
]
CTC_SEGMENTATIONS = [
    ('auto', 'Best annotation (silver truth, gold truth, markers)'),
    ('tra', 'Gold truth tracking markers only'),
    ('res', 'My results (<sequence>_RES folder)'),
]
CTC_STEP_LABELS = {'detect': 'Detect (not needed: annotated data)', 'track': 'Track (not needed: annotated data)',
                   'segment': 'Segment (not needed: annotated data)',
                   'postprocess': 'Outline the annotated cells, remove cells at the border'}

GROUP_COLUMNS = {'Condition': 'Conditions', 'Genotype': 'Genetics', 'Patient / batch': 'Patients',
                 'Marker': 'Marker', 'Video': 'Filename'}
GROUP_LABELS = {v: k for k, v in GROUP_COLUMNS.items()}

ADVANCED_FIELDS = [
    # section, key, label, help
    ('detect', 'obj_thresh', 'Detection threshold', 'Minimum YOLOv3 score of a box (low: keep many candidates, the tracking filters them)'),
    ('detect', 'nms_thresh', 'Overlap suppression', 'Boxes overlapping more than this (IoU) are merged'),
    ('detect', 'equalize_hist', 'Equalise histogram', 'Improves detection in brightfield / phase contrast (1 = yes, 0 = no)'),
    ('track', 'iou_match', 'Track matching IoU', 'Minimum overlap to link a box to a track in the next frame'),
    ('track', 'wait_time', 'Gap frames', 'Number of frames a track may miss a detection before it ends'),
    ('track', 'min_aspect_ratio', 'Max aspect ratio', 'Boxes more elongated than this are ignored'),
    ('track', 'max_dense_bbox_cover', 'Max boxes covered', 'Boxes covering more boxes than this are ignored'),
    ('segment', 'segment_thresh', 'Segmentation threshold', '0-1: lower = larger outlines, higher = tighter outlines'),
    ('segment', 'min_I', 'Min. intensity (0-255)', 'Discard segmentations darker than this on average'),
    ('segment', 'filter_prob_thresh', 'Min. track score', 'Keep tracks whose mean detection score is at least this'),
    ('segment', 'filter_life_thresh', 'Min. track lifetime', 'Keep tracks present for at least this fraction of the time after they appear'),
    ('postprocess', 'border_pixel_size', 'Border width (pixels)', 'Organoids within this distance of the image edge are removed'),
    ('postprocess', 'border_pixel_frac', 'Border fraction', 'An organoid is at the border if more than this fraction of its outline is in the border'),
    ('postprocess', 'temporal_smooth_win_size', 'Smoothing window (frames)', 'Moving average of the outlines over time'),
    ('postprocess', 'min_split_track_len', 'Min. track length', 'Shortest track (frames) kept after splitting at shape jumps'),
    ('postprocess', 'iou_consistency_thresh', 'Shape consistency IoU', 'A track is split where successive outlines overlap less than this'),
    ('features', 'use_mask', 'Masked features', 'Appearance / motion from inside the organoid only (recommended, 1 = yes)'),
    ('markers', 'spot_threshold', 'Spot threshold', 'Minimum contrast of a marker spot (0-1)'),
    ('markers', 'spot_min_distance', 'Spot min. distance', 'Minimum distance between marker spots (pixels)'),
]


def load_settings():
    try:
        with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


def save_settings(d):
    try:
        with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
            json.dump(d, f, indent=2)
    except Exception:
        pass


def open_path(path):
    """ open a file / folder with the default application """
    try:
        if sys.platform.startswith('win'):
            os.startfile(path)
        elif sys.platform == 'darwin':
            subprocess.Popen(['open', path])
        else:
            subprocess.Popen(['xdg-open', path])
    except Exception as e:
        messagebox.showerror(APP_TITLE, 'Could not open %s\n%s' % (path, e))


def guess_sample_info(name, paired, ctc=False):
    """ guess (condition, marker) from a video name such as 'C10 6 uM Cisplatin Cytotox 2':
    drops a leading well id (C10) and a trailing replicate number (2); with paired marker videos the last word is the
    marker (Cytotox). Cell Tracking Challenge sequences ('PhC-C2DH-U373-01'): the dataset name """
    if ctc:
        return re.sub(r'-\d+$', '', name), ''
    toks = name.split()
    if len(toks) > 1 and re.match(r'^[A-Pa-p]\d{1,2}$', toks[0]):
        toks = toks[1:]
    if len(toks) > 1 and re.match(r'^\d+$', toks[-1]):
        toks = toks[:-1]
    marker = ''
    if paired and len(toks) > 1:
        marker = toks[-1]
        toks = toks[:-1]
    return ' '.join(toks), marker


class ToolTip(object):
    def __init__(self, widget, text):
        self.widget, self.text, self.tip = widget, text, None
        widget.bind('<Enter>', self.show)
        widget.bind('<Leave>', self.hide)

    def show(self, _=None):
        if self.tip or not self.text:
            return
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        self.tip = tk.Toplevel(self.widget)
        self.tip.wm_overrideredirect(True)
        self.tip.wm_geometry('+%d+%d' % (x, y))
        tk.Label(self.tip, text=self.text, justify='left', background=TOOLTIP_BG, relief='solid', borderwidth=1,
                 wraplength=380, font=('Segoe UI', 9)).pack(ipadx=4, ipady=2)

    def hide(self, _=None):
        if self.tip:
            self.tip.destroy()
            self.tip = None


class SpotApp(tk.Tk):

    def __init__(self, project=None):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry('1320x860')
        self.minsize(1050, 700)
        self.cfg = sp.merge_config({})
        self.videos = []
        self.proc = None
        self.msg_queue = queue.Queue()
        self.run_status = {}
        self.current_steps = []
        self.settings = load_settings()
        self.qc_cache = {}
        self.result_image = None

        self._style()
        self._build_header()
        self.nb = ttk.Notebook(self)
        self.nb.pack(fill='both', expand=True, padx=8, pady=(0, 8))
        self._build_setup_tab()
        self._build_sample_tab()
        self._build_run_tab()
        self._build_qc_tab()
        self._build_analysis_tab()
        self._apply_theme(self.theme)   # again, for the plain tk widgets created with the tabs
        self.nb.bind('<<NotebookTabChanged>>', self._on_tab_changed)
        self.protocol('WM_DELETE_WINDOW', self._on_close)
        self.after(100, self._poll_queue)

        project = project or self.settings.get('last_project')
        if project and os.path.exists(project):
            self.open_project(project)
        else:
            self.apply_cfg()

    # ------------------------------------------------------------------ look
    def _style(self):
        name = self.settings.get('theme', 'Standard')
        self.theme = name if name in THEMES else 'Standard'
        self._apply_theme(self.theme)

    def _apply_theme(self, name):
        """ 'Standard' = the system look; the other themes recolour every widget (ttk 'clam' engine) """
        global TOOLTIP_BG
        style = ttk.Style(self)
        t = THEMES.get(name)
        if t is None:
            if 'vista' in style.theme_names():
                style.theme_use('vista')
            elif 'clam' in style.theme_names():
                style.theme_use('clam')
            t = dict(THEMES_STANDARD)
            t['bg'] = style.lookup('TFrame', 'background') or t['bg']
            self.option_add('*TCombobox*Listbox.background', 'white')
            self.option_add('*TCombobox*Listbox.foreground', 'black')
            self.option_add('*TCombobox*Listbox.selectBackground', '#0078d7')
        else:
            style.theme_use('clam')
            style.configure('.', background=t['bg'], foreground=t['fg'], fieldbackground=t['field'],
                            bordercolor=t['border'], lightcolor=t['panel'], darkcolor=t['border'],
                            troughcolor=t['panel'], selectbackground=t['select'], selectforeground='white',
                            insertcolor=t['fg'], arrowcolor=t['fg'])
            style.configure('TButton', background=t['panel'], padding=(8, 3))
            style.map('TButton', background=[('disabled', t['bg']), ('pressed', t['accent_dark']), ('active', t['heading'])],
                      foreground=[('disabled', t['muted']), ('pressed', 'white')])
            style.configure('TNotebook', background=t['bg'], tabmargins=(2, 4, 2, 0))
            style.configure('TNotebook.Tab', background=t['panel'], foreground=t['fg'], padding=(10, 4))
            style.map('TNotebook.Tab', background=[('selected', t['accent'])], foreground=[('selected', 'white')])
            style.configure('TLabelframe', background=t['bg'], bordercolor=t['border'])
            style.configure('TLabelframe.Label', background=t['bg'], foreground=t['accent_dark'],
                            font=('Segoe UI', 9, 'bold'))
            style.configure('Treeview', background=t['field'], fieldbackground=t['field'], foreground=t['fg'])
            style.map('Treeview', background=[('selected', t['select'])], foreground=[('selected', 'white')])
            style.configure('Treeview.Heading', background=t['heading'], foreground=t['fg'], relief='flat')
            style.map('Treeview.Heading', background=[('active', t['panel'])])
            for w in ('TCheckbutton', 'TRadiobutton'):
                style.configure(w, background=t['bg'], indicatorbackground=t['field'], indicatorforeground='white')
                style.map(w, background=[('active', t['bg'])],
                          indicatorbackground=[('selected', t['accent']), ('disabled', t['bg'])],
                          foreground=[('disabled', t['muted'])])
            style.map('TCombobox', fieldbackground=[('readonly', t['field']), ('disabled', t['bg'])],
                      foreground=[('disabled', t['muted'])], background=[('active', t['heading'])])
            style.configure('TEntry', fieldbackground=t['field'])
            style.configure('TSpinbox', fieldbackground=t['field'], background=t['panel'])
            style.configure('Horizontal.TProgressbar', background=t['accent'], troughcolor=t['panel'])
            style.configure('TScrollbar', background=t['panel'], troughcolor=t['bg'])
            style.configure('TPanedwindow', background=t['bg'])
            style.configure('Big.TButton', background=t['accent'], foreground='white')
            style.map('Big.TButton', background=[('disabled', t['panel']), ('pressed', t['accent_dark']),
                                                 ('active', t['accent_dark'])],
                      foreground=[('disabled', t['muted'])])
            self.option_add('*TCombobox*Listbox.background', t['field'])
            self.option_add('*TCombobox*Listbox.foreground', t['fg'])
            self.option_add('*TCombobox*Listbox.selectBackground', t['select'])
        style.configure('Header.TLabel', font=('Segoe UI', 15, 'bold'))
        style.configure('Sub.TLabel', foreground=t['muted'])
        style.configure('Help.TLabel', foreground=t['muted'], font=('Segoe UI', 8))
        style.configure('Stage.TLabel', font=('Segoe UI', 10, 'bold'), foreground=t['accent_dark'])
        style.configure('Big.TButton', font=('Segoe UI', 10, 'bold'), padding=6)
        style.configure('Treeview', rowheight=22)
        TOOLTIP_BG = t['tip']
        self.configure(background=t['bg'])
        # plain tk widgets are not styled by ttk
        if hasattr(self, 'log_text'):
            self.log_text.configure(background=t['log_bg'], foreground=t['fg'], insertbackground=t['fg'])
            self.log_text.tag_configure('step', foreground=t['accent_dark'])
        if hasattr(self, 'status_tree'):
            self.status_tree.tag_configure('dataset', background=t['dataset_row'])
        if hasattr(self, 'qc_fig'):
            self.qc_fig.set_facecolor([c / 65535. for c in self.winfo_rgb(t['bg'])])   # also system colour names
            self.qc_canvas.get_tk_widget().configure(background=t['bg'])
            self.qc_canvas.draw_idle()
        self.theme = name

    def set_theme(self, name):
        self._apply_theme(name)
        self.settings['theme'] = name
        save_settings(self.settings)

    def _set_icon(self):
        try:
            if sys.platform.startswith('win') and os.path.exists(os.path.join(ASSETS_DIR, 'sam_spot_icon.ico')):
                self.iconbitmap(default=os.path.join(ASSETS_DIR, 'sam_spot_icon.ico'))
            self._icon = tk.PhotoImage(file=os.path.join(ASSETS_DIR, 'sam_spot_icon.png'))
            self.iconphoto(True, self._icon)
        except Exception:
            pass

    def _build_header(self):
        self._set_icon()
        bar = ttk.Frame(self, padding=(10, 6))
        bar.pack(fill='x')
        self._logo = None
        try:
            from PIL import Image, ImageTk
            img = Image.open(os.path.join(ASSETS_DIR, 'sam_spot_logo_transparent.png'))
            h = 64
            img = img.resize((int(img.width * h / float(img.height)), h), Image.LANCZOS)
            self._logo = ImageTk.PhotoImage(img)
            ttk.Label(bar, image=self._logo).pack(side='left')
        except Exception:
            ttk.Label(bar, text='SAM-SPOT', style='Header.TLabel').pack(side='left')
        ttk.Label(bar, text='  Shape, Appearance and Motion\n  phenotyping of organoid and cell videos', style='Sub.TLabel').pack(side='left')
        right = ttk.Frame(bar)
        right.pack(side='right')
        bf = ttk.Frame(right)
        bf.pack(anchor='e')
        ttk.Button(bf, text='New project', command=self.new_project).pack(side='left', padx=2)
        ttk.Button(bf, text='Open project...', command=self.ask_open_project).pack(side='left', padx=2)
        b = ttk.Button(bf, text='Load settings...', command=self.ask_load_settings)
        b.pack(side='left', padx=2)
        ToolTip(b, 'Re-use the settings of another project, of a previous run (run_history/<time>/run_config.json) '
                   'or of an analysis (analysis/<name>/analysis_config.json), to reproduce it.')
        ttk.Button(bf, text='Save project', command=self.save_project).pack(side='left', padx=2)
        ttk.Label(bf, text='   Theme').pack(side='left')
        self.theme_var = tk.StringVar(value=self.theme)
        cb = ttk.Combobox(bf, textvariable=self.theme_var, values=list(THEMES), state='readonly', width=13)
        cb.pack(side='left', padx=(4, 0))
        cb.bind('<<ComboboxSelected>>', lambda _: self.set_theme(self.theme_var.get()))
        ToolTip(cb, 'Colours of the window. Remembered for the next time.')
        self.project_label = ttk.Label(right, text='', style='Help.TLabel')
        self.project_label.pack(anchor='e', pady=(4, 0))

    # ------------------------------------------------------------------ tab 1: set up
    def _build_setup_tab(self):
        tab = ttk.Frame(self.nb, padding=10)
        self.nb.add(tab, text='  1. Set up  ')
        left = ttk.Frame(tab)
        left.pack(side='left', fill='y')
        right = ttk.Frame(tab)
        right.pack(side='left', fill='both', expand=True, padx=(12, 0))

        self.v = {}

        def var(key, kind=tk.StringVar):
            self.v[key] = kind()
            return self.v[key]

        def row(parent, r, label, widget, help_text=None):
            lab = ttk.Label(parent, text=label)
            lab.grid(row=r, column=0, sticky='w', pady=3, padx=(0, 6))
            widget.grid(row=r, column=1, sticky='we', pady=3)
            if help_text:
                ToolTip(lab, help_text)
                ToolTip(widget, help_text)

        f = ttk.LabelFrame(right, text='What to analyse', padding=8)
        f.pack(fill='x', pady=(0, 8))
        for val, txt, help_text in INPUT_TYPES:
            rb = ttk.Radiobutton(f, text=txt, value=val, variable=var('input_type'),
                                 command=lambda: (self._on_input_type_changed(), self.scan_videos()))
            rb.pack(anchor='w')
            ToolTip(rb, help_text)

        f = ttk.LabelFrame(left, text='Folders', padding=8)
        f.pack(fill='x')
        f.columnconfigure(1, weight=1)
        self.folder_labels = {}
        for r, (key, label, help_text) in enumerate([
                ('input_folder', 'Videos folder', 'Folder containing the videos to analyse'),
                ('output_folder', 'Results folder', 'Every result is written here (can be a new, empty folder)')]):
            fr = ttk.Frame(f)
            ttk.Entry(fr, textvariable=var(key), width=42).pack(side='left', fill='x', expand=True)
            ttk.Button(fr, text='Browse...', command=lambda k=key: self._browse_folder(k)).pack(side='left', padx=(4, 0))
            row(f, r, label, fr, help_text)
            self.folder_labels[key] = f.grid_slaves(row=r, column=0)[0]
        row(f, 2, 'Dataset name', ttk.Entry(f, textvariable=var('dataset_name')),
            'Short name used for the compiled tables (default: name of the results folder)')

        # Cell Tracking Challenge options (shown instead of the video options)
        self.ctc_frame = f = ttk.LabelFrame(left, text='Cell Tracking Challenge', padding=8)
        f.columnconfigure(1, weight=1)
        row(f, 0, 'Cell outlines from', ttk.Combobox(f, textvariable=var('ctc_segmentation_label'),
                                                    values=[t for _, t in CTC_SEGMENTATIONS], state='readonly', width=34),
            'Which masks give the cell outlines. The tracks (cell identities over time) come from the gold truth '
            'tracking (man_track.txt), or from your results (res_track.txt).')
        row(f, 1, 'Min. track length (frames)', ttk.Spinbox(f, textvariable=var('ctc_min_track_len'), from_=1, to=1000, width=8),
            'Shorter tracks (after removing cells at the image border) are not analysed')
        ttk.Label(f, text='Choose the dataset folder (e.g. PhC-C2DH-U373, with 01, 01_GT, 01_ST, ...) or a folder of '
                          'several datasets. Each sequence (01, 02, ...) is analysed like a video; detection, tracking '
                          'and segmentation by the networks are not needed.',
                  wraplength=420, style='Help.TLabel').grid(row=2, column=0, columnspan=2, sticky='w', pady=(4, 0))

        self.videos_frame = f = ttk.LabelFrame(left, text='Videos', padding=8)
        f.pack(fill='x', pady=(8, 0))
        f.columnconfigure(1, weight=1)
        row(f, 0, 'File types', ttk.Entry(f, textvariable=var('video_extensions')),
            'Video file extensions to look for, separated by spaces')
        self.paired_check = ttk.Checkbutton(f, text='Each brightfield video has a matching fluorescent marker video',
                                            variable=var('paired_marker', tk.BooleanVar), command=self._on_paired_changed)
        self.paired_check.grid(row=1, column=0, columnspan=2, sticky='w', pady=3)
        ToolTip(self.paired_check, 'e.g. "B5 DMSO Caspase 1 BF.AVI" (brightfield) and "B5 DMSO Caspase 1.AVI" (marker). '
                                   'Organoids are found in the brightfield video; the marker is measured inside them.')
        self.bf_suffix_entry = ttk.Entry(f, textvariable=var('bf_suffix'), width=10)
        row(f, 2, 'Brightfield name ends with', self.bf_suffix_entry,
            'Text at the end of the brightfield file names (before the extension), e.g. " BF"')
        row(f, 3, 'Channel (colour videos)', ttk.Combobox(f, textvariable=var('channel_label'), values=CHANNELS,
                                                         state='readonly', width=16),
            'Channel used if the brightfield video is saved in colour (gray videos: any)')
        self.marker_channel_combo = ttk.Combobox(f, textvariable=var('marker_channel_label'), values=CHANNELS,
                                                 state='readonly', width=16)
        row(f, 4, 'Marker channel', self.marker_channel_combo, 'Channel of the marker video to measure')

        self.imaging_frame = f = ttk.LabelFrame(left, text='Imaging', padding=8)
        f.pack(fill='x', pady=(8, 0))
        f.columnconfigure(1, weight=1)
        row(f, 0, 'Pixel size (µm / pixel)', ttk.Entry(f, textvariable=var('pixel_size_um'), width=10),
            'Pixel size of the video files as they are (before any resizing by SPOT)')
        row(f, 1, 'Time between frames (hours)', ttk.Entry(f, textvariable=var('time_interval_h'), width=10))
        self.video_only_checks = []
        c = ttk.Checkbutton(f, text='Resize frames to 512 x 512 (recommended)', variable=var('resize', tk.BooleanVar))
        c.grid(row=2, column=0, columnspan=2, sticky='w')
        ToolTip(c, 'The detection and segmentation networks were trained on 512 x 512 images; the pixel size is corrected automatically.')
        self.video_only_checks.append(c)
        c = ttk.Checkbutton(f, text='Correct stage drift (register frames)', variable=var('register', tk.BooleanVar))
        c.grid(row=3, column=0, columnspan=2, sticky='w')
        self.video_only_checks.append(c)
        c = ttk.Checkbutton(f, text='Organoids are darker than the background (brightfield / phase contrast)',
                            variable=var('invert_intensity', tk.BooleanVar))
        c.grid(row=4, column=0, columnspan=2, sticky='w')
        ToolTip(c, 'Untick for fluorescence videos where organoids are bright.')
        self.video_only_checks.append(c)

        self.computer_frame = f = ttk.LabelFrame(left, text='Computer', padding=8)
        f.pack(fill='x', pady=(8, 0))
        f.columnconfigure(1, weight=1)
        row(f, 0, 'Device for the networks', ttk.Combobox(f, textvariable=var('device'), values=['auto', 'cuda', 'cpu'],
                                                          state='readonly', width=10),
            'auto = GPU (cuda) if available, else CPU')
        row(f, 1, 'CPU cores for features', ttk.Spinbox(f, textvariable=var('n_jobs'), from_=-1, to=128, width=8),
            '-1 = all cores')

        # video list
        top = ttk.Frame(right)
        top.pack(fill='x')
        ttk.Label(top, text='Videos to analyse', style='Stage.TLabel').pack(side='left')
        ttk.Button(top, text='Find videos', command=self.scan_videos).pack(side='left', padx=10)
        ttk.Button(top, text='Select all', command=lambda: self._select_all_videos(True)).pack(side='right')
        ttk.Button(top, text='Select none', command=lambda: self._select_all_videos(False)).pack(side='right', padx=4)
        ttk.Label(right, text='Click a row\'s "Use" box to include / exclude it. Tip: try 2-3 videos first, check the '
                              'segmentation, then run all.', style='Help.TLabel').pack(anchor='w', pady=(2, 4))
        tf = ttk.Frame(right)
        tf.pack(fill='both', expand=True)
        cols = ('use', 'name', 'bf', 'marker')
        self.video_tree = ttk.Treeview(tf, columns=cols, show='headings', selectmode='extended')
        for c, t, w in zip(cols, ['Use', 'Video', 'Brightfield / video file', 'Marker file'], [50, 240, 300, 260]):
            self.video_tree.heading(c, text=t)
            self.video_tree.column(c, width=w, stretch=(c != 'use'), anchor='center' if c == 'use' else 'w')
        sb = ttk.Scrollbar(tf, orient='vertical', command=self.video_tree.yview)
        self.video_tree.configure(yscrollcommand=sb.set)
        self.video_tree.pack(side='left', fill='both', expand=True)
        sb.pack(side='left', fill='y')
        self.video_tree.bind('<Button-1>', self._on_video_click)
        bf = ttk.Frame(right)
        bf.pack(fill='x', pady=(6, 0))
        self.video_count_label = ttk.Label(bf, text='', style='Sub.TLabel')
        self.video_count_label.pack(side='left')
        ttk.Button(bf, text='Save project', command=self.save_project).pack(side='right')
        ttk.Button(bf, text='Advanced settings...', command=self.advanced_dialog).pack(side='right', padx=4)

    def _browse_folder(self, key):
        d = filedialog.askdirectory(initialdir=self.v[key].get() or os.path.expanduser('~'))
        if d:
            self.v[key].set(os.path.normpath(d))
            if key == 'input_folder':
                self.scan_videos()
            if key == 'output_folder':
                pf = os.path.join(d, 'spot_project.json')
                if os.path.exists(pf) and messagebox.askyesno(APP_TITLE, 'This folder already contains a SPOT project. Open it?'):
                    self.open_project(pf)

    def _on_input_type_changed(self):
        ctc = self.v['input_type'].get() == 'ctc'
        if ctc:
            self.videos_frame.pack_forget()
            self.ctc_frame.pack(fill='x', pady=(8, 0), before=self.imaging_frame)
        else:
            self.ctc_frame.pack_forget()
            self.videos_frame.pack(fill='x', pady=(8, 0), before=self.imaging_frame)
        for c in self.video_only_checks:
            c.configure(state='disabled' if ctc else 'normal')
        self.folder_labels['input_folder'].configure(text='Dataset folder' if ctc else 'Videos folder')
        self.video_tree.heading('bf', text='Sequence folder' if ctc else 'Brightfield / video file')
        self.video_tree.heading('marker', text='Track table' if ctc else 'Marker file')
        for key, c in getattr(self, 'step_checks', {}).items():
            c.configure(text=CTC_STEP_LABELS.get(key, sp.STEP_LABELS[key]) if ctc else sp.STEP_LABELS[key],
                        state='disabled' if (ctc and key in sp.CTC_SKIPPED_STEPS) else 'normal')
        if hasattr(self, 'status_tree'):
            self.status_tree.heading('postprocess', text='Outlines' if ctc else SHORT_STEP['postprocess'])

    def _on_paired_changed(self):
        state = 'normal' if self.v['paired_marker'].get() else 'disabled'
        self.bf_suffix_entry.configure(state=state)
        self.marker_channel_combo.configure(state='readonly' if state == 'normal' else 'disabled')

    def _on_video_click(self, event):
        if self.video_tree.identify_region(event.x, event.y) != 'cell':
            return
        if self.video_tree.identify_column(event.x) != '#1':
            return
        item = self.video_tree.identify_row(event.y)
        if not item:
            return
        vals = list(self.video_tree.item(item, 'values'))
        vals[0] = '' if vals[0] == '☑' else '☑'
        self.video_tree.item(item, values=vals)
        self._update_selection_from_tree()
        return 'break'

    def _select_all_videos(self, on):
        for item in self.video_tree.get_children():
            vals = list(self.video_tree.item(item, 'values'))
            vals[0] = '☑' if on else ''
            self.video_tree.item(item, values=vals)
        self._update_selection_from_tree()

    def _update_selection_from_tree(self):
        sel = [self.video_tree.item(i, 'values')[1] for i in self.video_tree.get_children()
               if self.video_tree.item(i, 'values')[0] == '☑']
        all_names = [v['name'] for v in self.videos]
        self.cfg['selected_videos'] = [] if len(sel) == len(all_names) else sel
        self.video_count_label.configure(text='%d of %d videos selected' % (len(sel), len(all_names)))
        self.refresh_status()

    def scan_videos(self):
        self.collect_cfg()
        self.videos = sp.discover_videos(self.cfg)
        sel = set(self.cfg.get('selected_videos') or [])
        self.video_tree.delete(*self.video_tree.get_children())
        ctc = sp.is_ctc(self.cfg)
        for v in self.videos:
            use = '☑' if (not sel or v['name'] in sel) else ''
            if ctc:
                src = v['bf_file']
                marker = (os.path.relpath(v['ctc_tracks'], os.path.dirname(src)) if v.get('ctc_tracks')
                          else '(no track table!)')
                src = os.path.join(os.path.basename(os.path.dirname(src)), os.path.basename(src))
            else:
                src = os.path.basename(v['bf_file'])
                marker = os.path.basename(v['marker_file']) if v.get('marker_file') else ('(missing!)' if self.cfg.get('paired_marker') else '')
            self.video_tree.insert('', 'end', iid=v['name'], values=(use, v['name'], src, marker))
        if not self.videos and self.cfg.get('input_folder'):
            msg = 'No videos found.'
            if ctc:
                msg = ('No Cell Tracking Challenge sequences found: choose the dataset folder that contains the '
                       'sequence folders 01, 02, ... (with frames t000.tif, ...).')
            elif self.cfg.get('paired_marker'):
                msg += ' Check that the brightfield file names end with "%s".' % self.cfg.get('bf_suffix')
            self.video_count_label.configure(text=msg)
        else:
            self._update_selection_from_tree()
        self.sync_sample_rows()
        self.refresh_status()
        self.qc_video_combo.configure(values=[v['name'] for v in self.videos])

    def advanced_dialog(self):
        self.collect_cfg()
        dlg = tk.Toplevel(self)
        dlg.title('Advanced settings')
        dlg.transient(self)
        dlg.grab_set()
        nb = ttk.Notebook(dlg)
        nb.pack(fill='both', expand=True, padx=8, pady=8)
        entries = {}
        titles = {'detect': 'Detection', 'track': 'Tracking', 'segment': 'Segmentation', 'postprocess': 'Clean up',
                  'features': 'SAM features', 'markers': 'Marker'}
        frames = {}
        for sec, key, label, help_text in ADVANCED_FIELDS:
            if sec not in frames:
                frames[sec] = ttk.Frame(nb, padding=10)
                nb.add(frames[sec], text=titles[sec])
                ttk.Label(frames[sec], text=sp.STEP_HELP.get(sec if sec != 'features' else 'features', ''),
                          wraplength=460, style='Help.TLabel').grid(row=0, column=0, columnspan=2, sticky='w', pady=(0, 8))
            fr = frames[sec]
            r = len(fr.grid_slaves(column=0))
            val = self.cfg[sec].get(key)
            if isinstance(val, bool):
                val = int(val)
            e = ttk.Entry(fr, width=12)
            e.insert(0, str(val))
            lab = ttk.Label(fr, text=label)
            lab.grid(row=r, column=0, sticky='w', pady=3, padx=(0, 8))
            e.grid(row=r, column=1, sticky='w', pady=3)
            ToolTip(lab, help_text)
            ToolTip(e, help_text)
            entries[(sec, key)] = e
        wf = ttk.Frame(nb, padding=10)
        nb.add(wf, text='Networks')
        ttk.Label(wf, text='Leave empty to use the pretrained SPOT networks (from the repository models folder, or '
                           'downloaded automatically from Zenodo on first use).', wraplength=460,
                  style='Help.TLabel').grid(row=0, column=0, columnspan=3, sticky='w', pady=(0, 8))
        wvars = {}
        for r, (key, label) in enumerate([('detector_weights', 'Detector weights (.pt)'),
                                          ('segmenter_weights', 'Segmenter weights (.pt)')]):
            wvars[key] = tk.StringVar(value=self.cfg.get(key, ''))
            ttk.Label(wf, text=label).grid(row=r + 1, column=0, sticky='w')
            ttk.Entry(wf, textvariable=wvars[key], width=40).grid(row=r + 1, column=1, sticky='we', pady=3)
            ttk.Button(wf, text='...', width=3,
                       command=lambda k=key: wvars[k].set(filedialog.askopenfilename(filetypes=[('PyTorch weights', '*.pt')]) or wvars[k].get())
                       ).grid(row=r + 1, column=2)

        def ok():
            try:
                for (sec, key), e in entries.items():
                    default = sp.DEFAULT_CONFIG[sec][key]
                    txt = e.get().strip()
                    if isinstance(default, bool):
                        self.cfg[sec][key] = txt.lower() in ('1', 'true', 'yes', 'y')
                    elif isinstance(default, int):
                        self.cfg[sec][key] = int(float(txt))
                    else:
                        self.cfg[sec][key] = float(txt)
            except ValueError:
                messagebox.showerror(APP_TITLE, 'Please enter numbers only.', parent=dlg)
                return
            for k, wv in wvars.items():
                self.cfg[k] = wv.get().strip()
            dlg.destroy()

        def reset():
            for (sec, key), e in entries.items():
                e.delete(0, 'end')
                val = sp.DEFAULT_CONFIG[sec][key]
                e.insert(0, str(int(val) if isinstance(val, bool) else val))

        bf = ttk.Frame(dlg, padding=(8, 0, 8, 8))
        bf.pack(fill='x')
        ttk.Button(bf, text='Reset to defaults', command=reset).pack(side='left')
        ttk.Button(bf, text='Cancel', command=dlg.destroy).pack(side='right')
        ttk.Button(bf, text='OK', command=ok).pack(side='right', padx=4)

    # ------------------------------------------------------------------ tab 2: sample info
    def _build_sample_tab(self):
        tab = ttk.Frame(self.nb, padding=10)
        self.nb.add(tab, text='  2. Sample info  ')
        ttk.Label(tab, text='Describe each video. The Condition (e.g. drug treatment) is used to compare groups in the '
                            'analysis; Genotype, Patient / batch and Marker are optional. Double-click a cell to edit it, '
                            'or select several rows and use "Set value for selected rows".',
                  wraplength=1150, style='Sub.TLabel').pack(anchor='w')
        bf = ttk.Frame(tab)
        bf.pack(fill='x', pady=6)
        ttk.Button(bf, text='Guess from video names', command=self.guess_samples).pack(side='left')
        ttk.Button(bf, text='Set value for selected rows...', command=self.set_selected_samples).pack(side='left', padx=4)
        ttk.Button(bf, text='Import table (.csv)...', command=self.import_samples).pack(side='left')
        ttk.Button(bf, text='Save sample table', style='Big.TButton', command=self.save_samples).pack(side='right')
        self.sample_status = ttk.Label(tab, text='', style='Sub.TLabel')
        self.sample_status.pack(anchor='w')
        tf = ttk.Frame(tab)
        tf.pack(fill='both', expand=True, pady=(4, 0))
        self.sample_cols = ('Filename', 'Conditions', 'Genetics', 'Patients', 'Marker')
        titles = ['Video', 'Condition', 'Genotype', 'Patient / batch', 'Marker']
        self.sample_tree = ttk.Treeview(tf, columns=self.sample_cols, show='headings', selectmode='extended')
        for c, t, w in zip(self.sample_cols, titles, [330, 250, 160, 160, 140]):
            self.sample_tree.heading(c, text=t)
            self.sample_tree.column(c, width=w, anchor='w')
        sb = ttk.Scrollbar(tf, orient='vertical', command=self.sample_tree.yview)
        self.sample_tree.configure(yscrollcommand=sb.set)
        self.sample_tree.pack(side='left', fill='both', expand=True)
        sb.pack(side='left', fill='y')
        self.sample_tree.bind('<Double-1>', self._edit_sample_cell)
        self.sample_dirty = False

    def sync_sample_rows(self):
        """ rows = discovered videos, values from the saved table where available """
        existing = {}
        for item in self.sample_tree.get_children():
            vals = self.sample_tree.item(item, 'values')
            existing[vals[0]] = list(vals)
        mf = sp.metadata_file(self.cfg) if self.cfg.get('output_folder') else ''
        if mf and os.path.exists(mf) and not existing:
            try:
                import pandas as pd
                t = pd.read_csv(mf, encoding='utf-8')
                for _, r in t.iterrows():
                    existing[str(r['Filename'])] = [str(r['Filename'])] + [
                        '' if (c not in t.columns or pd.isna(r[c]) or str(r[c]) == 'NA') else str(r[c])
                        for c in self.sample_cols[1:]]
                self.sample_status.configure(text='Loaded %s' % mf)
            except Exception as e:
                self.sample_status.configure(text='Could not read %s: %s' % (mf, e))
        self.sample_tree.delete(*self.sample_tree.get_children())
        for v in self.videos:
            vals = existing.get(v['name'], [v['name'], '', '', '', ''])
            self.sample_tree.insert('', 'end', iid=v['name'], values=vals)

    def _edit_sample_cell(self, event):
        if self.sample_tree.identify_region(event.x, event.y) != 'cell':
            return
        item = self.sample_tree.identify_row(event.y)
        col = int(self.sample_tree.identify_column(event.x)[1:]) - 1
        if not item or col == 0:
            return
        x, y, w, h = self.sample_tree.bbox(item, '#%d' % (col + 1))
        vals = list(self.sample_tree.item(item, 'values'))
        known = sorted({self.sample_tree.item(i, 'values')[col] for i in self.sample_tree.get_children()} - {''})
        e = ttk.Combobox(self.sample_tree, values=known)
        e.insert(0, vals[col])
        e.place(x=x, y=y, width=w, height=h)
        e.focus_set()
        e.select_range(0, 'end')

        def commit(_=None):
            vals[col] = e.get().strip()
            self.sample_tree.item(item, values=vals)
            self.sample_dirty = True
            self.sample_status.configure(text='Unsaved changes')
            e.destroy()
        e.bind('<Return>', commit)
        e.bind('<FocusOut>', commit)
        e.bind('<Escape>', lambda _: e.destroy())

    def guess_samples(self):
        paired = bool(self.v['paired_marker'].get())
        n = 0
        ctc = self.v['input_type'].get() == 'ctc'
        for item in self.sample_tree.get_children():
            vals = list(self.sample_tree.item(item, 'values'))
            cond, marker = guess_sample_info(vals[0], paired and not ctc, ctc=ctc)
            if not vals[1]:
                vals[1] = cond
                n += 1
            if paired and not vals[4]:
                vals[4] = marker
            self.sample_tree.item(item, values=vals)
        self.sample_dirty = True
        self.sample_status.configure(text='Filled %d empty conditions from the video names - please check them, then save.' % n)

    def set_selected_samples(self):
        items = self.sample_tree.selection()
        if not items:
            messagebox.showinfo(APP_TITLE, 'Select one or more rows first (Ctrl / Shift + click).')
            return
        dlg = tk.Toplevel(self)
        dlg.title('Set value')
        dlg.transient(self)
        dlg.grab_set()
        ttk.Label(dlg, text='Set for %d selected videos:' % len(items)).grid(row=0, column=0, columnspan=2, padx=10, pady=8)
        col_var = tk.StringVar(value='Condition')
        ttk.Combobox(dlg, textvariable=col_var, values=['Condition', 'Genotype', 'Patient / batch', 'Marker'],
                     state='readonly', width=16).grid(row=1, column=0, padx=10)
        val = ttk.Entry(dlg, width=30)
        val.grid(row=1, column=1, padx=10)
        val.focus_set()

        def ok(_=None):
            col = self.sample_cols.index(GROUP_COLUMNS[col_var.get()])
            for it in items:
                vals = list(self.sample_tree.item(it, 'values'))
                vals[col] = val.get().strip()
                self.sample_tree.item(it, values=vals)
            self.sample_dirty = True
            self.sample_status.configure(text='Unsaved changes')
            dlg.destroy()
        val.bind('<Return>', ok)
        ttk.Button(dlg, text='OK', command=ok).grid(row=2, column=1, sticky='e', padx=10, pady=8)

    def import_samples(self, f=None):
        f = f or filedialog.askopenfilename(filetypes=[('CSV table', '*.csv')])
        if not f:
            return
        try:
            import pandas as pd
            t = pd.read_csv(f, encoding='utf-8')
            t = t.set_index(t.columns[0])
            alias = {'Condition': 'Conditions', 'Genotype': 'Genetics', 'Patient': 'Patients'}
            t = t.rename(columns=alias)
            n = 0
            for item in self.sample_tree.get_children():
                vals = list(self.sample_tree.item(item, 'values'))
                if vals[0] in t.index:
                    for jj, c in enumerate(self.sample_cols[1:], start=1):
                        if c in t.columns and not pd.isna(t.loc[vals[0], c]):
                            vals[jj] = str(t.loc[vals[0], c])
                    n += 1
                    self.sample_tree.item(item, values=vals)
            self.sample_dirty = True
            self.sample_status.configure(text='Imported information for %d videos (unsaved).' % n)
        except Exception as e:
            messagebox.showerror(APP_TITLE, 'Could not import the table:\n%s' % e)

    def save_samples(self, quiet=False):
        self.collect_cfg()
        if not self.cfg.get('output_folder'):
            messagebox.showerror(APP_TITLE, 'Choose a results folder on the "Set up" tab first.')
            return False
        import pandas as pd
        rows = []
        for item in self.sample_tree.get_children():
            vals = self.sample_tree.item(item, 'values')
            info = sp.read_info(self.cfg, vals[0])
            rows.append({'Filename': vals[0], 'Img_Channel': 1,
                         'Genetics': vals[2] or 'NA', 'Patients': vals[3] or 'NA', 'Conditions': vals[1] or 'NA',
                         'Marker': vals[4] or 'NA',
                         'pixel_resolution[um]': info['pixel_res_um'] if info else '',
                         'time_resolution[h]': info['time_res_h'] if info else self.cfg.get('time_interval_h'),
                         'n_frames': info['n_frames'] if info else ''})
        os.makedirs(self.cfg['output_folder'], exist_ok=True)
        pd.DataFrame(rows, columns=sp.METADATA_COLUMNS).to_csv(sp.metadata_file(self.cfg), index=None, encoding='utf-8')
        self.sample_dirty = False
        self.sample_status.configure(text='Saved %s' % sp.metadata_file(self.cfg))
        self._refresh_group_choices()
        missing = [r['Filename'] for r in rows if r['Conditions'] == 'NA']
        if missing and not quiet:
            messagebox.showwarning(APP_TITLE, '%d videos have no condition. They will be grouped as "NA" in the analysis.' % len(missing))
        return True

    # ------------------------------------------------------------------ tab 3: run
    def _build_run_tab(self):
        tab = ttk.Frame(self.nb, padding=10)
        self.nb.add(tab, text='  3. Run pipeline  ')
        pan = ttk.PanedWindow(tab, orient='horizontal')
        pan.pack(fill='both', expand=True)
        left = ttk.Frame(pan, padding=(0, 0, 10, 0))
        right = ttk.Frame(pan)
        pan.add(left, weight=0)
        pan.add(right, weight=1)

        self.step_vars = {}
        self.step_checks = {}
        stage_titles = {1: 'Stage 1 - find, track and outline organoids / cells', 2: 'Stage 2 - SAM phenome',
                        3: 'Stage 3 - analysis'}
        last_stage = None
        for key, label, stage, _ in sp.STEPS:
            if stage != last_stage:
                ttk.Label(left, text=stage_titles[stage], style='Stage.TLabel').pack(anchor='w', pady=(10 if last_stage else 0, 2))
                last_stage = stage
            self.step_vars[key] = tk.BooleanVar(value=True)
            c = ttk.Checkbutton(left, text=label, variable=self.step_vars[key])
            c.pack(anchor='w', padx=(8, 0))
            self.step_checks[key] = c
            ToolTip(c, sp.STEP_HELP[key])
        bf = ttk.Frame(left)
        bf.pack(anchor='w', pady=8, padx=(8, 0))
        ttk.Button(bf, text='All', width=6, command=lambda: [v.set(True) for v in self.step_vars.values()]).pack(side='left')
        ttk.Button(bf, text='None', width=6, command=lambda: [v.set(False) for v in self.step_vars.values()]).pack(side='left', padx=4)
        ttk.Button(bf, text='Stage 1 only', command=lambda: [v.set(k in sp.STEP_KEYS[:5]) for k, v in self.step_vars.items()]).pack(side='left')
        self.skip_var = tk.BooleanVar(value=True)
        c = ttk.Checkbutton(left, text='Skip steps already completed (resume)', variable=self.skip_var)
        c.pack(anchor='w', pady=(4, 0))
        ToolTip(c, 'Untick to recompute everything, e.g. after changing settings.')
        self.run_button = ttk.Button(left, text='▶  Run selected steps', style='Big.TButton', command=self.run_selected)
        self.run_button.pack(fill='x', pady=(14, 4))
        self.stop_button = ttk.Button(left, text='■  Stop', command=self.stop_run, state='disabled')
        self.stop_button.pack(fill='x')
        ttk.Label(left, text='The analysis settings are on tab 5.', style='Help.TLabel').pack(anchor='w', pady=8)

        lf = ttk.Frame(right)
        lf.pack(fill='x')
        b = ttk.Button(lf, text='Run history...', command=self.open_run_history)
        b.pack(side='right')
        ToolTip(b, 'Every run is recorded in <results folder>/run_history/<date-time>/: the complete settings '
                   '(run_config.json, can be re-loaded with "Load settings..."), the sample table, software and '
                   'network versions, and the log.')
        ttk.Label(lf, text='✓ done   · to do   0 org. = no organoids   ⟳ settings changed (redone '
                           'next run)   ✗ failed', style='Sub.TLabel').pack(side='left')
        tf = ttk.Frame(right)
        tf.pack(fill='both', expand=True, pady=(2, 6))
        cols = ['video'] + sp.STEP_KEYS
        self.status_tree = ttk.Treeview(tf, columns=cols, show='headings', height=10)
        self.status_tree.heading('video', text='Video')
        self.status_tree.column('video', width=230, anchor='w')
        for k in sp.STEP_KEYS:
            self.status_tree.heading(k, text=SHORT_STEP[k])
            self.status_tree.column(k, width=78, anchor='center', stretch=False)
        sb = ttk.Scrollbar(tf, orient='vertical', command=self.status_tree.yview)
        self.status_tree.configure(yscrollcommand=sb.set)
        self.status_tree.pack(side='left', fill='both', expand=True)
        sb.pack(side='left', fill='y')
        self.status_tree.tag_configure('dataset', background='#eef3fb')

        pf = ttk.Frame(right)
        pf.pack(fill='x')
        self.progress_label = ttk.Label(pf, text='Ready.')
        self.progress_label.pack(side='left')
        self.progress = ttk.Progressbar(pf, mode='determinate', length=300)
        self.progress.pack(side='right')
        ttk.Label(right, text='Log', style='Sub.TLabel').pack(anchor='w', pady=(6, 0))
        self.log_text = ScrolledText(right, height=14, font=('Consolas', 9), wrap='none', state='disabled')
        self.log_text.pack(fill='both', expand=True)
        self.log_text.tag_configure('error', foreground='#b00020')
        self.log_text.tag_configure('step', foreground='#0b5394', font=('Consolas', 9, 'bold'))

    def log(self, text, tag=None):
        self.log_text.configure(state='normal')
        self.log_text.insert('end', text + '\n', tag)
        if int(self.log_text.index('end-1c').split('.')[0]) > 6000:
            self.log_text.delete('1.0', '1000.0')
        self.log_text.see('end')
        self.log_text.configure(state='disabled')

    def refresh_status(self):
        if not hasattr(self, 'status_tree'):
            return
        self.status_tree.delete(*self.status_tree.get_children())
        if not self.cfg.get('output_folder'):
            return
        for v in sp.selected_videos(self.cfg) if self.videos else []:
            vals = [v['name']]
            for k in sp.STEP_KEYS:
                if k in sp.PER_VIDEO_STEPS:
                    st = self.run_status.get((k, v['name'])) or sp.step_status(self.cfg, v, k)
                    vals.append(STATUS_SYMBOL.get(st, st))
                else:
                    vals.append('')
            self.status_tree.insert('', 'end', iid='v:' + v['name'], values=vals)
        vals = ['Whole dataset'] + [''] * len(sp.PER_VIDEO_STEPS)
        for k in ('compile', 'analysis'):
            st = self.run_status.get((k, '*')) or sp.dataset_step_status(self.cfg, k)
            vals.append(STATUS_SYMBOL.get(st, st))
        self.status_tree.insert('', 'end', iid='dataset', values=vals, tags=('dataset',))

    def _set_cell(self, step, video, status):
        self.run_status[(step, video)] = None if status in ('done', 'skip', 'none') else status
        iid = 'dataset' if video == '*' else 'v:' + video
        if self.status_tree.exists(iid):
            self.status_tree.set(iid, step, STATUS_SYMBOL.get(status, status))
            if status == 'start':
                self.status_tree.see(iid)

    def run_selected(self):
        steps = [k for k in sp.STEP_KEYS if self.step_vars[k].get()]
        if not steps:
            messagebox.showinfo(APP_TITLE, 'Tick at least one step.')
            return
        self.start_run(steps)

    def validate(self, steps):
        self.collect_cfg()
        c = self.cfg
        if not c['input_folder'] or not os.path.isdir(c['input_folder']):
            return 'Choose the %s folder on the "Set up" tab.' % ('dataset' if sp.is_ctc(c) else 'videos')
        if not c['output_folder']:
            return 'Choose a results folder on the "Set up" tab.'
        if not self.videos:
            self.scan_videos()
        if not sp.selected_videos(c):
            return 'No videos selected on the "Set up" tab.'
        if sp.is_ctc(c) and 'postprocess' in steps:
            missing = [v['name'] for v in sp.selected_videos(c) if not v.get('ctc_tracks')]
            if missing:
                return ('No track table for %s. Cell Tracking Challenge sequences need %s.'
                        % (', '.join(missing), '<sequence>_RES/res_track.txt' if c['ctc'].get('segmentation') == 'res'
                           else '<sequence>_GT/TRA/man_track.txt'))
        try:
            assert float(c['pixel_size_um']) > 0 and float(c['time_interval_h']) > 0
        except Exception:
            return 'Enter a valid pixel size and time between frames on the "Set up" tab.'
        if c.get('paired_marker') and any(not v.get('marker_file') for v in sp.selected_videos(c)):
            if not messagebox.askyesno(APP_TITLE, 'Some brightfield videos have no matching marker video. Continue '
                                                  '(their marker will not be measured)?'):
                return ''
        if 'compile' in steps or 'analysis' in steps:
            if self.sample_dirty or not os.path.exists(sp.metadata_file(c)):
                if not self.save_samples(quiet=True):
                    return ''
        return None

    def start_run(self, steps):
        if self.proc is not None:
            messagebox.showinfo(APP_TITLE, 'A run is already in progress.')
            return
        err = self.validate(steps)
        if err is not None:
            if err:
                messagebox.showerror(APP_TITLE, err)
            return
        self.cfg['skip_done'] = bool(self.skip_var.get())
        self.save_project(quiet=True)
        self.run_status = {}
        self.current_steps = steps
        self.refresh_status()
        self.nb.select(2)
        cmd = [sys.executable, '-u', os.path.join(GUI_DIR, 'spot_pipeline.py'), '--config', sp.project_file(self.cfg),
               '--steps', ','.join(steps)]
        env = dict(os.environ, MPLBACKEND='Agg', PYTHONIOENCODING='utf-8', TQDM_DISABLE='1', PYTHONUNBUFFERED='1')
        flags = subprocess.CREATE_NO_WINDOW if sys.platform.startswith('win') else 0
        self.log('=' * 70)
        self.log('Starting: ' + ', '.join(SHORT_STEP[s] for s in steps), 'step')
        try:
            self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                         env=env, cwd=self.cfg['output_folder'], creationflags=flags,
                                         encoding='utf-8', errors='replace', bufsize=1)
        except Exception as e:
            self.proc = None
            messagebox.showerror(APP_TITLE, 'Could not start the pipeline:\n%s' % e)
            return
        threading.Thread(target=self._reader, args=(self.proc,), daemon=True).start()
        self.run_button.configure(state='disabled')
        self.analysis_run_button.configure(state='disabled')
        self.stop_button.configure(state='normal')
        self.progress_label.configure(text='Starting...')

    def _reader(self, proc):
        for line in proc.stdout:
            self.msg_queue.put(line.rstrip('\n'))
        proc.wait()
        self.msg_queue.put(('__EXIT__', proc.returncode))

    def _poll_queue(self):
        try:
            for _ in range(500):
                msg = self.msg_queue.get_nowait()
                if isinstance(msg, tuple):
                    self._on_run_finished(msg[1])
                else:
                    self._handle_line(msg)
        except queue.Empty:
            pass
        self.after(100, self._poll_queue)

    def _handle_line(self, line):
        if line.startswith('@@'):
            kind, _, rest = line[2:].partition(' ')
            args = rest.split('\t')
            if kind == 'STEP':
                step, video, status = args
                self._set_cell(step, video, status)
                if status == 'start':
                    self.progress_label.configure(text='%s: %s' % (SHORT_STEP.get(step, step), video if video != '*' else 'whole dataset'))
                elif status == 'fail':
                    self.log('FAILED: %s - %s' % (SHORT_STEP.get(step, step), video), 'error')
            elif kind == 'PROGRESS':
                i, n = int(args[0]), max(int(args[1]), 1)
                self.progress.configure(maximum=n, value=i)
            elif kind == 'STAGE':
                self.progress_label.configure(text=sp.STEP_LABELS.get(args[0], args[0]))
            elif kind == 'FINISHED':
                pass
            return
        tag = 'error' if ('Error' in line or 'Traceback' in line or line.startswith('WARNING')) else None
        if line.startswith('['):
            tag = 'step'
        self.log(line, tag)

    def _on_run_finished(self, code):
        self.proc = None
        self.run_button.configure(state='normal')
        self.analysis_run_button.configure(state='normal')
        self.stop_button.configure(state='disabled')
        failed = any(v == 'fail' for v in self.run_status.values())
        if getattr(self, '_stopped', False):
            text = 'Stopped.'
        elif failed or code:
            text = 'Finished with errors - see the log (red lines).'
        else:
            text = 'Finished.'
        self._stopped = False
        self.progress_label.configure(text=text)
        self.log(text, 'step' if text == 'Finished.' else 'error')
        self.refresh_status()
        self.qc_cache.clear()
        if 'analysis' in self.current_steps:
            self.refresh_results()
            if text == 'Finished.':
                self.nb.select(4)
        self.bell()

    def stop_run(self):
        if self.proc is None:
            return
        if not messagebox.askyesno(APP_TITLE, 'Stop the run? The step running now will be lost, finished steps are kept.'):
            return
        self._stopped = True
        try:
            if sys.platform.startswith('win'):
                subprocess.call(['taskkill', '/F', '/T', '/PID', str(self.proc.pid)],
                                creationflags=subprocess.CREATE_NO_WINDOW,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                self.proc.terminate()
        except Exception:
            pass
        self.log('Stopped by user.', 'error')

    # ------------------------------------------------------------------ tab 4: QC viewer
    def _build_qc_tab(self):
        tab = ttk.Frame(self.nb, padding=10)
        self.nb.add(tab, text='  4. Check segmentation  ')
        ctrl = ttk.Frame(tab)
        ctrl.pack(fill='x')
        ttk.Label(ctrl, text='Video').pack(side='left')
        self.qc_video = tk.StringVar()
        self.qc_video_combo = ttk.Combobox(ctrl, textvariable=self.qc_video, state='readonly', width=40)
        self.qc_video_combo.pack(side='left', padx=6)
        self.qc_video_combo.bind('<<ComboboxSelected>>', lambda _: self.qc_load())
        self.qc_layer = tk.StringVar(value='final')
        for val, txt in [('final', 'Final tracks'), ('raw', 'Raw segmentation'), ('boxes', 'Detected boxes')]:
            ttk.Radiobutton(ctrl, text=txt, value=val, variable=self.qc_layer, command=self.qc_draw).pack(side='left', padx=4)
        self.qc_marker = tk.BooleanVar(value=False)
        ttk.Checkbutton(ctrl, text='Show marker channel', variable=self.qc_marker, command=self.qc_draw).pack(side='left', padx=8)
        self.qc_ids = tk.BooleanVar(value=True)
        ttk.Checkbutton(ctrl, text='Show IDs', variable=self.qc_ids, command=self.qc_draw).pack(side='left')
        fr = ttk.Frame(tab)
        fr.pack(fill='x', pady=6)
        ttk.Button(fr, text='◀', width=3, command=lambda: self.qc_step(-1)).pack(side='left')
        self.qc_play_button = ttk.Button(fr, text='Play', width=6, command=self.qc_toggle_play)
        self.qc_play_button.pack(side='left', padx=2)
        ttk.Button(fr, text='▶', width=3, command=lambda: self.qc_step(1)).pack(side='left')
        self.qc_frame = tk.IntVar(value=0)
        self.qc_scale = ttk.Scale(fr, from_=0, to=1, orient='horizontal', command=self._qc_scale_moved)
        self.qc_scale.pack(side='left', fill='x', expand=True, padx=8)
        self.qc_info = ttk.Label(fr, text='', width=48)
        self.qc_info.pack(side='left')
        self.qc_playing = False

        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        self.qc_fig = Figure(figsize=(7, 7), dpi=100)
        self.qc_ax = self.qc_fig.add_axes([0, 0, 1, 1])
        self.qc_ax.axis('off')
        self.qc_canvas = FigureCanvasTkAgg(self.qc_fig, master=tab)
        self.qc_canvas.get_tk_widget().pack(fill='both', expand=True)
        ttk.Label(tab, text='Outlines are coloured by organoid track. Check that organoids are found and outlined well; '
                            'if not, change the settings (tab 1, Advanced settings) and re-run with "Skip steps already '
                            'completed" unticked.', style='Help.TLabel').pack(anchor='w')

    def qc_load(self):
        name = self.qc_video.get()
        if not name or not self.cfg.get('output_folder'):
            return
        p = sp.paths(self.cfg, name)
        if not os.path.exists(p['vid']):
            self.qc_data = None
            self.qc_ax.clear()
            self.qc_ax.axis('off')
            self.qc_ax.text(0.5, 0.5, 'This video has not been prepared yet (run Stage 1).', ha='center', va='center')
            self.qc_canvas.draw_idle()
            return
        key = (name, os.path.getmtime(p['vid']))
        if key not in self.qc_cache:
            import tifffile
            import scipy.io as spio
            d = {'vid': np.asarray(tifffile.imread(p['vid']))}
            if os.path.exists(p['marker_vid']):
                d['marker'] = np.asarray(tifffile.imread(p['marker_vid']))
            d['final'] = None
            d['raw'] = None
            if os.path.exists(p['final']):
                try:
                    d['final'] = sp.load_final_boundaries(self.cfg, name)
                except Exception:
                    pass
            if os.path.exists(p['seg']):
                try:
                    b = spio.loadmat(p['seg'])['boundaries']
                    d['raw'] = b.astype(np.float64) if b.ndim == 4 else None
                except Exception:
                    pass
            elif sp.is_ctc(self.cfg) and os.path.exists(p['final']):
                try:
                    # Cell Tracking Challenge data: every annotated outline, before removing cells at the border
                    b = spio.loadmat(p['final'])['boundaries_raw']
                    d['raw'] = b[0].astype(np.float64) if b.ndim == 5 else None
                except Exception:
                    pass
            d['boxes_folder'] = p['bbox_ch_folder']
            self.qc_cache = {key: d}
        self.qc_data = self.qc_cache[key]
        n = len(self.qc_data['vid'])
        self.qc_scale.configure(to=max(n - 1, 0))
        self.qc_frame.set(min(self.qc_frame.get(), n - 1))
        self.qc_draw()

    def _qc_scale_moved(self, val):
        f = int(round(float(val)))
        if f != self.qc_frame.get():
            self.qc_frame.set(f)
            self.qc_draw()

    def qc_step(self, d):
        if not getattr(self, 'qc_data', None):
            return
        n = len(self.qc_data['vid'])
        self.qc_frame.set((self.qc_frame.get() + d) % n)
        self.qc_scale.set(self.qc_frame.get())
        self.qc_draw()

    def qc_toggle_play(self):
        self.qc_playing = not self.qc_playing
        self.qc_play_button.configure(text='Pause' if self.qc_playing else 'Play')
        if self.qc_playing:
            self._qc_play_tick()

    def _qc_play_tick(self):
        if self.qc_playing:
            self.qc_step(1)
            self.after(300, self._qc_play_tick)

    def qc_draw(self):
        d = getattr(self, 'qc_data', None)
        if not d:
            return
        from matplotlib import colormaps
        from matplotlib.patches import Rectangle
        t = self.qc_frame.get()
        ax = self.qc_ax
        ax.clear()
        ax.axis('off')
        img = d['marker'][t] if (self.qc_marker.get() and 'marker' in d) else d['vid'][t]
        ax.imshow(img, cmap='magma' if (self.qc_marker.get() and 'marker' in d) else 'gray')
        layer = self.qc_layer.get()
        info = ''
        if layer in ('final', 'raw'):
            b = d.get(layer)
            if b is None or len(b) == 0:
                info = 'no %s outlines yet' % ('final' if layer == 'final' else 'raw')
            else:
                colors = colormaps['tab20']
                n_here = 0
                for ii in range(len(b)):
                    if t < b.shape[1] and not np.isnan(b[ii, t, 0, 0]):
                        c = b[ii, t]
                        ax.plot(c[:, 1], c[:, 0], '-', lw=2, color=colors(ii % 20))
                        if self.qc_ids.get():
                            ax.text(np.nanmean(c[:, 1]), np.nanmean(c[:, 0]), str(ii + 1), color='yellow',
                                    fontsize=9, ha='center', va='center', weight='bold')
                        n_here += 1
                info = '%d %s tracks, %d in this frame' % (len(b), 'cell' if sp.is_ctc(self.cfg) else 'organoid', n_here)
        else:
            f = os.path.join(d['boxes_folder'], 'bbox_predictions_%s_Ch-01.txt' % str(t).zfill(4))
            n_here = 0
            if os.path.exists(f):
                with open(f) as fh:
                    for line in fh:
                        parts = line.split()
                        if len(parts) == 6:
                            score = float(parts[1])
                            x1, y1, x2, y2 = [float(v) for v in parts[2:]]
                            ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, lw=1,
                                                   ec=(0, 1, 0, min(1, 0.2 + score))))
                            n_here += 1
                info = '%d candidate boxes in this frame' % n_here
            else:
                info = 'not used for annotated (Cell Tracking Challenge) data' if sp.is_ctc(self.cfg) else 'no detections yet'
        ax.set_xlim(0, img.shape[1] - 1)
        ax.set_ylim(img.shape[0] - 1, 0)
        self.qc_info.configure(text='Frame %d / %d   %s' % (t + 1, len(d['vid']), info))
        self.qc_canvas.draw_idle()

    # ------------------------------------------------------------------ tab 5: analysis
    def _build_analysis_tab(self):
        tab = ttk.Frame(self.nb, padding=10)
        self.nb.add(tab, text='  5. Analysis & results  ')
        pan = ttk.PanedWindow(tab, orient='horizontal')
        pan.pack(fill='both', expand=True)
        left = ttk.Frame(pan, padding=(0, 0, 10, 0))
        right = ttk.Frame(pan)
        pan.add(left, weight=0)
        pan.add(right, weight=1)

        self.a = {}

        def avar(key, kind=tk.StringVar):
            self.a[key] = kind()
            return self.a[key]

        def row(parent, r, label, widget, help_text=None):
            lab = ttk.Label(parent, text=label)
            lab.grid(row=r, column=0, sticky='w', pady=2, padx=(0, 6))
            widget.grid(row=r, column=1, sticky='we', pady=2)
            if help_text:
                ToolTip(lab, help_text)
                ToolTip(widget, help_text)

        f = ttk.LabelFrame(left, text='Compare', padding=8)
        f.pack(fill='x')
        row(f, 0, 'Analysis name', ttk.Entry(f, textvariable=avar('name'), width=22),
            'Results go to <results folder>/analysis/<name>; use a new name to keep an earlier analysis')
        self.group_combo = ttk.Combobox(f, textvariable=avar('group_label'), state='readonly', width=20)
        row(f, 1, 'Group by', self.group_combo, 'Sample information column compared in the plots')
        self.batch_combo = ttk.Combobox(f, textvariable=avar('batch_label'), state='readonly', width=20)
        row(f, 2, 'Batch (optional)', self.batch_combo,
            'e.g. plate or experiment. Its effect is regressed out of the features before the map is computed.')
        c = ttk.Checkbutton(f, text='Normalise each batch separately', variable=avar('normalise_per_batch', tk.BooleanVar))
        c.grid(row=3, column=0, columnspan=2, sticky='w')
        self.marker_group_combo = ttk.Combobox(f, textvariable=avar('marker_group_label'), state='readonly', width=20)
        row(f, 4, 'Split marker plots by', self.marker_group_combo,
            'Used when different videos were stained with different markers')

        f = ttk.LabelFrame(left, text='Phenotype map and clusters', padding=8)
        f.pack(fill='x', pady=(8, 0))
        row(f, 0, 'Map method', ttk.Combobox(f, textvariable=avar('embedding'), values=['UMAP', 'PaCMAP'],
                                            state='readonly', width=10))
        row(f, 1, 'UMAP neighbours', ttk.Spinbox(f, textvariable=avar('n_neighbors'), from_=5, to=500, width=8),
            'Larger = more global structure (default 100)')
        row(f, 2, 'Number of clusters', ttk.Combobox(f, textvariable=avar('n_clusters'),
                                                    values=['auto'] + [str(k) for k in range(2, 21)], width=8),
            'auto = chosen automatically between 2 and the maximum below')
        row(f, 3, 'Max. clusters (auto)', ttk.Spinbox(f, textvariable=avar('k_max'), from_=3, to=40, width=8))
        row(f, 4, 'SAM modules', ttk.Combobox(f, textvariable=avar('module_scale'),
                                             values=['best', 'coarse', 'fine'] + [str(k) for k in range(3, 21)], width=8),
            'best = automatic multi-scale choice; coarse / fine = fewer / more modules; or a number')
        row(f, 5, 'Time bins', ttk.Spinbox(f, textvariable=avar('n_time_bins'), from_=2, to=60, width=8),
            'Number of time windows for the trajectories and time course')

        f = ttk.LabelFrame(left, text='Options', padding=8)
        f.pack(fill='x', pady=(8, 0))
        for r, (key, txt, help_text) in enumerate([
                ('equal_weighting', 'Weight shape, appearance and motion equally', 'Recommended (SPOT 2.0)'),
                ('time_varying_selection', 'Keep only features that change over time', 'As in the SPOT papers, for time-lapse data'),
                ('hmm', 'Phenotype transitions (HMM)', 'Fits a hidden Markov model to the cluster sequence of each organoid track'),
                ('save_vector', 'Also save figures as .svg (for publication)', None)]):
            c = ttk.Checkbutton(f, text=txt, variable=avar(key, tk.BooleanVar))
            c.grid(row=r, column=0, sticky='w')
            if help_text:
                ToolTip(c, help_text)

        self.analysis_run_button = ttk.Button(left, text='▶  Run analysis', style='Big.TButton', command=self.run_analysis)
        self.analysis_run_button.pack(fill='x', pady=(12, 2))
        ttk.Label(left, text='Needs Stages 1-2 (tab 3). Compiles the dataset first if needed.',
                  style='Help.TLabel').pack(anchor='w')

        # results browser
        top = ttk.Frame(right)
        top.pack(fill='x')
        ttk.Label(top, text='Results', style='Stage.TLabel').pack(side='left')
        ttk.Button(top, text='Open results folder', command=self.open_results_folder).pack(side='right')
        ttk.Button(top, text='Refresh', command=self.refresh_results).pack(side='right', padx=4)
        self.summary_label = ttk.Label(right, text='', style='Sub.TLabel', wraplength=820, justify='left')
        self.summary_label.pack(anchor='w', pady=(2, 4))
        rp = ttk.PanedWindow(right, orient='horizontal')
        rp.pack(fill='both', expand=True)
        tf = ttk.Frame(rp)
        self.result_tree = ttk.Treeview(tf, show='tree', selectmode='browse')
        sb = ttk.Scrollbar(tf, orient='vertical', command=self.result_tree.yview)
        self.result_tree.configure(yscrollcommand=sb.set)
        self.result_tree.pack(side='left', fill='both', expand=True)
        sb.pack(side='left', fill='y')
        self.result_tree.bind('<<TreeviewSelect>>', self._on_result_select)
        self.result_tree.bind('<Double-1>', self._on_result_open)
        rp.add(tf, weight=0)
        self.image_frame = ttk.Frame(rp)
        rp.add(self.image_frame, weight=1)
        self.image_label = ttk.Label(self.image_frame, anchor='center',
                                     text='Run the analysis, then click a figure on the left.\n'
                                          'Double-click opens it (or a .csv table) in its own program.')
        self.image_label.pack(fill='both', expand=True)
        self.image_label.bind('<Configure>', lambda _: self._show_result_image())
        self.result_path = None

    def _refresh_group_choices(self):
        cols = ['Condition', 'Genotype', 'Patient / batch', 'Marker', 'Video']
        self.group_combo.configure(values=cols)
        self.batch_combo.configure(values=['(none)'] + cols[:4])
        self.marker_group_combo.configure(values=cols)

    def run_analysis(self):
        steps = ['analysis']
        self.collect_cfg()
        if self.cfg.get('output_folder') and sp.dataset_step_status(self.cfg, 'compile') != 'done':
            steps = ['compile', 'analysis']
        elif self.sample_dirty:
            # sample info changed: recompile so the tables carry the new conditions
            steps = ['compile', 'analysis']
        prev_skip = self.skip_var.get()
        if 'compile' in steps:
            self.skip_var.set(False)
        self.start_run(steps)
        self.skip_var.set(prev_skip)

    def open_results_folder(self):
        self.collect_cfg()
        d = sp.analysis_folder(self.cfg) if self.cfg.get('output_folder') else ''
        if d and os.path.isdir(d):
            open_path(d)
        elif self.cfg.get('output_folder') and os.path.isdir(self.cfg['output_folder']):
            open_path(self.cfg['output_folder'])

    def refresh_results(self):
        self.collect_cfg()
        self.result_tree.delete(*self.result_tree.get_children())
        if not self.cfg.get('output_folder'):
            return
        d = sp.analysis_folder(self.cfg)
        summ = os.path.join(d, 'summary.json')
        if os.path.exists(summ):
            try:
                with open(summ, 'r', encoding='utf-8') as f:
                    s = json.load(f)
                txt = ('%s: %d %s instances from %d videos, %d SAM features, %s SAM modules, %d phenotype clusters. '
                       'Groups (%s): %s.' % (os.path.basename(d), s.get('n_objects', 0),
                                             'cell' if sp.is_ctc(self.cfg) else 'organoid', s.get('n_videos', 0),
                                             s.get('n_features', 0), s.get('n_modules', '?'), s.get('n_clusters', 0),
                                             GROUP_LABELS.get(s.get('group_column'), s.get('group_column')),
                                             ', '.join(s.get('groups', []))))
                if s.get('warnings'):
                    txt += '\nNot completed: ' + '; '.join(w.split(':')[0] for w in s['warnings'])
                self.summary_label.configure(text=txt)
            except Exception:
                pass
        else:
            self.summary_label.configure(text='No analysis named "%s" yet.' % os.path.basename(d))
        if not os.path.isdir(d):
            return
        first = None
        for f in sorted(os.listdir(d)):
            p = os.path.join(d, f)
            if os.path.isfile(p) and f.endswith('.csv'):
                self.result_tree.insert('', 'end', iid=p, text='\U0001F4C4 ' + f)
        for sub in sorted(os.listdir(d)):
            sp_ = os.path.join(d, sub)
            if not os.path.isdir(sp_):
                continue
            files = sorted(os.listdir(sp_))
            if not files:
                continue
            label = sub.split('_', 1)[-1].replace('_', ' ') if sub[:2].isdigit() else sub
            label = label[:1].upper() + label[1:]
            node = self.result_tree.insert('', 'end', iid=sp_, text=label, open=(first is None))
            for f in files:
                p = os.path.join(sp_, f)
                if f.lower().endswith('.png'):
                    self.result_tree.insert(node, 'end', iid=p, text=os.path.splitext(f)[0].replace('_', ' '))
                    if first is None:
                        first = p
                elif f.lower().endswith('.csv'):
                    self.result_tree.insert(node, 'end', iid=p, text='\U0001F4C4 ' + f)
        if first:
            self.result_tree.selection_set(first)
            self.result_tree.see(first)

    def _on_result_select(self, _=None):
        sel = self.result_tree.selection()
        if sel and sel[0].lower().endswith('.png'):
            self.result_path = sel[0]
            self._show_result_image()

    def _on_result_open(self, _=None):
        sel = self.result_tree.selection()
        if sel and os.path.isfile(sel[0]):
            open_path(sel[0])

    def _show_result_image(self):
        if not self.result_path or not os.path.exists(self.result_path):
            return
        try:
            from PIL import Image, ImageTk
            img = Image.open(self.result_path)
            w = max(self.image_label.winfo_width() - 10, 100)
            h = max(self.image_label.winfo_height() - 10, 100)
            s = min(w / float(img.width), h / float(img.height), 1.5)
            img = img.convert('RGB').resize((max(1, int(img.width * s)), max(1, int(img.height * s))), Image.LANCZOS)
            self.result_image = ImageTk.PhotoImage(img)
            self.image_label.configure(image=self.result_image, text='')
        except Exception as e:
            self.image_label.configure(image='', text='Cannot show %s\n%s' % (self.result_path, e))

    # ------------------------------------------------------------------ config <-> widgets
    def _show_project(self, text, path=''):
        if path and len(path) > 70:
            path = '...' + path[-67:]
        self.project_label.configure(text=text + path)

    def apply_cfg(self):
        c = self.cfg
        for key in ('input_folder', 'output_folder', 'dataset_name', 'video_extensions', 'bf_suffix', 'device'):
            self.v[key].set(c.get(key, ''))
        self.v['input_type'].set(c.get('input_type') or 'videos')
        seg = dict(CTC_SEGMENTATIONS)
        self.v['ctc_segmentation_label'].set(seg.get(c['ctc'].get('segmentation'), CTC_SEGMENTATIONS[0][1]))
        self.v['ctc_min_track_len'].set(str(c['ctc'].get('min_track_len', 5)))
        for key in ('paired_marker', 'resize', 'register', 'invert_intensity'):
            self.v[key].set(bool(c.get(key)))
        self.v['pixel_size_um'].set(str(c.get('pixel_size_um')))
        self.v['time_interval_h'].set(str(c.get('time_interval_h')))
        self.v['n_jobs'].set(str(c.get('n_jobs')))
        self.v['channel_label'].set(CHANNELS[max(-1, min(int(c.get('channel', 0)), 2)) + 1])
        self.v['marker_channel_label'].set(CHANNELS[max(-1, min(int(c.get('marker_channel', -1)), 2)) + 1])
        self.skip_var.set(bool(c.get('skip_done', True)))
        a = c['analysis']
        for key in ('name', 'embedding'):
            self.a[key].set(a.get(key))
        for key in ('n_neighbors', 'n_clusters', 'k_max', 'module_scale', 'n_time_bins'):
            self.a[key].set(str(a.get(key)))
        for key in ('normalise_per_batch', 'equal_weighting', 'time_varying_selection', 'hmm', 'save_vector'):
            self.a[key].set(bool(a.get(key)))
        self.a['group_label'].set(GROUP_LABELS.get(a.get('group_column'), 'Condition'))
        self.a['batch_label'].set(GROUP_LABELS.get(a.get('batch_column'), '(none)') if a.get('batch_column') else '(none)')
        self.a['marker_group_label'].set(GROUP_LABELS.get(a.get('marker_group_column'), 'Marker'))
        self._refresh_group_choices()
        self._on_paired_changed()
        self._on_input_type_changed()
        self._show_project('Project: ', sp.project_file(c)) if c.get('output_folder') else self._show_project('New project (not saved)')

    def collect_cfg(self):
        c = self.cfg
        for key in ('input_folder', 'output_folder', 'dataset_name', 'video_extensions', 'bf_suffix', 'device'):
            c[key] = self.v[key].get().strip() if key != 'bf_suffix' else self.v[key].get()
        for key in ('paired_marker', 'resize', 'register', 'invert_intensity'):
            c[key] = bool(self.v[key].get())
        c['input_type'] = self.v['input_type'].get() or 'videos'
        seg = {t: k for k, t in CTC_SEGMENTATIONS}
        c['ctc']['segmentation'] = seg.get(self.v['ctc_segmentation_label'].get(), 'auto')
        try:
            c['ctc']['min_track_len'] = max(1, int(float(self.v['ctc_min_track_len'].get())))
        except ValueError:
            pass
        for key, conv in (('pixel_size_um', float), ('time_interval_h', float), ('n_jobs', int)):
            try:
                c[key] = conv(self.v[key].get())
            except ValueError:
                pass
        c['channel'] = CHANNELS.index(self.v['channel_label'].get()) - 1 if self.v['channel_label'].get() in CHANNELS else 0
        c['marker_channel'] = CHANNELS.index(self.v['marker_channel_label'].get()) - 1 if self.v['marker_channel_label'].get() in CHANNELS else -1
        c['skip_done'] = bool(self.skip_var.get())
        a = c['analysis']
        a['name'] = self.a['name'].get().strip() or 'SPOT_analysis'
        a['embedding'] = self.a['embedding'].get() or 'UMAP'
        for key in ('n_neighbors', 'k_max', 'n_time_bins'):
            try:
                a[key] = int(float(self.a[key].get()))
            except ValueError:
                pass
        a['n_clusters'] = self.a['n_clusters'].get().strip() or 'auto'
        a['module_scale'] = self.a['module_scale'].get().strip() or 'best'
        for key in ('normalise_per_batch', 'equal_weighting', 'time_varying_selection', 'hmm', 'save_vector'):
            a[key] = bool(self.a[key].get())
        a['group_column'] = GROUP_COLUMNS.get(self.a['group_label'].get(), 'Conditions')
        a['batch_column'] = GROUP_COLUMNS.get(self.a['batch_label'].get(), '')
        a['marker_group_column'] = GROUP_COLUMNS.get(self.a['marker_group_label'].get(), 'Marker')
        return c

    # ------------------------------------------------------------------ projects
    def new_project(self):
        if self.proc is not None:
            return
        self.cfg = sp.merge_config({})
        self.videos = []
        self.video_tree.delete(*self.video_tree.get_children())
        self.sample_tree.delete(*self.sample_tree.get_children())
        self.apply_cfg()
        self.refresh_status()
        self.refresh_results()
        self.nb.select(0)

    def ask_open_project(self):
        f = filedialog.askopenfilename(title='Open SPOT project', filetypes=[('SPOT project', 'spot_project.json'),
                                                                            ('JSON', '*.json')])
        if f:
            self.open_project(f)

    def open_project(self, path):
        try:
            self.cfg = sp.load_config(path)
        except Exception as e:
            messagebox.showerror(APP_TITLE, 'Could not open the project:\n%s' % e)
            return
        # the project may have been moved together with its results folder
        self.cfg['output_folder'] = os.path.dirname(os.path.abspath(path))
        self.sample_tree.delete(*self.sample_tree.get_children())
        self.apply_cfg()
        self.scan_videos()
        self.refresh_results()
        self.settings['last_project'] = os.path.abspath(path)
        save_settings(self.settings)

    def ask_load_settings(self):
        if self.proc is not None:
            messagebox.showinfo(APP_TITLE, 'Wait for the current run to finish.')
            return
        self.collect_cfg()
        start = self.cfg.get('output_folder') if os.path.isdir(self.cfg.get('output_folder') or '') else os.path.expanduser('~')
        f = filedialog.askopenfilename(title='Load settings', initialdir=start,
                                       filetypes=[('SAM-SPOT settings', '*.json')])
        if f:
            self.load_settings_file(f)

    def load_settings_file(self, f, use_folders=None, restore_samples=None):
        """ load the settings of a project (spot_project.json), a run (run_history/<id>/run_config.json) or an
        analysis (analysis/<name>/analysis_config.json) """
        raw = sp.read_json(f)
        if not isinstance(raw, dict) or not (('config' in raw) or ('input_folder' in raw)):
            messagebox.showerror(APP_TITLE, '%s is not a SAM-SPOT settings file.' % f)
            return False
        new = sp.merge_config(raw)
        have_folders = bool(self.cfg.get('output_folder'))
        if use_folders is None:
            use_folders = True
            if have_folders and os.path.normpath(new.get('output_folder', '')) != os.path.normpath(self.cfg['output_folder']):
                ans = messagebox.askyesnocancel(
                    APP_TITLE, 'Use the videos and results folders saved in these settings?\n\n'
                               'Yes: use their folders (reopen that project as it was run)\n'
                               'No: keep the current folders and apply the settings to this project')
                if ans is None:
                    return False
                use_folders = ans
        if not use_folders:
            for k in ('input_folder', 'output_folder', 'dataset_name', 'selected_videos'):
                new[k] = self.cfg.get(k)
        self.cfg = new
        self.sample_tree.delete(*self.sample_tree.get_children())
        self.apply_cfg()
        self.scan_videos()
        # the sample table saved with a run / analysis record
        rec_dir = os.path.dirname(os.path.abspath(f))
        samples = [os.path.join(rec_dir, n) for n in ('sample_info.csv', 'sample_info_used.csv')
                   if os.path.exists(os.path.join(rec_dir, n))]
        if samples:
            if restore_samples is None:
                restore_samples = messagebox.askyesno(APP_TITLE, 'Also restore the sample information (conditions, '
                                                                 'genotypes, ...) saved with these settings?')
            if restore_samples:
                self.import_samples(samples[0])
        self.refresh_status()
        self.refresh_results()
        when = raw.get('finished') or raw.get('_saved_by') or ''
        self.progress_label.configure(text='Loaded settings from %s %s - not saved to the project yet.'
                                           % (os.path.basename(f), ('(%s)' % when) if when else ''))
        self._show_project('Unsaved settings from: ', f)
        messagebox.showinfo(APP_TITLE, 'Settings loaded%s.\n\nOn the "Run pipeline" tab, ⟳ marks results that '
                                       'were computed with different settings; they are recomputed when you run.'
                            % (' from run %s' % raw['run_id'] if raw.get('run_id') else ''))
        return True

    def open_run_history(self):
        self.collect_cfg()
        d = sp.run_history_folder(self.cfg) if self.cfg.get('output_folder') else ''
        if d and os.path.isdir(d):
            open_path(d)
        else:
            messagebox.showinfo(APP_TITLE, 'No run has been recorded in this results folder yet.')

    def save_project(self, quiet=False):
        self.collect_cfg()
        if not self.cfg.get('output_folder'):
            if not quiet:
                messagebox.showerror(APP_TITLE, 'Choose a results folder first.')
            return False
        sp.save_config(self.cfg, sp.project_file(self.cfg))
        self.settings['last_project'] = sp.project_file(self.cfg)
        save_settings(self.settings)
        self._show_project('Project: ', sp.project_file(self.cfg))
        if self.sample_dirty and self.sample_tree.get_children():
            self.save_samples(quiet=True)
        if not quiet:
            self.progress_label.configure(text='Project saved.')
        return True

    # ------------------------------------------------------------------ misc
    def _on_tab_changed(self, _=None):
        idx = self.nb.index('current')
        if idx == 2:
            self.collect_cfg()
            self.refresh_status()
        elif idx == 3:
            self.collect_cfg()
            if not self.qc_video.get() and self.videos:
                done = [v['name'] for v in self.videos if os.path.exists(sp.paths(self.cfg, v['name'])['final'])]
                self.qc_video.set(done[0] if done else self.videos[0]['name'])
            self.qc_load()
        elif idx == 4:
            self.refresh_results()

    def _on_close(self):
        if self.proc is not None:
            if not messagebox.askyesno(APP_TITLE, 'A run is in progress. Stop it and quit?'):
                return
            self._stopped = True
            try:
                if sys.platform.startswith('win'):
                    subprocess.call(['taskkill', '/F', '/T', '/PID', str(self.proc.pid)],
                                    creationflags=subprocess.CREATE_NO_WINDOW,
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                else:
                    self.proc.terminate()
            except Exception:
                pass
        if self.cfg.get('output_folder') and os.path.isdir(self.cfg['output_folder']):
            self.save_project(quiet=True)
        self.destroy()


def main():
    project = sys.argv[1] if len(sys.argv) > 1 else None
    app = SpotApp(project)
    app.mainloop()


if __name__ == '__main__':
    main()
