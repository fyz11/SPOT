# -*- coding: utf-8 -*-
"""
SPOT pipeline engine used by the SPOT GUI (spot_gui.py).

Runs the full SPOT workflow on a folder of videos, following the reference analysis scripts in
data/full_dataset_with_biomarker/reference_analysis_scripts:

    Stage 1  prepare (resize + drift registration) -> detect (YOLOv3) -> track -> segment (U-Net) -> postprocess
    Stage 2  SAM features (shape, appearance, motion) -> marker measurements (optional) -> compile dataset tables
    Stage 3  analysis: SAM modules, embedding, phenotype clusters, trajectories and HMM transitions (no p-Creode)

Instead of videos it can also take a Cell Tracking Challenge dataset (celltrackingchallenge.net): the cells are then
outlined and tracked from the annotations (or a tracker's results in the challenge format) instead of the networks.

It can also be run from the command line:

    python spot_pipeline.py --config <output_folder>/spot_project.json --steps prepare,detect,...

Progress is reported on stdout with lines starting with '@@' which the GUI parses.
"""
import os
import sys
import json
import glob
import re
import time
import shutil
import subprocess
import tempfile
import traceback

import numpy as np

GUI_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(GUI_DIR)
if os.path.isdir(os.path.join(REPO_DIR, 'SPOT')) and REPO_DIR not in sys.path:
    # use the SPOT source of this repository if SPOT is not pip installed
    sys.path.insert(0, REPO_DIR)


# =============================================================================
#   Configuration
# =============================================================================

STEPS = [
    # key, label, stage, per-video
    ('prepare', 'Prepare videos (resize + drift correction)', 1, True),
    ('detect', 'Detect organoids (YOLOv3 network)', 1, True),
    ('track', 'Track organoids over time', 1, True),
    ('segment', 'Segment organoid outlines (U-Net)', 1, True),
    ('postprocess', 'Clean up tracks (filter, smooth, remove border)', 1, True),
    ('features', 'Compute SAM phenome (shape, appearance, motion)', 2, True),
    ('markers', 'Measure fluorescent marker (paired videos only)', 2, True),
    ('compile', 'Compile dataset tables and image patches', 2, False),
    ('analysis', 'Analyse SAM phenome (modules, clusters, trajectories)', 3, False),
]
STEP_KEYS = [s[0] for s in STEPS]
STEP_LABELS = {s[0]: s[1] for s in STEPS}
PER_VIDEO_STEPS = [s[0] for s in STEPS if s[3]]

STEP_HELP = {
    'prepare': 'Reads each video, converts it to grayscale, resizes the frames to 512 x 512 (the size the networks were '
               'trained at) and removes stage drift by translation registration. The paired marker video, if any, is '
               'shifted identically.',
    'detect': 'Finds organoid bounding boxes in every frame with the pretrained YOLOv3 organoid detector.',
    'track': 'Links the bounding boxes over time into organoid tracks, using optical flow to predict the motion.',
    'segment': 'Segments the outline of each tracked organoid in every frame with the pretrained attention U-Net.',
    'postprocess': 'Removes duplicate / unstable tracks, smooths the outlines over time, splits tracks where the shape '
                   'jumps and removes organoids touching the image border. Cell Tracking Challenge data: outlines the '
                   'annotated cells of every track and removes cells touching the image border.',
    'features': 'Computes the Shape, Appearance and Motion (SAM) phenome of every organoid at every time point.',
    'markers': 'Measures the fluorescent marker inside each organoid: mean intensity, number of bright spots and their '
               'intensity. Not part of the SAM phenome; used to annotate the analysis.',
    'compile': 'Combines all videos with the sample information into dataset tables (.csv) of SAM features, and crops '
               'an image patch of every organoid for display.',
    'analysis': 'Normalises the features and finds SAM modules (co-varying features), a 2D map (UMAP / PaCMAP) of all '
                'organoids, phenotype clusters, the average phenotype trajectory per condition and the transitions '
                'between phenotypes (hidden Markov model).',
}

DEFAULT_CONFIG = {
    'input_type': 'videos',    # 'videos' or 'ctc' (Cell Tracking Challenge dataset)
    'input_folder': '',
    'output_folder': '',
    'dataset_name': '',
    'video_extensions': '.avi .tif .tiff .wmv .mp4 .mov',
    'paired_marker': False,
    'bf_suffix': ' BF',
    'channel': 0,              # channel used for colour videos (0 = red / first, -1 = brightest)
    'marker_channel': -1,
    'pixel_size_um': 1.0,      # pixel size of the video files as provided
    'time_interval_h': 1.0,    # time between frames
    'resize': True,
    'resize_size': 512,
    'register': True,
    'invert_intensity': True,  # organoids darker than background (brightfield / phase contrast)
    'device': 'auto',
    'n_jobs': -1,
    'skip_done': True,
    'selected_videos': [],     # empty = all
    'detector_weights': '',
    'segmenter_weights': '',
    'detect': {'obj_thresh': 0.001, 'nms_thresh': 0.4, 'equalize_hist': True},
    'track': {'iou_match': 0.25, 'wait_time': 5, 'min_aspect_ratio': 3, 'max_dense_bbox_cover': 8},
    'segment': {'segment_thresh': 0.6, 'min_I': 0.1, 'filter_prob_thresh': 0.05, 'filter_life_thresh': 0.1},
    'postprocess': {'border_pixel_size': 20, 'border_pixel_frac': 0.1, 'temporal_smooth_win_size': 5,
                    'min_split_track_len': 5, 'iou_consistency_thresh': 0.25},
    'ctc': {'segmentation': 'auto', 'min_track_len': 5, 'contour_smooth_sigma': 1.0},
    'features': {'use_mask': True},
    'markers': {'spot_threshold': 0.05, 'spot_min_distance': 3},
    'analysis': {
        'name': 'SPOT_analysis',
        'group_column': 'Conditions',
        'batch_column': '',
        'regress_batch': True,
        'normalise_per_batch': False,
        'time_varying_selection': True,
        'equal_weighting': True,
        'embedding': 'UMAP',
        'n_neighbors': 100,
        'module_scale': 'best',
        'n_clusters': 'auto',
        'k_min': 2,
        'k_max': 20,
        'n_time_bins': 16,
        'trajectory_density_sigma': 2.0,
        'hmm': True,
        'marker_group_column': 'Marker',
        'save_vector': False,
        'max_patches_on_embedding': 1500,
    },
}

METADATA_COLUMNS = ['Filename', 'Img_Channel', 'Genetics', 'Patients', 'Conditions', 'Marker',
                    'pixel_resolution[um]', 'time_resolution[h]', 'n_frames']

MARKER_FEATURE_NAMES = ['mean_marker_intensity', 'mean_marker_intensity_raw', 'n_spots',
                        'mean_marker_spot_intensity', 'mean_marker_spot_intensity_raw']

BOUNDARY_SAMPLES = 200


GUI_VERSION = '1.0'


def merge_config(cfg):
    """ fill in missing entries of a (possibly older) project config with the defaults """
    out = json.loads(json.dumps(DEFAULT_CONFIG))
    cfg = cfg or {}
    if 'config' in cfg and isinstance(cfg['config'], dict):
        # a run record (run_config.json) or analysis record (analysis_config.json)
        cfg = cfg['config']
    for k, v in cfg.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k].update(v)
        else:
            out[k] = v
    return out


def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return merge_config(json.load(f))


def save_config(cfg, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    cfg = dict(cfg)
    cfg['_saved_by'] = 'SAM-SPOT GUI %s, %s' % (GUI_VERSION, time.strftime('%Y-%m-%d %H:%M:%S'))
    write_json(path, cfg)


def write_json(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)
    os.replace(tmp, path)


def read_json(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


# =============================================================================
#   Reproducibility: parameters of every step, run records and software versions
# =============================================================================

# settings that change the result of each per-video step (device / n_jobs do not)
STEP_PARAM_KEYS = {
    'prepare': ['input_type', 'paired_marker', 'bf_suffix', 'channel', 'marker_channel', 'pixel_size_um',
                'time_interval_h', 'resize', 'resize_size', 'register'],
    'detect': ['detect', 'detector_weights'],
    'track': ['track'],
    'segment': ['segment', 'invert_intensity', 'segmenter_weights'],
    'postprocess': ['postprocess', 'ctc'],
    'features': ['features'],
    'markers': ['markers'],
}
UPSTREAM = {'detect': 'prepare', 'track': 'detect', 'segment': 'track', 'postprocess': 'segment',
            'features': 'postprocess', 'markers': 'postprocess'}


def _canonical(obj):
    """ json round trip so that e.g. tuples / numpy scalars compare equal to what was saved """
    return json.loads(json.dumps(obj, sort_keys=True, default=str))


def step_params(cfg, video, step):
    """ the settings that determine a step's output, including those of the steps it depends on """
    d = {k: cfg.get(k) for k in STEP_PARAM_KEYS[step]}
    if step == 'prepare':
        d['input_files'] = [os.path.basename(video['bf_file']),
                            os.path.basename(video['marker_file']) if video.get('marker_file') else None]
    if upstream_step(cfg, step) is not None:
        d['upstream'] = step_params(cfg, video, upstream_step(cfg, step))
    return _canonical(d)


def is_ctc(cfg):
    return cfg.get('input_type') == 'ctc'


# steps not used for Cell Tracking Challenge data: the annotations give the outlines and tracks
CTC_SKIPPED_STEPS = ('detect', 'track', 'segment')


def upstream_step(cfg, step):
    if is_ctc(cfg) and step == 'postprocess':
        return 'prepare'
    return UPSTREAM.get(step)


def params_file(cfg, name, step):
    return os.path.join(video_folder(cfg, name), 'params', step + '.json')


def write_step_params(cfg, video, step, run_id):
    write_json(params_file(cfg, video['name'], step),
               {'step': step, 'video': video['name'], 'run_id': run_id,
                'finished': time.strftime('%Y-%m-%d %H:%M:%S'), 'params': step_params(cfg, video, step)})


def params_changed(cfg, video, step):
    """ True if the saved output of the step was computed with other settings than the current ones """
    rec = read_json(params_file(cfg, video['name'], step))
    if rec is None:
        return False   # computed before parameter records existed: unknown, assume up to date
    return rec.get('params') != step_params(cfg, video, step)


def compile_params(cfg, videos):
    table = None
    try:
        table = read_metadata(cfg)
    except Exception:
        pass
    samples = {}
    if table is not None:
        for _, r in table.iterrows():
            samples[str(r['Filename'])] = {c: str(r[c]) for c in ['Conditions', 'Genetics', 'Patients', 'Marker']
                                           if c in table.columns}
    vids = {}
    for v in videos:
        if step_status(cfg, v, 'features', check_params=False) == 'done':
            vids[v['name']] = {'features': step_params(cfg, v, 'features'),
                               'markers': step_params(cfg, v, 'markers') if v.get('marker_file') and cfg.get('paired_marker') else None,
                               'sample_info': samples.get(v['name'])}
    return _canonical({'dataset_name': dataset_name(cfg), 'videos': vids})


def compile_record_file(cfg):
    return os.path.join(cfg['output_folder'], dataset_name(cfg) + '_compile_params.json')


def analysis_record_file(cfg):
    return os.path.join(analysis_folder(cfg), 'analysis_config.json')


def analysis_params(cfg):
    rec = read_json(compile_record_file(cfg)) or {}
    return _canonical({'analysis': cfg['analysis'], 'compiled_data': rec.get('params')})


def _version(pkg):
    try:
        import importlib.metadata as md
        return md.version(pkg)
    except Exception:
        return None


def _git(*args):
    try:
        return subprocess.run(['git', '-C', REPO_DIR] + list(args), capture_output=True, text=True, timeout=10,
                              creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0)).stdout.strip()
    except Exception:
        return ''


_SHA_CACHE = {}


def _sha256(path):
    import hashlib
    if not path or not os.path.exists(path):
        return None
    key = (path, os.path.getmtime(path))
    if key not in _SHA_CACHE:
        h = hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(1 << 20), b''):
                h.update(chunk)
        _SHA_CACHE[key] = h.hexdigest()
    return _SHA_CACHE[key]


def provenance(cfg, include_weights=True):
    """ software, hardware and model versions, to report and reproduce a run """
    import platform
    spot_file = None
    try:
        import SPOT
        spot_file = os.path.dirname(SPOT.__file__)
    except Exception:
        pass
    spot_version = _version('SAM-SPOT')
    if spot_file and os.path.abspath(spot_file).startswith(os.path.abspath(REPO_DIR)):
        # SPOT used from this repository's source: version from pyproject.toml + git commit
        try:
            with open(os.path.join(REPO_DIR, 'pyproject.toml'), encoding='utf-8') as f:
                m = re.search(r'^version\s*=\s*"([^"]+)"', f.read(), re.M)
            spot_version = (m.group(1) if m else spot_version) + ' (source)'
        except Exception:
            pass
    commit = _git('rev-parse', 'HEAD')
    prov = {
        'time': time.strftime('%Y-%m-%d %H:%M:%S'),
        'gui_version': GUI_VERSION,
        'spot_version': spot_version,
        'spot_location': spot_file,
        'git_commit': commit or None,
        'git_uncommitted_changes': bool(_git('status', '--porcelain', '--untracked-files=no')) if commit else None,
        'python': sys.version.split()[0],
        'platform': platform.platform(),
        'packages': {p: _version(p) for p in ['numpy', 'scipy', 'scikit-image', 'scikit-learn', 'pandas', 'torch',
                                              'opencv-python-headless', 'opencv-python', 'umap-learn', 'pacmap',
                                              'hmmlearn', 'mahotas', 'matplotlib', 'seaborn']},
        'device': cfg.get('device'),
    }
    try:
        import torch
        prov['cuda_device'] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        pass
    if include_weights:
        prov['detector_weights'] = {'path': detector_weights(cfg) or 'pretrained (downloaded)',
                                    'sha256': _sha256(detector_weights(cfg))}
        prov['segmenter_weights'] = {'path': segmenter_weights(cfg) or 'pretrained (downloaded)',
                                     'sha256': _sha256(segmenter_weights(cfg))}
    return prov


def project_file(cfg):
    return os.path.join(cfg['output_folder'], 'spot_project.json')


def dataset_name(cfg):
    name = (cfg.get('dataset_name') or '').strip()
    if not name:
        name = os.path.basename(os.path.normpath(cfg['output_folder'])) or 'SPOT_dataset'
    return safe_name(name)


def safe_name(name):
    """ a name usable in filenames and in SPOT's '_' separated object ids """
    name = re.sub(r'[\\/:*?"<>|]+', '-', str(name)).strip()
    return name.replace('_', '-') or 'unnamed'


def metadata_file(cfg):
    return os.path.join(cfg['output_folder'], 'metadata_expt.csv')


def analysis_folder(cfg):
    return os.path.join(cfg['output_folder'], 'analysis', safe_label(cfg['analysis'].get('name') or 'SPOT_analysis'))


# =============================================================================
#   Video discovery and file layout
# =============================================================================

def _extensions(cfg):
    exts = [e.strip().lower() for e in re.split(r'[\s,;]+', cfg.get('video_extensions', '')) if e.strip()]
    return [e if e.startswith('.') else '.' + e for e in exts]


def discover_ctc_sequences(folder):
    """ the sequences of a Cell Tracking Challenge dataset folder (e.g. PhC-C2DH-U373 with 01, 01_GT, 01_ST, ...),
    or of every dataset folder inside the given folder, as dicts {name, bf_file (= the sequence folder), marker_file} """
    def sequences(ds):
        out = []
        for seq in sorted(os.listdir(ds)):
            sf = os.path.join(ds, seq)
            if re.match(r'^\d+$', seq) and os.path.isdir(sf) and glob.glob(os.path.join(sf, '*.tif')):
                out.append({'name': safe_name('%s-%s' % (os.path.basename(os.path.normpath(ds)), seq)),
                            'bf_file': sf, 'marker_file': None})
        return out
    videos = sequences(folder)
    if not videos:
        for sub in sorted(os.listdir(folder)):
            if os.path.isdir(os.path.join(folder, sub)):
                videos += sequences(os.path.join(folder, sub))
    return videos


def ctc_track_file(seq_folder, segmentation='auto'):
    """ the track table of a sequence: <seq>_RES/res_track.txt for 'res' (a tracker's results in the challenge
    format), otherwise the gold truth <seq>_GT/TRA/man_track.txt """
    seq_folder = os.path.normpath(seq_folder)
    res = os.path.join(seq_folder + '_RES', 'res_track.txt')
    gt = os.path.join(seq_folder + '_GT', 'TRA', 'man_track.txt')
    if segmentation == 'res':
        return res if os.path.exists(res) else None
    return gt if os.path.exists(gt) else None


def discover_videos(cfg):
    """ list the videos of the input folder as dicts {name, bf_file, marker_file}

    With paired_marker, each brightfield video '<name><bf_suffix>.<ext>' is paired with the marker video '<name>.<ext>'.
    """
    folder = cfg.get('input_folder', '')
    if not folder or not os.path.isdir(folder):
        return []
    if is_ctc(cfg):
        videos = discover_ctc_sequences(folder)
        for v in videos:
            v['ctc_tracks'] = ctc_track_file(v['bf_file'], cfg['ctc'].get('segmentation', 'auto'))
        return videos
    exts = _extensions(cfg)
    files = sorted(f for f in os.listdir(folder)
                   if os.path.isfile(os.path.join(folder, f)) and os.path.splitext(f)[1].lower() in exts)
    videos = []
    if cfg.get('paired_marker'):
        suffix = cfg.get('bf_suffix', ' BF')
        stems = {}
        for f in files:
            stems.setdefault(os.path.splitext(f)[0], f)
        for f in files:
            stem = os.path.splitext(f)[0]
            if suffix and stem.endswith(suffix):
                base = stem[:-len(suffix)]
                marker = stems.get(base)
                videos.append({'name': safe_name(base),
                               'bf_file': os.path.join(folder, f),
                               'marker_file': os.path.join(folder, marker) if marker else None})
    else:
        for f in files:
            videos.append({'name': safe_name(os.path.splitext(f)[0]),
                           'bf_file': os.path.join(folder, f),
                           'marker_file': None})
    return videos


def selected_videos(cfg):
    videos = discover_videos(cfg)
    sel = cfg.get('selected_videos') or []
    if sel:
        videos = [v for v in videos if v['name'] in sel]
    return videos


def video_folder(cfg, name):
    return os.path.join(cfg['output_folder'], 'videos', name)


def paths(cfg, name):
    vf = video_folder(cfg, name)
    seg = os.path.join(vf, 'organoid_segmentation')
    return {
        'folder': vf,
        'info': os.path.join(vf, name + '_info.json'),
        'vid': os.path.join(vf, name + '_register.tif'),
        'marker_vid': os.path.join(vf, name + '_marker_register.tif'),
        'register_params': os.path.join(vf, name + '_translation-register-params.mat'),
        'bbox_folder': os.path.join(vf, 'bbox_detection'),
        'bbox_ch_folder': os.path.join(vf, 'bbox_detection', 'Channel-01'),
        'tracks': os.path.join(vf, 'bbox_detection', 'flow_bbox_tracks_Ch-1.mat'),
        'seg_folder': seg,
        'seg': os.path.join(seg, 'org_boundaries-Channel-1.mat'),
        'final': os.path.join(seg, name + '_boundaries_final_RGB.mat'),
        'shape': os.path.join(seg, name + '_final_cont_RGB_shape-feats.mat'),
        'image': os.path.join(seg, name + '_final_cont_RGB_image-feats.mat'),
        'motion': os.path.join(seg, name + '_final_cont_RGB_motion-feats.mat'),
        'marker_feats': os.path.join(seg, name + '_final_cont_RGB_additional-feats.mat'),
        'no_objects': os.path.join(seg, 'no_organoids_found.txt'),
    }


def dataset_paths(cfg):
    out = cfg['output_folder']
    expt = dataset_name(cfg)
    return {
        'shape': os.path.join(out, expt + '_shape_features.csv'),
        'image': os.path.join(out, expt + '_image_features.csv'),
        'motion': os.path.join(out, expt + '_motion_features.csv'),
        'marker': os.path.join(out, expt + '_marker_features.csv'),
        'boundaries': os.path.join(out, expt + '_boundaries_smooth_final.mat'),
        'patches': os.path.join(out, expt + '_boundaries_smooth_final_img_patches.mat'),
    }


def read_info(cfg, name):
    p = paths(cfg, name)['info']
    if os.path.exists(p):
        with open(p, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def step_status(cfg, video, step, check_params=True):
    """ 'done', 'todo', 'n/a' (not applicable), 'none' (no organoids found) or 'stale' (computed with other settings
    than the current ones) for a per-video step """
    st = _output_status(cfg, video, step)
    if check_params and st in ('done', 'none') and params_changed(cfg, video, step):
        return 'stale'
    return st


def _output_status(cfg, video, step):
    name = video['name']
    p = paths(cfg, name)
    if is_ctc(cfg) and step in CTC_SKIPPED_STEPS:
        return 'n/a'
    if step == 'prepare':
        ok = os.path.exists(p['vid']) and os.path.exists(p['info'])
        if ok and cfg.get('paired_marker') and video.get('marker_file'):
            ok = os.path.exists(p['marker_vid'])
        return 'done' if ok else 'todo'
    if step == 'detect':
        info = read_info(cfg, name)
        if info is None or not os.path.isdir(p['bbox_ch_folder']):
            return 'todo'
        n = len(glob.glob(os.path.join(p['bbox_ch_folder'], '*.txt')))
        return 'done' if n >= info.get('n_frames', 1) else 'todo'
    if step == 'track':
        return 'done' if os.path.exists(p['tracks']) else 'todo'
    if step == 'segment':
        return 'done' if os.path.exists(p['seg']) else 'todo'
    if step == 'postprocess':
        if os.path.exists(p['final']):
            return 'none' if os.path.exists(p['no_objects']) else 'done'
        return 'todo'
    if step == 'features':
        if os.path.exists(p['no_objects']) and os.path.exists(p['final']):
            return 'none'
        return 'done' if os.path.exists(p['motion']) else 'todo'
    if step == 'markers':
        if not (cfg.get('paired_marker') and video.get('marker_file')):
            return 'n/a'
        if os.path.exists(p['no_objects']) and os.path.exists(p['final']):
            return 'none'
        return 'done' if os.path.exists(p['marker_feats']) else 'todo'
    return 'todo'


def dataset_step_status(cfg, step, check_params=True):
    if step == 'compile':
        d = dataset_paths(cfg)
        if not (os.path.exists(d['shape']) and os.path.exists(d['patches'])):
            return 'todo'
        rec = read_json(compile_record_file(cfg))
        if check_params and rec is not None and rec.get('params') != compile_params(cfg, selected_videos(cfg)):
            return 'stale'
        return 'done'
    if step == 'analysis':
        if not os.path.exists(os.path.join(analysis_folder(cfg), 'summary.json')):
            return 'todo'
        rec = read_json(analysis_record_file(cfg))
        if check_params and rec is not None and (rec.get('params') != analysis_params(cfg) or
                                                 dataset_step_status(cfg, 'compile') == 'stale'):
            return 'stale'
        return 'done'
    return 'todo'


# =============================================================================
#   Messages to the GUI
# =============================================================================

_LOG_FH = None   # the run's log file in run_history/<run id>/


def emit(kind, *args):
    line = '@@' + kind + ' ' + '\t'.join(str(a) for a in args)
    print(line, flush=True)
    if _LOG_FH is not None and kind != 'PROGRESS':
        _LOG_FH.write(time.strftime('%H:%M:%S ') + line + '\n')
        _LOG_FH.flush()


def log(*args):
    print(*args, flush=True)
    if _LOG_FH is not None:
        _LOG_FH.write(' '.join(str(a) for a in args) + '\n')
        _LOG_FH.flush()


# =============================================================================
#   Helpers
# =============================================================================

def read_video(path):
    """ read a video (.avi, .wmv, .mp4, ... with OpenCV; .tif with tifffile) as a numpy array (n_frames, rows, cols[, ch]) """
    ext = os.path.splitext(path)[1].lower()
    if ext in ('.tif', '.tiff'):
        import tifffile
        return np.asarray(tifffile.imread(path))
    import SPOT.Utility_Functions.file_io as fio
    vid = fio.read_video_cv2(path)
    if vid.size == 0:
        # OpenCV cannot open non-ASCII paths on Windows (e.g. 'µM' in the filename): read a temporary copy
        tmpdir = tempfile.mkdtemp(prefix='spot_')
        try:
            tmp = os.path.join(tmpdir, 'video' + ext)
            shutil.copyfile(path, tmp)
            vid = fio.read_video_cv2(tmp)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
    if vid.size == 0:
        raise IOError('could not read any frame from %s' % path)
    return vid


def to_gray(vid, channel=0):
    """ (n_frames, rows, cols[, ch]) -> (n_frames, rows, cols) """
    vid = np.asarray(vid)
    if vid.ndim == 4 and int(channel) < 0:
        # auto: the brightest channel (e.g. a marker saved in the blue channel of an RGB movie)
        axis = -1 if vid.shape[-1] in (3, 4) else 1
        means = [np.take(vid, c, axis=axis).mean() for c in range(vid.shape[axis])]
        channel = int(np.argmax(means))
        log('using channel %d (brightest)' % (channel + 1))
    if vid.ndim == 4:
        if vid.shape[-1] in (3, 4):
            # an RGB(A) video saved from a grayscale movie has identical channels
            ch = min(int(channel), vid.shape[-1] - 1)
            vid = vid[..., ch]
        else:
            # (n_frames, ch, rows, cols) tif
            ch = min(int(channel), vid.shape[1] - 1)
            vid = vid[:, ch]
    elif vid.ndim == 2:
        vid = vid[None]
    return vid


def to_uint8(vid):
    vid = np.asarray(vid, dtype=np.float32)
    lo, hi = np.nanmin(vid), np.nanmax(vid)
    if vid.dtype != np.uint8 and (hi > 255 or lo < 0):
        vid = 255. * (vid - lo) / max(hi - lo, 1e-8)
    elif hi <= 1.0 and hi > 0:
        vid = vid * 255.
    return np.uint8(np.clip(vid, 0, 255))


def resize_vid(vid, size):
    import skimage.transform as sktform
    return np.array([sktform.resize(vv, output_shape=(size, size), order=1, preserve_range=True) for vv in vid])


def imsave_tif(path, arr):
    import tifffile
    tifffile.imwrite(path, arr)


def imread_tif(path):
    import tifffile
    vid = np.asarray(tifffile.imread(path))
    if vid.ndim == 4:
        vid = vid[..., 0]
    return vid


def device_arg(cfg):
    dev = (cfg.get('device') or 'auto').strip().lower()
    return None if dev in ('', 'auto') else dev


def _repo_weights(subfolder, filename):
    p = os.path.join(REPO_DIR, 'models', subfolder, filename)
    return p if os.path.exists(p) else None


def detector_weights(cfg):
    w = (cfg.get('detector_weights') or '').strip()
    if w:
        return w
    # None = SPOT downloads the pretrained detector from Zenodo on first use
    return _repo_weights('detect_CNN_model', 'keras_YOLOv3_organoid_detector2.pt')


def segmenter_weights(cfg):
    w = (cfg.get('segmenter_weights') or '').strip()
    if w:
        return w
    return _repo_weights('segment_CNN_model', 'organoid-bbox_attn_seg-unet-master-v2_64x64-8feats_v2.pt')


def mat_names(arr):
    return np.array([str(n).strip() for n in np.ravel(arr)])


def load_final_boundaries(cfg, name):
    """ final tracked boundaries (n_objects, n_frames, n_points, 2) of the single analysed channel """
    import scipy.io as spio
    obj = spio.loadmat(paths(cfg, name)['final'])
    b = obj['boundaries_smooth_final']
    if b.ndim == 5:
        b = b[0]
    if b.ndim != 4:
        b = np.zeros((0, 0, BOUNDARY_SAMPLES, 2))
    return b.astype(np.float64)


# =============================================================================
#   Stage 1
# =============================================================================

def read_ctc_sequence(folder, channel=0):
    """ the frames t000.tif, t001.tif, ... of a Cell Tracking Challenge sequence as (n_frames, rows, cols) uint8;
    16 bit images are scaled between the 0.1 and 99.9 percentiles of the whole sequence """
    import tifffile
    files = sorted(glob.glob(os.path.join(folder, 't*.tif')),
                   key=lambda f: int(re.sub(r'\D', '', os.path.basename(f)) or 0))
    if not files:
        raise IOError('no t###.tif frames in %s' % folder)
    vid = to_gray(np.array([np.asarray(tifffile.imread(f)) for f in files]), channel)
    if vid.dtype != np.uint8:
        lo, hi = np.percentile(vid, [0.1, 99.9])
        vid = np.uint8(np.clip(255. * (vid.astype(np.float32) - lo) / max(hi - lo, 1e-8), 0, 255))
    return vid, files


def step_prepare(cfg, video):
    import scipy.io as spio
    import SPOT.Registration.registration_2D as SPOT_register

    if is_ctc(cfg):
        return step_prepare_ctc(cfg, video)
    name = video['name']
    p = paths(cfg, name)
    os.makedirs(p['folder'], exist_ok=True)

    log('reading', video['bf_file'])
    vid = to_gray(read_video(video['bf_file']), cfg.get('channel', 0))
    orig_shape = vid.shape[1:3]
    vid_marker = None
    if cfg.get('paired_marker') and video.get('marker_file'):
        log('reading', video['marker_file'])
        vid_marker = to_gray(read_video(video['marker_file']), cfg.get('marker_channel', 0))
        if len(vid_marker) != len(vid):
            n = min(len(vid), len(vid_marker))
            log('WARNING: brightfield and marker videos have %d and %d frames, using the first %d'
                % (len(vid), len(vid_marker), n))
            vid, vid_marker = vid[:n], vid_marker[:n]

    if cfg.get('resize', True):
        size = int(cfg.get('resize_size', 512))
        vid = resize_vid(vid, size)
        if vid_marker is not None:
            vid_marker = resize_vid(vid_marker, size)
    vid = to_uint8(vid)
    if vid_marker is not None:
        vid_marker = to_uint8(vid_marker)

    disps = np.zeros((len(vid), 2))
    blurred = []
    if cfg.get('register', True) and len(vid) > 1:
        log('registering frames to remove drift')
        vid_reg, blurred, disps = SPOT_register.translation_register_blurry_phase_contrast_videos(
            np.uint8(255. * vid / float(max(vid.max(), 1))),
            shape=vid.shape[1:3],
            sub_pix_res=1,
            impute_pixels=True,
            use_registered=True,
            use_dog=False,
            dog_sigma=3,
            detect_blurred_frames=False,
            blur_factor=3.,
            apply_crop_mask=False)
        vid = to_uint8(vid_reg)
        if vid_marker is not None:
            vid_marker = to_uint8(SPOT_register.apply_translation_register_correlation(vid_marker, disps,
                                                                                       use_registered=True))

    imsave_tif(p['vid'], vid)
    if vid_marker is not None:
        imsave_tif(p['marker_vid'], vid_marker)
    spio.savemat(p['register_params'], {'disps': np.asarray(disps), 'blurred_frames': np.asarray(blurred)})

    scale = orig_shape[1] / float(vid.shape[2])
    info = {'name': name,
            'bf_file': video['bf_file'],
            'marker_file': video.get('marker_file'),
            'original_shape': [int(s) for s in orig_shape],
            'shape': [int(s) for s in vid.shape[1:3]],
            'n_frames': int(len(vid)),
            'pixel_res_um': float(cfg.get('pixel_size_um', 1.0)) * scale,
            'time_res_h': float(cfg.get('time_interval_h', 1.0))}
    with open(p['info'], 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    log('saved %d frames of %d x %d pixels (%.3f um / pixel)' % (len(vid), vid.shape[1], vid.shape[2],
                                                                info['pixel_res_um']))


def step_prepare_ctc(cfg, video):
    """ Cell Tracking Challenge sequence: the frames as they are (no resizing / registration, so that they stay
    aligned with the annotation masks) """
    name = video['name']
    p = paths(cfg, name)
    os.makedirs(p['folder'], exist_ok=True)
    log('reading', video['bf_file'])
    vid, files = read_ctc_sequence(video['bf_file'], cfg.get('channel', 0))
    imsave_tif(p['vid'], vid)
    info = {'name': name,
            'bf_file': video['bf_file'],
            'marker_file': None,
            'input_type': 'ctc',
            'original_shape': [int(s) for s in vid.shape[1:3]],
            'shape': [int(s) for s in vid.shape[1:3]],
            'n_frames': int(len(vid)),
            'pixel_res_um': float(cfg.get('pixel_size_um', 1.0)),
            'time_res_h': float(cfg.get('time_interval_h', 1.0))}
    with open(p['info'], 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    log('saved %d frames of %d x %d pixels (%.3f um / pixel)' % (len(vid), vid.shape[1], vid.shape[2],
                                                                info['pixel_res_um']))


def step_detect(cfg, video):
    import SPOT.Detection.detection as SPOT_detection

    p = paths(cfg, video['name'])
    vid = imread_tif(p['vid'])
    if os.path.isdir(p['bbox_ch_folder']):
        shutil.rmtree(p['bbox_ch_folder'])
    os.makedirs(p['bbox_folder'], exist_ok=True)
    d = cfg['detect']
    SPOT_detection.load_and_run_YOLOv3_weights_torch_detection_model_RGB(vid[..., None],
                                                                         detector_weights(cfg),
                                                                         p['bbox_folder'],
                                                                         equalize_hist=bool(d['equalize_hist']),
                                                                         obj_thresh=float(d['obj_thresh']),
                                                                         nms_thresh=float(d['nms_thresh']),
                                                                         imsize=(512, 512),
                                                                         debug_viz=False,
                                                                         device=device_arg(cfg))
    log('detected bounding boxes in %d frames' % len(glob.glob(os.path.join(p['bbox_ch_folder'], '*.txt'))))


def step_track(cfg, video):
    import scipy.io as spio
    import SPOT.Utility_Functions.file_io as fio
    import SPOT.Tracking.track as SPOT_track
    import SPOT.Tracking.optical_flow as SPOT_optical_flow

    p = paths(cfg, video['name'])
    vid = imread_tif(p['vid'])
    log('computing optical flow')
    vid_flow = SPOT_optical_flow.extract_vid_optflow(vid,
                                                     flow_params=SPOT_optical_flow.get_default_optical_flow_params(),
                                                     rescale_intensity=True)
    vid_flow = np.array(vid_flow).astype(np.float32)

    bboxfiles = np.sort(glob.glob(os.path.join(p['bbox_ch_folder'], '*.txt')))
    vid_bboxes = fio.read_detected_bboxes_from_file_for_tracking(bboxfiles)
    t = cfg['track']
    log('tracking bounding boxes')
    tracks_prob, tracks, match_checks, (lens, start_time, lifetime_ratios) = SPOT_track.track_organoid_bbox(
        vid_flow,
        vid_bboxes,
        vid=vid,
        iou_match=float(t['iou_match']),
        ds_factor=1,
        wait_time=int(t['wait_time']),
        min_aspect_ratio=float(t['min_aspect_ratio']),
        max_dense_bbox_cover=int(t['max_dense_bbox_cover']),
        to_viz=False,
        saveanalysisfolder=None)
    spio.savemat(p['tracks'], {'tracks': tracks,
                               'prob_tracks': tracks_prob,
                               'track_quality': match_checks,
                               'track_lens': lens,
                               'track_start_time': start_time,
                               'track_lifetime_ratio': lifetime_ratios})
    log('found %d bounding box tracks' % len(tracks))


_SEGMENT_MODEL = {}


def _segment_model(cfg):
    import SPOT.Detection.detection as SPOT_detection
    key = (segmenter_weights(cfg), device_arg(cfg))
    if key not in _SEGMENT_MODEL:
        _SEGMENT_MODEL[key] = SPOT_detection.load_torch_segmentation_model(key[0], device=key[1])
    return _SEGMENT_MODEL[key]


def step_segment(cfg, video):
    import scipy.io as spio
    import SPOT.Tracking.track as SPOT_track

    p = paths(cfg, video['name'])
    os.makedirs(p['seg_folder'], exist_ok=True)
    vid = imread_tif(p['vid'])
    obj = spio.loadmat(p['tracks'])
    s = cfg['segment']
    pred_size = 64

    bbox_tracks = obj['tracks']
    keep_bbox_tracks = []
    segs, contours = [], []
    keep_track_index = np.zeros(0, dtype=bool)
    if bbox_tracks.size > 0 and bbox_tracks.ndim == 3:
        bbox_tracks_prob = obj['prob_tracks']
        bbox_tracks_lifetime = obj['track_lifetime_ratio'].ravel()
        filter_prob_index = np.nanmean(bbox_tracks_prob, axis=1) >= float(s['filter_prob_thresh'])
        filter_lifetime_index = bbox_tracks_lifetime >= float(s['filter_life_thresh'])
        keep_track_index = np.logical_and(filter_prob_index, filter_lifetime_index)
        log('kept %d of %d tracks (mean probability >= %s, lifetime ratio >= %s)'
            % (keep_track_index.sum(), len(keep_track_index), s['filter_prob_thresh'], s['filter_life_thresh']))

        if np.sum(keep_track_index) > 0:
            keep_bbox_tracks = bbox_tracks[keep_track_index].copy()
            # enforce valid bounding boxes
            bbox_val_size = np.logical_and(keep_bbox_tracks[..., 2] > keep_bbox_tracks[..., 0] + 1,
                                           keep_bbox_tracks[..., 3] > keep_bbox_tracks[..., 1] + 1)
            keep_bbox_tracks[np.logical_not(bbox_val_size > 0), :] = np.nan
            segs, contours = SPOT_track.segment_organoid_bbox_track(vid,
                                                                    keep_bbox_tracks,
                                                                    _segment_model(cfg),
                                                                    segment_size=(pred_size, pred_size),
                                                                    min_I=float(s['min_I']),
                                                                    segment_thresh=float(s['segment_thresh']),
                                                                    clip_boundary_border=10,
                                                                    smooth_boundary_iter=2,
                                                                    boundary_sample_pts=BOUNDARY_SAMPLES,
                                                                    scale_method='scale',
                                                                    invert_intensity=bool(cfg.get('invert_intensity', True)))
    else:
        log('no bounding box tracks to segment')

    spio.savemat(p['seg'], {'boundaries': contours,
                            'boundaries_segs': segs,
                            'cnn_model_path': str(segmenter_weights(cfg)),
                            'pred_size': pred_size,
                            'keep_track_index': keep_track_index,
                            'keep_bbox_tracks': keep_bbox_tracks,
                            'bboxfile': p['tracks'],
                            'filter_prob_thresh': float(s['filter_prob_thresh']),
                            'filter_life_thresh': float(s['filter_life_thresh']),
                            'segment_min_I': float(s['min_I']),
                            'segment_thresh': float(s['segment_thresh'])})
    log('segmented %d organoid tracks' % len(contours))


def read_ctc_tracks(path):
    """ the track table of the Cell Tracking Challenge (man_track.txt / res_track.txt): rows of
    (label, first frame, last frame, parent label) """
    rows = []
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                rows.append([int(v) for v in line.split()[:4]])
    return np.array(rows, dtype=np.int64).reshape(-1, 4)


def ctc_mask_files(seq_folder, segmentation='auto'):
    """ {frame: mask file} of each annotation source of a sequence, in the order they are used.
    'auto': the silver truth segmentation (<seq>_ST/SEG), the gold truth segmentation (<seq>_GT/SEG), then the gold
    truth tracking markers (<seq>_GT/TRA: complete, but often only a small marker inside each cell);
    'tra': the tracking markers only; 'res': a tracker's results in the challenge format (<seq>_RES/mask###.tif) """
    seq_folder = os.path.normpath(seq_folder)

    def frames(pattern):
        out = {}
        for f in glob.glob(pattern):
            m = re.search(r'(\d+)\.tiff?$', os.path.basename(f))
            if m:
                out[int(m.group(1))] = f
        return out
    if segmentation == 'res':
        return [frames(os.path.join(seq_folder + '_RES', 'mask*.tif'))]
    tra = frames(os.path.join(seq_folder + '_GT', 'TRA', 'man_track*.tif'))
    if segmentation == 'tra':
        return [tra]
    return [frames(os.path.join(seq_folder + '_ST', 'SEG', 'man_seg*.tif')),
            frames(os.path.join(seq_folder + '_GT', 'SEG', 'man_seg*.tif')),
            tra]


def ctc_frame_labels(sources, tt, shape, track_ids):
    """ label image of frame tt whose labels are the track labels. A segmentation region gets the label of the
    tracking marker it overlaps most (segmentation labels need not equal track labels); a track without a segmented
    region falls back to the next source, in the end to its tracking marker """
    import tifffile
    present = [np.asarray(tifffile.imread(src[tt])) for src in sources if tt in src]
    if not present:
        return np.zeros(shape, dtype=np.int32)
    if len(present) == 1 or len(sources) == 1:
        return present[0].astype(np.int32)
    tra = present[-1].astype(np.int32) if tt in sources[-1] else None
    if tra is None:
        return present[0].astype(np.int32)
    out = np.zeros(shape, dtype=np.int32)
    todo = set(int(t) for t in np.unique(tra) if t > 0) & set(int(t) for t in track_ids)
    for seg in present[:-1]:
        for tid in sorted(todo):
            labs, counts = np.unique(seg[tra == tid], return_counts=True)
            keep = labs > 0
            if not np.any(keep):
                continue
            region = (seg == labs[keep][np.argmax(counts[keep])]) & (out == 0)
            if region.sum() > 0:
                out[region] = tid
                todo.discard(tid)
    for tid in todo:
        out[(tra == tid) & (out == 0)] = tid
    return out


def mask_outline(mask, n_points=BOUNDARY_SAMPLES, sigma=1.0):
    """ outline of a binary mask as n_points (row, col) points equally spaced along its longest contour """
    import scipy.ndimage as ndimage
    import skimage.measure as skmeasure
    if mask.sum() < 3:
        return None
    rr, cc = np.nonzero(mask)
    pad = int(3 * sigma) + 2
    r0, c0 = max(rr.min() - pad, 0), max(cc.min() - pad, 0)
    crop = np.pad(mask[r0:rr.max() + pad + 1, c0:cc.max() + pad + 1].astype(np.float32), 1)
    if sigma > 0:
        crop = ndimage.gaussian_filter(crop, sigma=sigma)
    contours = skmeasure.find_contours(crop, level=min(0.5, 0.5 * crop.max()))
    if not contours:
        return None
    c = contours[int(np.argmax([len(cc_) for cc_ in contours]))]
    if len(c) < 4:
        return None
    if not np.allclose(c[0], c[-1]):
        c = np.vstack([c, c[:1]])
    arc = np.hstack([0, np.cumsum(np.sqrt(np.sum(np.diff(c, axis=0) ** 2, axis=1)))])
    t = np.linspace(0, arc[-1], n_points + 1)[:-1]
    out = np.stack([np.interp(t, arc, c[:, 0]), np.interp(t, arc, c[:, 1])], axis=1)
    out += np.array([r0 - 1, c0 - 1])
    out[:, 0] = np.clip(out[:, 0], 0, mask.shape[0] - 1)
    out[:, 1] = np.clip(out[:, 1], 0, mask.shape[1] - 1)
    return out


def step_postprocess_ctc(cfg, video):
    """ Cell Tracking Challenge sequence: outline every annotated cell of every track, then remove cells at the image
    border and short tracks. A track ends at a division (the daughters are new tracks), as in the challenge format. """
    import scipy.io as spio
    import SPOT.Tracking.track as SPOT_track

    name = video['name']
    p = paths(cfg, name)
    os.makedirs(p['seg_folder'], exist_ok=True)
    c = cfg['ctc']
    pp = cfg['postprocess']
    segmentation = c.get('segmentation', 'auto')
    track_file = ctc_track_file(video['bf_file'], segmentation)
    if track_file is None:
        raise IOError('no track table for %s (expected %s)' % (
            video['bf_file'], '<seq>_RES/res_track.txt' if segmentation == 'res' else '<seq>_GT/TRA/man_track.txt'))
    tracks = read_ctc_tracks(track_file)
    sources = ctc_mask_files(video['bf_file'], segmentation)
    info = read_info(cfg, name)
    n_frames = int(info['n_frames'])
    shape = tuple(info['shape'])
    log('%d tracks in %s; mask frames per source: %s' % (len(tracks), track_file,
                                                       ', '.join(str(len(s)) for s in sources)))

    track_ids = tracks[:, 0]
    index = {int(t): ii for ii, t in enumerate(track_ids)}
    raw = np.full((len(tracks), n_frames, BOUNDARY_SAMPLES, 2), np.nan)
    min_len = max(int(c.get('min_track_len', 1)), 1)
    for tt in range(n_frames):
        emit('PROGRESS', tt, n_frames)
        alive = track_ids[(tracks[:, 1] <= tt) & (tracks[:, 2] >= tt)]
        if len(alive) == 0:
            continue
        labels = ctc_frame_labels(sources, tt, shape, alive)
        for tid in alive:
            outline = mask_outline(labels == tid, sigma=float(c.get('contour_smooth_sigma', 1.0)))
            if outline is not None:
                raw[index[int(tid)], tt] = outline

    # remove cells touching the image border (as for organoids), then the tracks that became too short
    border = np.zeros(raw.shape[:2], dtype=bool)
    final = raw.copy()
    if len(raw) > 0:
        border, final = SPOT_track.detect_image_border_organoid_tracks(raw,
                                                                      img_shape=shape,
                                                                      border_pad=int(pp['border_pixel_size']),
                                                                      out_percent=float(pp['border_pixel_frac']),
                                                                      prepadded_tracks=True,
                                                                      pad_track_len=BOUNDARY_SAMPLES,
                                                                      apply_mask=True)
        final = np.asarray(final, dtype=np.float64).reshape(raw.shape)
    keep = np.sum(~np.isnan(final[:, :, 0, 0]), axis=1) >= min_len
    final = final[keep]
    log('kept %d of %d tracks (cells at the image border removed, tracks of at least %d frames)'
        % (keep.sum(), len(keep), min_len))
    empty = np.zeros((1, 0, n_frames, BOUNDARY_SAMPLES, 2))
    spio.savemat(p['final'], {'expt': name,
                              'boundaries_raw': raw[None] if len(raw) else empty,
                              'boundaries_smooth_final': final[None] if len(final) else empty,
                              'border_organoids_smooth_final_bool': np.asarray(border),
                              'ctc_track_file': track_file,
                              'ctc_segmentation': segmentation,
                              'ctc_track_table': tracks,
                              'ctc_track_ids_final': track_ids[keep],
                              'ctc_parent_ids_final': tracks[keep, 3],
                              'ds': 1,
                              'boundary_samples': BOUNDARY_SAMPLES,
                              'border_pixel_size': pp['border_pixel_size'],
                              'border_pixel_frac': pp['border_pixel_frac'],
                              'min_track_len': min_len})
    if len(final) == 0:
        with open(p['no_objects'], 'w') as f:
            f.write('no cell tracks were left after removing the cells at the image border\n')
        log('WARNING: no cells left')
    elif os.path.exists(p['no_objects']):
        os.remove(p['no_objects'])
    log('final: %d cell tracks' % len(final))


def step_postprocess(cfg, video):
    import scipy.io as spio
    import SPOT.Tracking.track as SPOT_track

    if is_ctc(cfg):
        return step_postprocess_ctc(cfg, video)
    p = paths(cfg, video['name'])
    vid = imread_tif(p['vid'])
    img_shape = vid.shape[1:3]
    pp = cfg['postprocess']
    n_frames = len(vid)
    empty = np.zeros((1, 0, n_frames, BOUNDARY_SAMPLES, 2))

    raw = spio.loadmat(p['seg'])['boundaries']
    boundaries_raw = boundaries_filter = border_filter = final = breaks = border_final = None

    if raw.size > 0 and raw.ndim == 4:
        boundaries_raw = SPOT_track.filter_nan_tracks(SPOT_track.pad_tracks(raw, boundary_samples=BOUNDARY_SAMPLES))
        if len(boundaries_raw) > 0:
            # 1. suppress unstable / duplicated tracks
            boundaries_filter = SPOT_track.non_stable_track_suppression_filter(vid[..., None],
                                                                               [boundaries_raw],
                                                                               track_overlap_thresh=0.5,
                                                                               weight_nan=0., weight_smooth=0.1,
                                                                               max_obj_frames=10,
                                                                               obj_mean_func=np.nanmean,
                                                                               smoothness_mean_func=np.nanmean,
                                                                               debug_viz=False)[0]
        if boundaries_filter is not None and len(boundaries_filter) > 0:
            border_filter = SPOT_track.detect_image_border_organoid_tracks(boundaries_filter,
                                                                           img_shape=img_shape,
                                                                           border_pad=int(pp['border_pixel_size']),
                                                                           out_percent=float(pp['border_pixel_frac']),
                                                                           prepadded_tracks=True,
                                                                           pad_track_len=BOUNDARY_SAMPLES,
                                                                           apply_mask=False)
            # 2. temporal smoothing
            smooth = SPOT_track.temporal_smooth_tracks(boundaries_filter,
                                                       method='ma',
                                                       win_size=int(pp['temporal_smooth_win_size']),
                                                       prepadded_tracks=True,
                                                       pad_track_len=BOUNDARY_SAMPLES,
                                                       ma_avg_func=np.nanmean,
                                                       ma_pad_mode='edge')
            # 3. split tracks where the shape jumps between frames (low IoU)
            breaks, continuous = SPOT_track.detect_iou_breaks_trackset(smooth,
                                                                      use_bbox=True,
                                                                      iou_thresh=float(pp['iou_consistency_thresh']),
                                                                      prepadded_tracks=True,
                                                                      pad_track_len=BOUNDARY_SAMPLES,
                                                                      split_tracks=True,
                                                                      min_split_track_len=int(pp['min_split_track_len']))
            # 4. remove organoids at the image border
            border_final, final = SPOT_track.detect_image_border_organoid_tracks(continuous,
                                                                                img_shape=img_shape,
                                                                                border_pad=int(pp['border_pixel_size']),
                                                                                out_percent=float(pp['border_pixel_frac']),
                                                                                prepadded_tracks=True,
                                                                                pad_track_len=BOUNDARY_SAMPLES,
                                                                                apply_mask=True)
            final = np.asarray(final)
            if final.size > 0:
                final = SPOT_track.filter_nan_tracks(final)

    def _wrap(b):
        b = None if b is None else np.asarray(b)
        return empty if (b is None or b.size == 0 or b.ndim != 4) else b[None]

    final_arr = _wrap(final)
    spio.savemat(p['final'], {'expt': video['name'],
                              'boundaries_raw': _wrap(boundaries_raw),
                              'boundaries_filter': _wrap(boundaries_filter),
                              'border_organoids_filter_bool': [] if border_filter is None else border_filter,
                              'boundaries_smooth_final': final_arr,
                              'border_organoids_smooth_final_bool': [] if border_final is None else border_final,
                              'boundaries_organoids_smooth_final_breaks': [] if breaks is None else breaks,
                              'ds': 1,
                              'boundary_samples': BOUNDARY_SAMPLES,
                              'border_pixel_size': pp['border_pixel_size'],
                              'border_pixel_frac': pp['border_pixel_frac'],
                              'temporal_smooth_win_size': pp['temporal_smooth_win_size'],
                              'split_tracks_bool': True,
                              'min_split_track_len': pp['min_split_track_len'],
                              'iou_consistency_thresh': pp['iou_consistency_thresh']})
    n = final_arr.shape[1]
    if n == 0:
        with open(p['no_objects'], 'w') as f:
            f.write('no organoid tracks were left after post-processing\n')
        log('WARNING: no organoids left after post-processing')
    elif os.path.exists(p['no_objects']):
        os.remove(p['no_objects'])
    log('final: %d organoid tracks' % n)


# =============================================================================
#   Stage 2
# =============================================================================

def _has_objects(cfg, video):
    p = paths(cfg, video['name'])
    if os.path.exists(p['no_objects']):
        return False
    return load_final_boundaries(cfg, video['name']).shape[0] > 0


def step_features(cfg, video):
    import scipy.io as spio
    import skimage.exposure as skexposure
    import SPOT.Features.features as SPOT_SAM_features
    import SPOT.Tracking.optical_flow as SPOT_optical_flow

    name = video['name']
    p = paths(cfg, name)
    if not _has_objects(cfg, video):
        log('no organoids in this video, skipping')
        return 'none'
    info = read_info(cfg, name)
    pixel_res = info['pixel_res_um']
    time_res = info['time_res_h']
    n_jobs = int(cfg.get('n_jobs', -1))
    use_mask = bool(cfg['features'].get('use_mask', True))
    expt = dataset_name(cfg)

    vid = imread_tif(p['vid'])
    boundaries = load_final_boundaries(cfg, name)
    imshape = vid.shape[1:3]

    def _save(path, metrics, names, norm, extra=None):
        d = {'expt': expt, 'imgfile': p['vid'], 'rgb_rev': False,
             'metric_names': np.array(names), 'metric_norm_bool': np.asarray(norm),
             'metrics': np.asarray(metrics)[None]}
        d.update(extra or {})
        spio.savemat(path, d)

    log('shape features for %d organoids' % len(boundaries))
    metrics, names, norm = SPOT_SAM_features.compute_boundary_morphology_features(boundaries,
                                                                                  imshape,
                                                                                  all_feats_compute=True,
                                                                                  contour_curve_feats=True,
                                                                                  curve_order=4,
                                                                                  curve_err=1.,
                                                                                  geom_features=True,
                                                                                  morph_features=True,
                                                                                  fourier_features=True,
                                                                                  shape_context_features=True,
                                                                                  norm_scm=True,
                                                                                  n_ref_pts_scm=5,
                                                                                  pixel_xy_res=pixel_res,
                                                                                  n_jobs=n_jobs)
    _save(p['shape'], metrics, names, norm, {'pixel_res': pixel_res})

    log('appearance features')
    metrics, names, norm = SPOT_SAM_features.compute_boundary_texture_features(vid,
                                                                               boundaries,
                                                                               use_gradient_vid=False,
                                                                               compute_all_feats=True,
                                                                               compute_intensity_feats=True,
                                                                               compute_contour_intensity_feats=True,
                                                                               n_contours=3,
                                                                               n_angles=1,
                                                                               angle_start=None,
                                                                               compute_sift=True,
                                                                               siftshape=(64, 64),
                                                                               compute_haralick=True,
                                                                               haralick_distance=15,
                                                                               use_mask=use_mask,
                                                                               n_jobs=n_jobs)
    _save(p['image'], metrics, names, norm)

    log('motion features (optical flow)')
    vid_norm = np.array([skexposure.rescale_intensity(frame) for frame in vid])
    flow = SPOT_optical_flow.extract_vid_optflow(vid_norm,
                                                 flow_params=SPOT_optical_flow.get_default_optical_flow_params(),
                                                 rescale_intensity=True)
    metrics, names, norm = SPOT_SAM_features.compute_boundary_motion_features(flow,
                                                                              boundaries,
                                                                              compute_all_feats=True,
                                                                              compute_global_feats=True,
                                                                              compute_contour_feats=True,
                                                                              n_contour_feat_bins=8,
                                                                              cnt_sigma=3.,
                                                                              n_contours=3,
                                                                              n_angles=1,
                                                                              angle_start=None,
                                                                              pixel_res=pixel_res,
                                                                              time_res=time_res,
                                                                              compute_sift_feats=True,
                                                                              siftshape=(64, 64),
                                                                              use_mask=use_mask,
                                                                              n_jobs=n_jobs)
    _save(p['motion'], metrics, names, norm, {'pixel_res': pixel_res, 'time_res': time_res})
    log('saved the SAM phenome of %d organoids x %d frames' % boundaries.shape[:2])


def step_markers(cfg, video):
    """ marker intensity and spot counts inside every organoid (reference script Stage2 Step1b) """
    import scipy.io as spio
    import scipy.ndimage as ndimage
    import skimage.exposure as skexposure
    from skimage.feature import peak_local_max
    from skimage.draw import polygon

    name = video['name']
    p = paths(cfg, name)
    if not (cfg.get('paired_marker') and video.get('marker_file')):
        return 'n/a'
    if not _has_objects(cfg, video):
        log('no organoids in this video, skipping')
        return 'none'

    m = cfg['markers']
    vid_u8 = imread_tif(p['marker_vid'])
    vid_raw = vid_u8.astype(np.float64)
    boundaries = load_final_boundaries(cfg, name)
    n_obj, n_frames = boundaries.shape[:2]
    shape = vid_raw.shape[1:3]

    # contrast stretch each uint8 frame to 0-255, then 0-1 (as the reference script)
    vid_rescale = np.array([skexposure.rescale_intensity(frame) for frame in vid_u8]).astype(np.float64) / 255.
    vid_norm = (vid_raw - np.mean(vid_raw)) / max(np.std(vid_raw), 1e-8)   # z-score over the whole video
    vid_raw01 = vid_raw / 255.

    out = np.full((n_obj, n_frames, len(MARKER_FEATURE_NAMES)), np.nan)
    for tt in range(min(n_frames, len(vid_raw))):
        frame = ndimage.median_filter(vid_rescale[tt], size=3)
        coords = peak_local_max(frame - ndimage.gaussian_filter(frame, sigma=3),
                                min_distance=int(m['spot_min_distance']),
                                threshold_abs=float(m['spot_threshold']))
        spot_I = vid_norm[tt][coords[:, 0], coords[:, 1]] if len(coords) else np.zeros(0)
        spot_I_raw = vid_raw01[tt][coords[:, 0], coords[:, 1]] if len(coords) else np.zeros(0)

        for oo in range(n_obj):
            b = boundaries[oo, tt]
            if np.isnan(b[0, 0]):
                continue
            rr, cc = polygon(b[:, 0], b[:, 1], shape=shape)
            if len(rr) == 0:
                continue
            mask = np.zeros(shape, dtype=bool)
            mask[rr, cc] = True
            inside = mask[coords[:, 0], coords[:, 1]] if len(coords) else np.zeros(0, dtype=bool)
            n_spots = int(np.sum(inside))
            out[oo, tt] = [np.mean(vid_norm[tt][mask]),
                           np.mean(vid_raw01[tt][mask]),
                           n_spots,
                           np.mean(spot_I[inside]) if n_spots > 0 else np.nan,
                           np.mean(spot_I_raw[inside]) if n_spots > 0 else np.nan]

    spio.savemat(p['marker_feats'], {'expt': dataset_name(cfg),
                                     'imgfile': p['marker_vid'],
                                     'rgb_rev': False,
                                     'metric_names': np.array(MARKER_FEATURE_NAMES),
                                     'metrics': out[None]})
    log('measured the marker in %d organoids' % n_obj)


def read_metadata(cfg):
    import pandas as pd
    f = metadata_file(cfg)
    if not os.path.exists(f):
        raise IOError('The sample information table %s does not exist. Fill it in on the "Sample info" tab and save it.' % f)
    table = pd.read_csv(f, encoding='utf-8')
    table['Filename'] = table['Filename'].astype(str)
    return table


def step_compile(cfg, videos):
    """ reference scripts Stage2 Step2-4: compile the per-video SAM phenomes into dataset tables + image patches """
    import pandas as pd
    import scipy.io as spio
    import skimage.transform as sktform
    import SPOT.Utility_Functions.file_io as fio

    table = read_metadata(cfg)
    # SPOT reads the channel as an integer (single channel videos: channel 1)
    table['Img_Channel'] = 1
    table['Img_Channel'] = table['Img_Channel'].astype(np.int64)
    for col in ['Genetics', 'Patients', 'Conditions']:
        table[col] = table[col].astype(object)
        table.loc[table[col].isna() | (table[col].astype(str).str.strip() == ''), col] = 'NA'
        table[col] = table[col].astype(str).str.strip()

    patch_size = (64, 64)
    shape_tables, image_tables, motion_tables, marker_tables = [], [], [], []
    boundaries_all, patches_all, patch_sizes_all, bbox_all = [], [], [], []
    meta_all = {k: [] for k in ['filenames', 'condition', 'genetics', 'patients', 'img_channel_no', 'org_id',
                                'frame_no', 'pixel_res', 'frame_duration']}

    n_used = 0
    for ii, video in enumerate(videos):
        name = video['name']
        p = paths(cfg, name)
        emit('PROGRESS', ii, len(videos))
        if name not in set(table['Filename'].values):
            log('WARNING: %s is not in the sample information table, skipping' % name)
            continue
        if step_status(cfg, video, 'features') != 'done':
            log('skipping %s (%s)' % (name, 'no organoids' if step_status(cfg, video, 'features') == 'none'
                                      else 'SAM features not computed'))
            continue
        info = read_info(cfg, name)
        table.loc[table['Filename'] == name, 'pixel_resolution[um]'] = info['pixel_res_um']
        table.loc[table['Filename'] == name, 'time_resolution[h]'] = info['time_res_h']
        table.loc[table['Filename'] == name, 'n_frames'] = info['n_frames']

        B = load_final_boundaries(cfg, name)
        S = spio.loadmat(p['shape'])
        A = spio.loadmat(p['image'])
        M = spio.loadmat(p['motion'])
        Sm, Am, Mm = [np.asarray(o['metrics'], dtype=np.float64)[0] for o in (S, A, M)]
        n_t = B.shape[1]
        # motion needs the flow to the next frame, so has no value for the last frame(s): pad with nan
        Am, Mm = [np.concatenate([m, np.full((m.shape[0], n_t - m.shape[1], m.shape[2]), np.nan)], axis=1)
                  if m.shape[1] < n_t else m for m in (Am, Mm)]

        # the boundaries, patches and every table share the same rows: keep object instances valid in all
        valid = np.logical_and(~np.isnan(B[:, :, 0, 0]), ~np.isnan(Sm[..., 0]))
        B[~valid] = np.nan
        for arr in (Sm, Am, Mm):
            arr[~valid] = np.nan

        meta, boundaries = fio.compile_boundaries_arrays(name, org_expt_table=table, boundaries=B[None])
        if len(boundaries) == 0:
            continue
        (filenames, conditions, genetics, patients, img_channel_no, org_id, frame_no, pix_res, frame_duration, _) = meta
        for k, v in zip(['filenames', 'condition', 'genetics', 'patients', 'img_channel_no', 'org_id', 'frame_no',
                         'pixel_res', 'frame_duration'],
                        [filenames, conditions, genetics, patients, img_channel_no, org_id, frame_no, pix_res,
                         frame_duration]):
            meta_all[k].append(np.asarray(v))
        boundaries_all.append(boundaries)

        shape_tables.append(fio.construct_metrics_table_csv(name, org_expt_table=table, metrics=Sm[None],
                                                            metricslabels=mat_names(S['metric_names'])))
        image_tables.append(fio.construct_metrics_table_csv(name, org_expt_table=table, metrics=Am[None],
                                                            metricslabels=mat_names(A['metric_names'])))
        motion_tables.append(fio.construct_metrics_table_csv(name, org_expt_table=table, metrics=Mm[None],
                                                             metricslabels=mat_names(M['metric_names'])))
        if os.path.exists(p['marker_feats']):
            K = spio.loadmat(p['marker_feats'])
            Km = np.asarray(K['metrics'], dtype=np.float64)[0]
            Km[~valid] = np.nan
            # keep rows of valid organoids even where the marker could not be measured
            Km_out = Km.copy()
            Km_out[..., 0][valid & np.isnan(Km[..., 0])] = -9999.
            mt = fio.construct_metrics_table_csv(name, org_expt_table=table, metrics=Km_out[None],
                                                 metricslabels=mat_names(K['metric_names']))
            mt = mt.replace({MARKER_FEATURE_NAMES[0]: {-9999.: np.nan, '-9999.0': np.nan}})
            marker_tables.append(mt)

        # image patches of every object instance (reference Stage2 Step4)
        vid = imread_tif(p['vid'])
        for bb, fr in zip(boundaries, frame_no):
            frame = vid[int(fr) - 1]
            x0 = int(np.clip(np.floor(np.min(bb[:, 1])), 0, frame.shape[1] - 1))
            y0 = int(np.clip(np.floor(np.min(bb[:, 0])), 0, frame.shape[0] - 1))
            x1 = int(np.clip(np.ceil(np.max(bb[:, 1])), x0 + 1, frame.shape[1]))
            y1 = int(np.clip(np.ceil(np.max(bb[:, 0])), y0 + 1, frame.shape[0]))
            patch = frame[y0:y1, x0:x1]
            bbox_all.append([x0, y0, x1, y1])
            patch_sizes_all.append(patch.shape)
            patches_all.append(np.uint8(sktform.resize(patch, patch_size, order=1, preserve_range=True)))
        n_used += 1
        log('added %s: %d organoid instances' % (name, len(boundaries)))

    if n_used == 0:
        raise RuntimeError('No video has computed SAM features and an entry in the sample information table.')

    d = dataset_paths(cfg)
    pd.concat(shape_tables, ignore_index=True).to_csv(d['shape'], index=None)
    pd.concat(image_tables, ignore_index=True).to_csv(d['image'], index=None)
    pd.concat(motion_tables, ignore_index=True).to_csv(d['motion'], index=None)
    if marker_tables:
        pd.concat(marker_tables, ignore_index=True).to_csv(d['marker'], index=None)
    elif os.path.exists(d['marker']):
        os.remove(d['marker'])

    spio.savemat(d['boundaries'], {k: np.hstack(v) for k, v in meta_all.items()} |
                 {'boundaries': np.concatenate(boundaries_all, axis=0)})
    spio.savemat(d['patches'], {'std_size': patch_size,
                                'patches_all': np.array(patches_all),
                                'bbox_size': np.vstack(bbox_all),
                                'patch_size': np.vstack(patch_sizes_all)})
    # keep the filled in resolutions / frame numbers in the sample table
    table.to_csv(metadata_file(cfg), index=None, encoding='utf-8')
    log('compiled %d videos, %d organoid instances' % (n_used, len(patches_all)))


# =============================================================================
#   Stage 3: analysis
# =============================================================================

def _install_png_mirror(save_vector):
    """ SPOT's plotting functions save .svg files: also save a .png copy for the GUI (and drop the .svg if unwanted) """
    import matplotlib.figure as mfigure
    if getattr(mfigure.Figure.savefig, '_spot_mirror', False):
        return
    orig = mfigure.Figure.savefig

    def savefig(self, fname, *args, **kwargs):
        if isinstance(fname, str) and os.path.splitext(fname)[1].lower() in ('.svg', '.pdf'):
            kw = dict(kwargs)
            kw['dpi'] = 150
            kw.pop('format', None)
            orig(self, os.path.splitext(fname)[0] + '.png', *args, **kw)
            if not save_vector:
                return None
        return orig(self, fname, *args, **kwargs)

    savefig._spot_mirror = True
    mfigure.Figure.savefig = savefig


def run_analysis(cfg):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    import warnings
    obj = 'cell' if is_ctc(cfg) else 'organoid'   # wording of the logs and plots
    warnings.filterwarnings('ignore')
    from sklearn.preprocessing import StandardScaler, power_transform

    import SPOT.Utility_Functions.file_io as fio
    import SPOT.Analysis.preprocessing as preprocess
    import SPOT.Analysis.sam_analysis as SAM
    import SPOT.Utility_Functions.plotting as plotting

    A = cfg['analysis']
    save_vector = bool(A.get('save_vector', False))
    _install_png_mirror(save_vector)
    expt = dataset_name(cfg)
    out = cfg['output_folder']
    resdir = analysis_folder(cfg)
    if os.path.isdir(resdir):
        shutil.rmtree(resdir)
    os.makedirs(resdir, exist_ok=True)
    dirs = {k: os.path.join(resdir, v) for k, v in {
        'qc': '00_feature_overview', 'modules': '01_SAM_modules', 'embed': '02_phenotype_map',
        'density': '03_condition_density', 'clusters': '04_phenotype_clusters', 'time': '05_time_course',
        'traj': '06_phenotype_trajectories', 'hmm': '07_phenotype_transitions', 'markers': '08_markers'}.items()}
    for v in dirs.values():
        os.makedirs(v, exist_ok=True)
    summary = {'dataset': expt, 'started': time.strftime('%Y-%m-%d %H:%M:%S'), 'warnings': []}
    n_sections = 10
    section = [0]

    def stage(msg):
        section[0] += 1
        emit('PROGRESS', section[0], n_sections)
        log('--- ' + msg)

    def save(fig, folder, name):
        fig.savefig(os.path.join(dirs[folder], name + '.png'), dpi=150, bbox_inches='tight')
        if save_vector:
            fig.savefig(os.path.join(dirs[folder], name + '.svg'), bbox_inches='tight')
        plt.close(fig)

    def attempt(label, fn):
        try:
            return fn()
        except Exception as e:
            msg = '%s failed: %s' % (label, e)
            log('WARNING: ' + msg)
            log(traceback.format_exc())
            summary['warnings'].append(msg)
            plt.close('all')
            return None

    # ------------------------------------------------------------------ load
    stage('loading the compiled SAM phenome')
    d = dataset_paths(cfg)
    all_feats, metadict = fio.load_SPOT_features_files([out], [expt],
                                                       boundaryfile_suffix='_boundaries_smooth_final.mat',
                                                       boundarykey='boundaries',
                                                       patchfile_suffix='_boundaries_smooth_final_img_patches.mat',
                                                       patches_key='patches_all',
                                                       patches_size_key='patch_size',
                                                       shapefeatsfile_suffix='_shape_features.csv',
                                                       appearfeatsfile_suffix='_image_features.csv',
                                                       motionfeatsfile_suffix='_motion_features.csv',
                                                       read_chunksize=2000)
    X = all_feats.astype(np.float64)
    feature_names = np.array([str(f) for f in metadict['feature_names']])
    patches = np.asarray(metadict['all_object_patches']).astype(np.uint8)
    patch_sizes = metadict['all_object_patches_sizes']
    TP = np.asarray(metadict['all_object_TP']).astype(np.float64)
    Ts = np.asarray(metadict['all_object_Ts']).astype(np.float64)
    N = len(X)

    # object ids: <expt>_<Filename>_<channel>_<org>_<frame>
    uids = metadict['all_object_uniq_row_ids']
    parsed = [u[len(expt) + 1:].rsplit('_', 3) for u in uids]
    obj_file = np.array([pp[0] for pp in parsed])
    obj_ch = np.array([int(pp[1]) for pp in parsed])
    obj_org = np.array([int(pp[2]) for pp in parsed])
    obj_frame = np.array([int(pp[3]) for pp in parsed])

    meta = read_metadata(cfg).set_index('Filename')

    def meta_column(col):
        if not col or col == 'Filename':
            return obj_file.copy()
        if col not in meta.columns:
            return np.array(['NA'] * N)
        vals = meta[col].astype(object).where(meta[col].notna(), 'NA').astype(str).str.strip()
        vals = vals.replace('', 'NA')
        return np.array([safe_label(vals.get(f, 'NA')) for f in obj_file])

    group_col = A.get('group_column') or 'Conditions'
    groups = meta_column(group_col)
    uniq_groups = np.unique(groups)
    batch_col = A.get('batch_column') or ''
    batches = meta_column(batch_col) if batch_col else None
    log('%d %s instances from %d videos, %d features; %d groups (%s): %s'
        % (N, obj, len(np.unique(obj_file)), X.shape[1], len(uniq_groups), group_col, ', '.join(uniq_groups)))
    summary.update({'n_objects': int(N), 'n_videos': int(len(np.unique(obj_file))), 'group_column': group_col,
                    'groups': [str(g) for g in uniq_groups], 'n_features_raw': int(X.shape[1])})

    # markers (not part of the SAM phenome, used to annotate)
    marker_feats = None
    if os.path.exists(d['marker']):
        mt = pd.read_csv(d['marker'], encoding='utf-8')
        key = mt['Filename'].astype(str) + '_' + mt['Img_Channel_No'].astype(str) + '_' + \
            mt['Org_ID'].astype(str) + '_' + mt['Frame_No'].astype(str)
        mt.index = key.values
        obj_key = np.array(['%s_%d_%d_%d' % (f, c, o, t) for f, c, o, t in zip(obj_file, obj_ch, obj_org, obj_frame)])
        marker_feats = mt.reindex(obj_key)[MARKER_FEATURE_NAMES].values.astype(np.float64)
        log('loaded marker measurements for %d objects' % np.sum(~np.isnan(marker_feats[:, 0])))
    marker_groups = meta_column(A.get('marker_group_column') or 'Marker') if marker_feats is not None else None

    # ------------------------------------------------------------------ preprocessing
    stage('normalising features')
    X, feature_names = preprocess.scale_normalize_curvature_features(X, feature_names)
    X, feature_names = preprocess.remove_zero_features(X, feature_names)
    X, feature_names = preprocess.remove_nan_variance_features(X, feature_names)
    X, feature_names = preprocess.remove_high_variance_features(X, feature_names, variance_threshold_sigma=2)
    feature_names = np.asarray(feature_names)
    res = attempt('kernel reduction of the ECT features',
                  lambda: preprocess.kernel_dim_reduction_ECC_features(X.astype(np.float64), feature_names, n_dim=100,
                                                                       gamma=None, random_state=1))
    if res is not None:
        X, feature_names = res[0], np.asarray(res[1])

    if A.get('normalise_per_batch') and batches is not None:
        Xn = np.zeros_like(X)
        for b in np.unique(batches):
            sel = batches == b
            Xn[sel] = power_transform(StandardScaler().fit_transform(X[sel]))
        X = Xn
    else:
        X = power_transform(StandardScaler().fit_transform(X))

    if A.get('time_varying_selection', True) and len(np.unique(TP)) > 2:
        res = attempt('selection of time-varying features',
                      lambda: preprocess.select_time_varying_features(X.astype(np.float64), feature_names, TP,
                                                                      ridge_alpha=1.))
        if res is not None and len(res[1]) >= 10:
            X, feature_names = res[0], np.asarray(res[1])
    n_nan = int(np.sum(np.isnan(X)))
    if n_nan:
        log('%d missing feature values set to the mean (0) after normalisation' % n_nan)
        X[np.isnan(X)] = 0.

    if batches is not None and A.get('regress_batch', True) and len(np.unique(batches)) > 1:
        res = attempt('regressing out the batch',
                      lambda: preprocess.regress_out_confounders(X.astype(np.float64), feature_names=feature_names,
                                                                 categorical={'batch': batches}))
        if res is not None:
            X, feature_names = res[0], np.asarray(res[1])
            X[np.isnan(X)] = 0.
    log('%d features after normalisation and selection' % X.shape[1])
    summary['n_features'] = int(X.shape[1])

    embed_feats = X
    if A.get('equal_weighting', True):
        embed_feats = attempt('equal weighting of the SAM sets',
                              lambda: preprocess.weight_sam_features(X.astype(np.float64), feature_names,
                                                                     group_weighting=True, set_weighting=True))
        if embed_feats is None:
            embed_feats = X

    def find_feature(pattern):
        ind = [i for i, n in enumerate(feature_names) if pattern in n]
        return ind[0] if ind else None

    # ------------------------------------------------------------------ SAM contributions + modules
    stage('SAM feature contributions and SAM modules')
    feature_type, feature_scope = SAM.get_SAM_feature_type(feature_names)

    def _contrib():
        (ftype, fscope), _, (SAM_contribution, scope_contribution) = \
            SAM.compute_SAM_features_and_scope_contribution(X, feature_names, random_state=0)
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
        ax[0].bar(['Shape', 'Appearance', 'Motion'], np.ravel(SAM_contribution)[:3], color=sns.color_palette('Spectral', 3))
        ax[0].set_title('Contribution of the SAM sets')
        ax[1].bar(['Global', 'Local regional', 'Local distribution'], np.ravel(scope_contribution)[:3],
                  color=sns.color_palette('coolwarm_r', 3))
        ax[1].set_title('Contribution of the feature scopes')
        for a in ax:
            a.tick_params(axis='x', rotation=20)
            a.spines[['right', 'top']].set_visible(False)
        save(fig, 'qc', 'SAM_contribution')
        summary['SAM_contribution'] = [float(v) for v in np.ravel(SAM_contribution)[:3]]
    attempt('SAM contribution', _contrib)

    module_expr = None
    module_names = []

    def _modules():
        scale = A.get('module_scale', 'best')
        try:
            scale = int(scale)
        except (TypeError, ValueError):
            pass
        res = SAM.hierarchical_cluster_features_into_SAM_modules(X, feature_names,
                                                                 feature_type=feature_type,
                                                                 feature_scope=feature_scope,
                                                                 hcluster_heatmap_color='vlag',
                                                                 hcluster_method='average',
                                                                 hcluster_metric='euclidean',
                                                                 debugviz=False,
                                                                 savefile=os.path.join(dirs['modules'], 'SAM_module_heatmap.svg'),
                                                                 module_selection='multiscale',
                                                                 scale=scale,
                                                                 return_scales=True)
        clustermap, modules, colorbar, mod_featnames, mod_indices = res[:5]
        plt.close('all')
        if len(res) > 5 and res[5] is not None:
            res[5].to_csv(os.path.join(dirs['modules'], 'SAM_module_scales.csv'))
        expr, contrib = SAM.compute_individual_feature_contributions_in_SAM_modules(X, mod_indices, random_state=0)
        rows = []
        ftype_name = {0: 'Shape', 1: 'Appearance', 2: 'Motion'}
        for ii in range(len(mod_featnames)):
            order = np.argsort(np.abs(contrib[ii]))[::-1]
            for jj in order:
                rows.append({'Module': ii + 1, 'Feature': mod_featnames[ii][jj],
                             'SAM_set': ftype_name.get(int(feature_type[mod_indices[ii][jj]]), ''),
                             'Contribution': contrib[ii][jj]})
            top = order[:25]
            fig, ax = plt.subplots(figsize=(max(4, 0.32 * len(top) + 2), 4))
            ax.bar(np.arange(len(top)), np.asarray(contrib[ii])[top], width=0.6, color='steelblue')
            ax.set_xticks(np.arange(len(top)))
            ax.set_xticklabels(np.asarray(mod_featnames[ii])[top], rotation=90, fontsize=8)
            ax.set_title('Module %d: %d features (top %d shown)' % (ii + 1, len(order), len(top)))
            ax.set_ylabel('Contribution')
            ax.spines[['right', 'top']].set_visible(False)
            save(fig, 'modules', 'Module_%02d_feature_contributions' % (ii + 1))
        pd.DataFrame(rows).to_csv(os.path.join(dirs['modules'], 'SAM_module_features.csv'), index=None)

        # most representative organoids of each module
        _, sample_images, _ = SAM.compute_most_representative_image_patches(expr, patches, n_rows=3, n_cols=3,
                                                                            rescale_intensity=False)
        n_mod = len(sample_images)
        ncol = min(n_mod, 5)
        nrow = int(np.ceil(n_mod / float(ncol)))
        fig, axs = plt.subplots(nrow, ncol, figsize=(3 * ncol, 3.2 * nrow), squeeze=False)
        for ii, a in enumerate(axs.ravel()):
            a.axis('off')
            if ii < n_mod:
                a.imshow(np.uint8(sample_images[ii]), cmap='gray')
                a.set_title('Module %d' % (ii + 1))
        save(fig, 'modules', 'Module_representative_organoids')
        summary['n_modules'] = int(n_mod)
        log('found %d SAM modules' % n_mod)
        return expr
    module_expr = attempt('SAM module analysis', _modules)
    if module_expr is not None:
        module_names = ['Module_%d' % (ii + 1) for ii in range(module_expr.shape[1])]

    # ------------------------------------------------------------------ embedding
    stage('computing the 2D phenotype map (%s)' % A.get('embedding', 'UMAP'))
    if A.get('embedding', 'UMAP').upper() == 'PACMAP':
        import pacmap
        uu = pacmap.PaCMAP(n_components=2, random_state=0).fit_transform(embed_feats.astype(np.float32))
    else:
        import umap
        nn = int(max(2, min(int(A.get('n_neighbors', 100)), N - 1)))
        uu = umap.UMAP(n_neighbors=nn, random_state=0, metric='euclidean').fit_transform(embed_feats.astype(np.float64))
    emb_label = 'UMAP' if A.get('embedding', 'UMAP').upper() != 'PACMAP' else 'PaCMAP'

    def scatter_map(values=None, title='', cmap='coolwarm', vmin=None, vmax=None, ax=None, s=4, cbar=True):
        own = ax is None
        if own:
            fig, ax = plt.subplots(figsize=(6.5, 6))
        ax.scatter(uu[:, 0], uu[:, 1], s=s, c='lightgrey', lw=0)
        if values is not None:
            ok = ~np.isnan(values)
            sc = ax.scatter(uu[ok, 0], uu[ok, 1], s=s, c=values[ok], cmap=cmap, vmin=vmin, vmax=vmax, lw=0)
            if cbar:
                plt.colorbar(sc, ax=ax, shrink=0.7)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(emb_label + ' 1')
        ax.set_ylabel(emb_label + ' 2')
        return ax.figure if own else None

    def _patches_map():
        fig, ax = plt.subplots(figsize=(12, 12))
        sub = max(1, int(np.ceil(N / float(A.get('max_patches_on_embedding', 1500)))))
        plotting.plot_image_patches(positions=uu, patches=patches, ax=ax, subsample=sub, zoom=0.25)
        ax.axis('off')
        save(fig, 'embed', 'phenotype_map_organoid_images')
    attempt('organoid images on the map', _patches_map)

    def _group_map():
        colors = sns.color_palette('tab10' if len(uniq_groups) <= 10 else 'husl', len(uniq_groups))
        fig, ax = plt.subplots(figsize=(7.5, 6))
        for g, c in zip(uniq_groups, colors):
            sel = groups == g
            ax.scatter(uu[sel, 0], uu[sel, 1], s=4, color=c, lw=0, label='%s (n=%d)' % (g, sel.sum()))
        ax.legend(markerscale=4, fontsize=8, bbox_to_anchor=(1.01, 1), loc='upper left', frameon=False)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title('%ss coloured by %s' % (obj.capitalize(), group_col))
        save(fig, 'embed', 'phenotype_map_by_%s' % safe_name(group_col))
        fig = scatter_map(TP * Ts, 'Time (h)', cmap='viridis')
        save(fig, 'embed', 'phenotype_map_by_time')
    attempt('map coloured by group', _group_map)

    def _feature_maps():
        for pattern, label in [('equivalent_diameter', 'Size (equivalent diameter)'),
                               ('major_minor_axis_ratio', 'Elongation (major / minor axis)'),
                               ('mean_speed_global', 'Mean speed'),
                               ('mean_intensity', 'Mean intensity')]:
            ind = find_feature(pattern)
            if ind is not None:
                fig = scatter_map(X[:, ind], label + '\n(%s, normalised)' % feature_names[ind], vmin=-3, vmax=3)
                save(fig, 'embed', 'phenotype_map_%s' % pattern)
    attempt('feature maps', _feature_maps)

    def _module_maps():
        n_mod = module_expr.shape[1]
        ncol = min(n_mod, 4)
        nrow = int(np.ceil(n_mod / float(ncol)))
        fig, axs = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 4 * nrow), squeeze=False)
        for ii, a in enumerate(axs.ravel()):
            if ii < n_mod:
                scatter_map(module_expr[:, ii], 'Module %d' % (ii + 1), vmin=-6, vmax=6, ax=a, s=2)
            else:
                a.axis('off')
        save(fig, 'modules', 'Module_scores_on_phenotype_map')
    if module_expr is not None:
        attempt('module maps', _module_maps)

    # ------------------------------------------------------------------ density per group
    stage('density of each %s on the map' % group_col)
    attempt('condition density', lambda: SAM.compute_heatmap_density_image_Embedding(
        uu, all_conditions=groups, unique_conditions=uniq_groups, cmap='coolwarm', grid_scale_factor=500,
        sigma_factor=0.25, contour_sigma_levels=[1, 2, 3, 3.5, 4, 4.5, 5], saveplotsfolder=dirs['density']))
    plt.close('all')

    # ------------------------------------------------------------------ phenotype clusters
    stage('phenotype clusters')
    from sklearn.cluster import KMeans
    k_setting = str(A.get('n_clusters', 'auto')).strip().lower()
    if k_setting in ('', 'auto'):
        k_lo = max(2, int(A.get('k_min', 2)))
        k_hi = max(k_lo + 1, min(int(A.get('k_max', 20)), N - 1))
        clust_labels, clusterer, best_k, scan = SAM.KMeans_cluster_low_dimensional_embedding_with_scan(
            uu, k_range=(k_lo, k_hi), ploton=False)
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.plot(np.arange(k_lo, k_lo + len(scan)), scan, 'o-')
        ax.set_xlabel('Number of clusters K'); ax.set_ylabel('Adjusted inertia (lower = better)')
        ax.set_title('Automatic choice of K: %d' % best_k)
        save(fig, 'clusters', 'choice_of_number_of_clusters')
    else:
        clusterer = KMeans(n_clusters=int(k_setting), random_state=0, n_init=10).fit(uu)
    clust_labels = clusterer.predict(uu)
    ecc_ind = find_feature('major_minor_axis_ratio')
    if ecc_ind is not None:
        clust_labels, _ = SAM.reorder_and_relabel_clusters(clust_labels, score=X[:, ecc_ind], order='ascending')
    clust_labels = np.asarray(clust_labels).astype(int)
    uniq_clusters = np.unique(clust_labels)
    K = len(uniq_clusters)
    clust_colors = sns.color_palette('rainbow', K)
    summary['n_clusters'] = int(K)
    log('%d phenotype clusters%s' % (K, ' (ordered by elongation)' if ecc_ind is not None else ''))

    fig, ax = plt.subplots(figsize=(7, 6.5))
    for ii, lab in enumerate(uniq_clusters):
        sel = clust_labels == lab
        ax.scatter(uu[sel, 0], uu[sel, 1], s=4, color=clust_colors[ii], lw=0)
        m = np.median(uu[sel], axis=0)
        ax.text(m[0], m[1], str(ii + 1), fontsize=18, ha='center', va='center', weight='bold',
                bbox=dict(boxstyle='circle', fc='white', alpha=0.7, lw=0))
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title('%d phenotype clusters' % K)
    save(fig, 'clusters', 'phenotype_clusters')

    def _cluster_images():
        rep = SAM.find_representative_images_per_cluster_PCA(X, uu, patches, patch_sizes, clust_labels,
                                                             unique_cluster_labels=uniq_clusters, rev_sign=False,
                                                             mosaic_size=0.98, percentile_cutoff=0.75,
                                                             n_rows_mosaic=2, n_cols_mosaic=2, pca_random_state=0,
                                                             debugviz=False)
        imgs = rep['representative_images_std_size']
        ncol = min(K, 6)
        nrow = int(np.ceil(K / float(ncol)))
        fig, axs = plt.subplots(nrow, ncol, figsize=(2.8 * ncol, 3.1 * nrow), squeeze=False)
        for ii, a in enumerate(axs.ravel()):
            a.axis('off')
            if ii < len(imgs):
                a.imshow(imgs[ii], cmap='gray')
                a.set_title('Cluster %d' % (ii + 1), color=clust_colors[ii], weight='bold')
        save(fig, 'clusters', 'phenotype_cluster_representative_organoids')
    attempt('representative organoids per cluster', _cluster_images)

    def _cluster_modules():
        _, means, _ = SAM.compute_mean_score_features_cluster(clust_labels, all_feats=module_expr, avg_func=np.nanmean)
        df = pd.DataFrame(means, index=['Cluster %d' % (ii + 1) for ii in range(K)],
                          columns=['Module %d' % (ii + 1) for ii in range(module_expr.shape[1])])
        df.to_csv(os.path.join(dirs['clusters'], 'mean_module_score_per_cluster.csv'))
        fig, ax = plt.subplots(figsize=(1 + 0.7 * df.shape[1], 1 + 0.5 * K))
        lim = np.nanmax(np.abs(df.values)) or 1
        sns.heatmap(df, cmap='vlag', center=0, vmin=-lim, vmax=lim, annot=df.shape[1] <= 12, fmt='.1f', ax=ax,
                    cbar_kws={'label': 'mean module score'})
        ax.set_title('SAM module signature of each phenotype cluster')
        save(fig, 'clusters', 'module_signature_per_cluster')
    if module_expr is not None:
        attempt('module signature per cluster', _cluster_modules)

    def _cluster_props():
        counts = pd.crosstab(pd.Series(groups, name=group_col), pd.Series(clust_labels + 1, name='Cluster'))
        props = counts.div(counts.sum(axis=1), axis=0)
        counts.to_csv(os.path.join(dirs['clusters'], 'cluster_counts_per_%s.csv' % safe_name(group_col)))
        props.to_csv(os.path.join(dirs['clusters'], 'cluster_proportions_per_%s.csv' % safe_name(group_col)))
        fig, ax = plt.subplots(figsize=(1.5 + 0.8 * len(props), 4.5))
        bottom = np.zeros(len(props))
        for ii, c in enumerate(props.columns):
            ax.bar(np.arange(len(props)), props[c].values, bottom=bottom, color=clust_colors[ii], ec='k', lw=0.5,
                   label='Cluster %d' % c)
            bottom += props[c].values
        ax.set_xticks(np.arange(len(props)))
        ax.set_xticklabels(props.index, rotation=45, ha='right')
        ax.set_ylabel('Fraction of %s instances' % obj); ax.set_ylim(0, 1)
        ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', frameon=False, fontsize=8)
        ax.spines[['right', 'top']].set_visible(False)
        save(fig, 'clusters', 'cluster_proportions_per_%s' % safe_name(group_col))
    attempt('cluster proportions', _cluster_props)

    # ------------------------------------------------------------------ time course
    stage('cluster composition over time')
    n_bins = int(A.get('n_time_bins', 16))
    obj_time = TP * Ts

    def _temporal():
        t_int = np.linspace(0, np.max(obj_time), n_bins + 1)
        _, prop = SAM.compute_temporal_cluster_proportions(clust_labels, objects_time=obj_time, time_intervals=t_int,
                                                          all_conditions=groups, unique_conditions=uniq_groups)
        plotting.stacked_barplot_temporal_cluster_proportions(prop, unique_conditions=uniq_groups, time_intervals=t_int,
                                                             clust_labels_colors=clust_colors, figsize=(7, 5),
                                                             saveplotsfolder=dirs['time'])
        plt.close('all')
    attempt('temporal cluster proportions', _temporal)

    # ------------------------------------------------------------------ trajectories
    stage('average phenotype trajectory of each %s' % group_col)
    trajs = None

    def _traj():
        tr, _ = SAM.compute_phenotypic_trajectory(uu, objects_time=TP,
                                                  time_intervals=np.linspace(0, np.max(TP), n_bins + 1),
                                                  all_conditions=groups, unique_conditions=uniq_groups, cmap='coolwarm',
                                                  grid_scale_factor=500, sigma_factor=0.25,
                                                  thresh_density_sigma=float(A.get('trajectory_density_sigma', 2.)),
                                                  debugviz=False)
        plt.close('all')
        colors = sns.color_palette('tab10' if len(uniq_groups) <= 10 else 'husl', len(uniq_groups))
        fig_all, ax_all = plt.subplots(figsize=(7.5, 6.5))
        ax_all.scatter(uu[:, 0], uu[:, 1], s=4, c='lightgrey', lw=0)
        rows = []
        for jj, g in enumerate(uniq_groups):
            t = np.asarray(tr[jj])
            fig, ax = plt.subplots(figsize=(6.5, 6))
            ax.scatter(uu[:, 0], uu[:, 1], s=4, c='lightgrey', lw=0)
            ax.plot(t[:, 0], t[:, 1], 'ko-', lw=2.5, ms=4)
            ax.plot(t[0, 0], t[0, 1], 'o', color='limegreen', ms=12, mec='k', mew=2, label='start')
            ax.plot(t[-1, 0], t[-1, 1], 's', color='red', ms=11, mec='k', mew=2, label='end')
            ax.legend(frameon=False)
            ax.set_title('Phenotype trajectory: %s' % g)
            ax.set_xticks([]); ax.set_yticks([])
            save(fig, 'traj', 'trajectory_%s' % safe_name(g))
            ax_all.plot(t[:, 0], t[:, 1], 'o-', color=colors[jj], lw=2.5, ms=3, label=g)
            ax_all.plot(t[0, 0], t[0, 1], 'o', color=colors[jj], ms=10, mec='k', mew=1.5)
            for kk in range(len(t)):
                rows.append({group_col: g, 'time_bin': kk + 1, emb_label + '_1': t[kk, 0], emb_label + '_2': t[kk, 1]})
        ax_all.legend(bbox_to_anchor=(1.01, 1), loc='upper left', frameon=False, fontsize=8)
        ax_all.set_xticks([]); ax_all.set_yticks([])
        ax_all.set_title('Phenotype trajectories (large dot = start)')
        save(fig_all, 'traj', 'trajectories_all')
        pd.DataFrame(rows).to_csv(os.path.join(dirs['traj'], 'trajectories.csv'), index=None)
        return tr
    trajs = attempt('phenotype trajectories', _traj)

    if trajs is not None and len(uniq_groups) >= 3:
        def _traj_cluster():
            aff = SAM.construct_affinity_matrix_phenotypic_trajectory(trajs, affinity='dtw', use_std_denom=True)
            pd.DataFrame(aff, index=uniq_groups, columns=uniq_groups).to_csv(
                os.path.join(dirs['traj'], 'trajectory_similarity_dtw.csv'))
            SAM.hcluster_phenotypic_trajectory_affinity_matrix(aff, uniq_conditions=np.asarray(uniq_groups),
                                                               linkage='average', metric='euclidean', cmap='vlag',
                                                               figsize=(8, 8), save_dpi=150, savefolder=dirs['traj'])
            plt.close('all')
        attempt('clustering of the trajectories', _traj_cluster)

    # ------------------------------------------------------------------ HMM transitions
    stage('phenotype transitions (hidden Markov model)')
    if A.get('hmm', True):
        def _hmm():
            file_index = {f: ii for ii, f in enumerate(np.unique(obj_file))}
            ids = np.array(['%d_%d_%d_%d' % (file_index[f], c, o, t)
                            for f, c, o, t in zip(obj_file, obj_ch, obj_org, obj_frame)])
            traj_dict = SAM.construct_obj_traj_from_uniq_obj_ids(ids, separator='_', wanted_entries=[0, 1, 2], d_Frame=1)
            label_trajs = SAM.get_labels_for_trajectory(traj_dict['all_obj_trajectories'], clust_labels,
                                                        all_conditions=groups, all_unique_conditions=uniq_groups)
            from scipy.stats import entropy
            for jj, g in enumerate(uniq_groups):
                seqs = [s for s in label_trajs[jj] if len(s) > 1]
                if len(seqs) == 0:
                    log('no %s tracked over 2+ frames for %s, no transitions' % (obj, g))
                    continue

                def _one():
                    models, _ = SAM.fit_categoricalHMM_model_to_phenotype_cluster_label_trajectories(
                        [seqs], clust_labels, hmm_algorithm='map', hmm_random_state=0, hmm_implementation='scaling')
                    Q = np.asarray(models[0][0])
                    names = ['Cluster %d' % (ii + 1) for ii in range(len(Q))]
                    pd.DataFrame(Q, index=names, columns=names).to_csv(
                        os.path.join(dirs['hmm'], 'transition_matrix_%s.csv' % safe_name(g)))
                    SAM.draw_HMM_transition_graph(Q, ax=None, node_colors=clust_colors, edgescale=10,
                                                  edgelabelpos=.5, figsize=(9, 9),
                                                  savefile=os.path.join(dirs['hmm'], 'transition_graph_%s.svg' % safe_name(g)))
                    plt.close('all')
                    fig, ax = plt.subplots(figsize=(1.5 + 0.6 * len(Q), 1 + 0.5 * len(Q)))
                    sns.heatmap(pd.DataFrame(Q, index=names, columns=names), cmap='magma', vmin=0, vmax=1,
                                annot=len(Q) <= 10, fmt='.2f', ax=ax, cbar_kws={'label': 'probability'})
                    ax.set_xlabel('to'); ax.set_ylabel('from')
                    ax.set_title('Phenotype transitions: %s' % g)
                    save(fig, 'hmm', 'transition_matrix_%s' % safe_name(g))
                    ent = [entropy(r) for r in Q]
                    fig, ax = plt.subplots(figsize=(1.5 + 0.5 * len(Q), 3.5))
                    ax.bar(np.arange(1, len(Q) + 1), ent, color=clust_colors, ec='k')
                    ax.set_xlabel('Cluster'); ax.set_ylabel('Transition entropy')
                    ax.set_title('Plasticity of each phenotype: %s' % g)
                    ax.spines[['right', 'top']].set_visible(False)
                    save(fig, 'hmm', 'transition_entropy_%s' % safe_name(g))
                attempt('HMM for %s' % g, _one)
        attempt('phenotype transitions', _hmm)

    # ------------------------------------------------------------------ markers
    stage('marker measurements')
    if marker_feats is not None:
        def _markers():
            labels = {'mean_marker_intensity': 'Mean marker intensity (z-score)',
                      'mean_marker_intensity_raw': 'Mean marker intensity (0-1)',
                      'n_spots': 'Number of marker spots (log(1+n))',
                      'mean_marker_spot_intensity': 'Mean spot intensity (z-score)',
                      'mean_marker_spot_intensity_raw': 'Mean spot intensity (0-1)'}
            uniq_mg = np.unique(marker_groups)
            for kk, mname in enumerate(MARKER_FEATURE_NAMES):
                vals = marker_feats[:, kk].copy()
                if mname == 'n_spots':
                    vals = np.log1p(vals)
                ok = vals[~np.isnan(vals)]
                if len(ok) == 0:
                    continue
                lo, hi = np.percentile(ok, [2, 98])
                for mg in uniq_mg:
                    sel = (marker_groups == mg) & ~np.isnan(vals)
                    v = np.full(N, np.nan)
                    v[sel] = vals[sel]
                    tag = '' if len(uniq_mg) == 1 else '_%s' % safe_name(mg)
                    fig = scatter_map(v, labels[mname] + ('' if len(uniq_mg) == 1 else '\nmarker: %s' % mg),
                                      cmap='magma' if mname == 'n_spots' else 'coolwarm', vmin=lo, vmax=hi)
                    save(fig, 'markers', 'phenotype_map_%s%s' % (mname, tag))
                # mean +- 2 SEM per group and per cluster, split by marker
                for by, by_vals, by_uniq, by_name in [(group_col, groups, uniq_groups, safe_name(group_col)),
                                                      ('Cluster', clust_labels + 1, uniq_clusters + 1, 'cluster')]:
                    fig, ax = plt.subplots(figsize=(2 + 0.9 * len(by_uniq), 4))
                    w = 0.8 / len(uniq_mg)
                    grays = sns.color_palette('Greys', len(uniq_mg) + 1)[1:]
                    rows = []
                    for mm, mg in enumerate(uniq_mg):
                        means, errs = [], []
                        for b in by_uniq:
                            x = marker_feats[(by_vals == b) & (marker_groups == mg), kk]
                            x = x[~np.isnan(x)]
                            means.append(np.mean(x) if len(x) else np.nan)
                            errs.append(2 * np.std(x, ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan)
                            rows.append({by: b, 'marker': mg, 'mean': means[-1], '2xSEM': errs[-1], 'n': len(x)})
                        xpos = np.arange(len(by_uniq)) - 0.4 + w * (mm + 0.5)
                        ax.bar(xpos, means, width=w, color=grays[mm], ec='k', label=str(mg))
                        ax.errorbar(xpos, means, yerr=errs, fmt='none', color='k', capsize=4)
                    ax.set_xticks(np.arange(len(by_uniq)))
                    ax.set_xticklabels([str(b) for b in by_uniq], rotation=45 if by != 'Cluster' else 0,
                                       ha='right' if by != 'Cluster' else 'center')
                    ax.set_ylabel(labels[mname].replace(' (log(1+n))', ''))
                    ax.set_xlabel(by)
                    if len(uniq_mg) > 1:
                        ax.legend(frameon=False, fontsize=8)
                    ax.spines[['right', 'top']].set_visible(False)
                    ax.set_title('mean +- 2 SEM')
                    save(fig, 'markers', '%s_per_%s' % (mname, by_name))
                    pd.DataFrame(rows).to_csv(os.path.join(dirs['markers'], '%s_per_%s.csv' % (mname, by_name)),
                                              index=None)
        attempt('marker plots', _markers)
    else:
        log('no marker measurements in this dataset')

    # ------------------------------------------------------------------ tables + summary
    table = pd.DataFrame({'Filename': obj_file, 'Img_Channel_No': obj_ch, 'Org_ID': obj_org, 'Frame_No': obj_frame,
                          'Time[h]': TP * Ts})
    for col in ['Conditions', 'Genetics', 'Patients', 'Marker']:
        table[col] = meta_column(col)
    if group_col not in table.columns:
        table[group_col] = groups
    table[emb_label + '_1'] = uu[:, 0]
    table[emb_label + '_2'] = uu[:, 1]
    table['Phenotype_Cluster'] = clust_labels + 1
    if module_expr is not None:
        for ii, mn in enumerate(module_names):
            table[mn] = module_expr[:, ii]
    if marker_feats is not None:
        for kk, mname in enumerate(MARKER_FEATURE_NAMES):
            table[mname] = marker_feats[:, kk]
    table.to_csv(os.path.join(resdir, 'organoid_phenotypes.csv'), index=None, encoding='utf-8')
    pd.DataFrame(X, columns=feature_names).assign(Filename=obj_file, Org_ID=obj_org, Frame_No=obj_frame).to_csv(
        os.path.join(resdir, 'normalised_SAM_features.csv'), index=None)

    summary['finished'] = time.strftime('%Y-%m-%d %H:%M:%S')
    summary['settings'] = A
    with open(os.path.join(resdir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    emit('PROGRESS', n_sections, n_sections)
    log('analysis saved to', resdir)


def safe_label(x):
    return re.sub(r'[\\/:*?"<>|]+', '-', str(x)).strip() or 'NA'


# =============================================================================
#   Runner
# =============================================================================

STEP_FUNCS = {'prepare': step_prepare, 'detect': step_detect, 'track': step_track, 'segment': step_segment,
              'postprocess': step_postprocess, 'features': step_features, 'markers': step_markers}


def run_history_folder(cfg):
    return os.path.join(cfg['output_folder'], 'run_history')


def start_run_record(cfg, steps, videos):
    """ run_history/<run id>/: the complete settings, sample table, software versions and log of this run """
    global _LOG_FH
    run_id = time.strftime('%Y%m%d-%H%M%S')
    folder = os.path.join(run_history_folder(cfg), run_id)
    os.makedirs(folder, exist_ok=True)
    if os.path.exists(metadata_file(cfg)):
        shutil.copyfile(metadata_file(cfg), os.path.join(folder, 'sample_info.csv'))
    _LOG_FH = open(os.path.join(folder, 'log.txt'), 'w', encoding='utf-8')
    record = {'run_id': run_id,
              'steps': steps,
              'videos': [v['name'] for v in videos],
              'command': 'python "%s" --config "%s" --steps %s' % (os.path.abspath(__file__),
                                                                  os.path.join(folder, 'run_config.json'),
                                                                  ','.join(steps)),
              'provenance': provenance(cfg, include_weights=any(s in steps for s in ('detect', 'segment'))),
              'config': cfg}
    write_json(os.path.join(folder, 'run_config.json'), record)
    return run_id, folder, record


def finish_run_record(folder, record, results, failed):
    global _LOG_FH
    record['finished'] = time.strftime('%Y-%m-%d %H:%M:%S')
    record['results'] = results
    record['n_failed'] = failed
    write_json(os.path.join(folder, 'run_config.json'), record)
    if _LOG_FH is not None:
        _LOG_FH.close()
        _LOG_FH = None


def run(cfg, steps):
    steps = [s for s in STEP_KEYS if s in steps]
    videos = selected_videos(cfg)
    skip_done = bool(cfg.get('skip_done', True))
    os.makedirs(cfg['output_folder'], exist_ok=True)
    failed = 0
    run_id, run_folder, record = start_run_record(cfg, steps, videos)
    results = {}
    log('SPOT pipeline run %s: %d videos, steps: %s' % (run_id, len(videos), ', '.join(steps)))
    log('settings and log of this run: %s' % run_folder)

    for step in steps:
        if step in PER_VIDEO_STEPS:
            emit('STAGE', step)
            for ii, video in enumerate(videos):
                emit('PROGRESS', ii, len(videos))
                status = step_status(cfg, video, step)
                if status == 'n/a':
                    continue
                if skip_done and status in ('done', 'none'):
                    emit('STEP', step, video['name'], 'skip')
                    results.setdefault(step, {})[video['name']] = 'skipped (up to date)'
                    continue
                # a step needs the output of the previous one, computed with the current settings
                prev = upstream_step(cfg, step)
                if prev is not None:
                    prev_status = step_status(cfg, video, prev)
                    if prev_status in ('todo', 'stale'):
                        log('[%s] %s: the previous step (%s) %s - run it first' % (
                            step, video['name'], STEP_LABELS[prev],
                            'has not been run' if prev_status == 'todo' else 'was computed with other settings'))
                        emit('STEP', step, video['name'], 'fail')
                        results.setdefault(step, {})[video['name']] = 'not run: previous step ' + prev_status
                        failed += 1
                        continue
                emit('STEP', step, video['name'], 'start')
                log('[%s] %s%s' % (step, video['name'],
                                   '  (settings changed since it was computed)' if status == 'stale' else ''))
                t0 = time.time()
                try:
                    res = STEP_FUNCS[step](cfg, video)
                    write_step_params(cfg, video, step, run_id)
                    emit('STEP', step, video['name'], 'none' if res == 'none' else 'done')
                    results.setdefault(step, {})[video['name']] = 'no organoids' if res == 'none' else 'done'
                    log('    done in %.1f s' % (time.time() - t0))
                except Exception:
                    failed += 1
                    log(traceback.format_exc())
                    emit('STEP', step, video['name'], 'fail')
                    results.setdefault(step, {})[video['name']] = 'failed'
            emit('PROGRESS', len(videos), len(videos))
        else:
            emit('STAGE', step)
            if skip_done and step == 'compile' and dataset_step_status(cfg, step) == 'done':
                emit('STEP', step, '*', 'skip')
                results[step] = 'skipped (up to date)'
                continue
            emit('STEP', step, '*', 'start')
            t0 = time.time()
            try:
                if step == 'compile':
                    step_compile(cfg, videos)
                    write_json(compile_record_file(cfg), {'run_id': run_id, 'finished': time.strftime('%Y-%m-%d %H:%M:%S'),
                                                          'params': compile_params(cfg, videos)})
                else:
                    run_analysis(cfg)
                    resdir = analysis_folder(cfg)
                    if os.path.exists(metadata_file(cfg)):
                        shutil.copyfile(metadata_file(cfg), os.path.join(resdir, 'sample_info_used.csv'))
                    write_json(analysis_record_file(cfg),
                               {'run_id': run_id, 'finished': time.strftime('%Y-%m-%d %H:%M:%S'),
                                'note': 'Load this file with "Load settings..." in the SAM-SPOT GUI to reproduce '
                                        'this analysis.',
                                'params': analysis_params(cfg),
                                'provenance': record['provenance'],
                                'config': cfg})
                emit('STEP', step, '*', 'done')
                results[step] = 'done'
                log('    done in %.1f s' % (time.time() - t0))
            except Exception:
                failed += 1
                log(traceback.format_exc())
                emit('STEP', step, '*', 'fail')
                results[step] = 'failed'
                if step == 'compile':
                    break
    finish_run_record(run_folder, record, results, failed)
    emit('FINISHED', failed)
    return failed


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description='Run the SPOT pipeline on a project created with the SPOT GUI.')
    parser.add_argument('--config', required=True, help='spot_project.json')
    parser.add_argument('--steps', default=','.join(STEP_KEYS), help='comma separated steps: ' + ','.join(STEP_KEYS))
    args = parser.parse_args(argv)
    os.environ.setdefault('MPLBACKEND', 'Agg')
    import matplotlib
    matplotlib.use('Agg')
    cfg = load_config(args.config)
    raw = read_json(args.config) or {}
    if 'config' in raw and os.path.dirname(os.path.dirname(os.path.abspath(args.config))).endswith('run_history'):
        log('re-running the settings of run %s' % raw.get('run_id'))
    failed = run(cfg, [s.strip() for s in args.steps.split(',') if s.strip()])
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
