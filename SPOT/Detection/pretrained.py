# -*- coding: utf-8 -*-
"""
Download (and cache) the pretrained SPOT organoid detection and segmentation networks.

    from SPOT.Detection.pretrained import get_pretrained_weights

    ptfile = get_pretrained_weights('yolov3_organoid_detector')          # PyTorch .pt (downloads + converts once)
    h5file = get_pretrained_weights('yolov3_organoid_detector', torch=False)  # the original Keras .h5

Models (see README.md):

    'yolov3_organoid_detector'          YOLOv3 organoid bounding box detector (Dropbox)
    'attn_unet_organoid_segmentation'   attention U-Net organoid segmentation of a bounding box crop (GitHub repo)

Files are cached in the folder given by ``cache_dir``, else the SPOT_MODELS_DIR environment variable, else
~/.cache/SPOT/models. A file already in the cache, or in the repository's models/ folder, is not downloaded again.

@author: fyz11
"""

import hashlib
import os
import shutil
import tempfile

REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PRETRAINED_MODELS = {
    'yolov3_organoid_detector': dict(
        url='https://www.dropbox.com/scl/fi/qzowc9s9n30zh6qdyzeqw/keras_YOLOv3_organoid_detector2.h5'
            '?rlkey=6deiqemsmcz3yin9b5dnz0e6y&dl=1',
        filename='keras_YOLOv3_organoid_detector2.h5',
        repo_folder=os.path.join('models', 'detect_CNN_model'),
        sha256=None),
    'attn_unet_organoid_segmentation': dict(
        url='https://github.com/fyz11/SPOT/raw/main/models/segment_CNN_model/'
            'organoid-bbox_attn_seg-unet-master-v2_64x64-8feats_v2.h5',
        filename='organoid-bbox_attn_seg-unet-master-v2_64x64-8feats_v2.h5',
        repo_folder=os.path.join('models', 'segment_CNN_model'),
        sha256=None),
}

_HDF5_SIGNATURE = b'\x89HDF\r\n\x1a\n'


def default_cache_dir():
    """ the folder pretrained weights are cached in: $SPOT_MODELS_DIR, else ~/.cache/SPOT/models """
    return os.environ.get('SPOT_MODELS_DIR', os.path.join(os.path.expanduser('~'), '.cache', 'SPOT', 'models'))


def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def download_file(url, path, progress=True, timeout=60):
    """ Download url to path (written to a temporary file first, so an interrupted download leaves no partial file).

    Parameters
    ----------
    url : str
    path : str
        destination file
    progress : bool
        show a tqdm progress bar
    timeout : float
        socket timeout in seconds

    Returns
    -------
    path : str
    """
    import urllib.request
    from tqdm import tqdm

    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)
    request = urllib.request.Request(url, headers={'User-Agent': 'SAM-SPOT'})
    fd, tmp = tempfile.mkstemp(dir=folder, suffix='.part')
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response, os.fdopen(fd, 'wb') as out:
            total = response.headers.get('Content-Length')
            total = int(total) if total else None
            with tqdm(total=total, unit='B', unit_scale=True, desc=os.path.basename(path), disable=not progress) as bar:
                for block in iter(lambda: response.read(1 << 20), b''):
                    out.write(block)
                    bar.update(len(block))
        shutil.move(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return path


def _check_hdf5(path, name):
    with open(path, 'rb') as f:
        if f.read(8) != _HDF5_SIGNATURE:
            raise IOError('The download of %s (%s) is not an HDF5 / Keras .h5 file; the link may have changed or '
                          'require a browser. Download it manually (see README.md) and pass its path, or place it in %s.'
                          % (name, path, os.path.dirname(path)))


def get_pretrained_weights(name, torch=True, cache_dir=None, force_download=False, progress=True):
    """ Path to the pretrained weights of a SPOT network, downloading (and converting to PyTorch) on first use.

    Parameters
    ----------
    name : str
        'yolov3_organoid_detector' or 'attn_unet_organoid_segmentation' (see PRETRAINED_MODELS)
    torch : bool
        if True return the PyTorch .pt (converted from the Keras .h5 with SPOT.Detection.convert_weights, needs torch
        and h5py), else the Keras .h5
    cache_dir : None or str
        download folder. Default: $SPOT_MODELS_DIR, else ~/.cache/SPOT/models
    force_download : bool
        download (and convert) again even if the file is cached
    progress : bool
        show a download progress bar

    Returns
    -------
    path : str
        the .pt (torch=True) or .h5 file
    """
    if name not in PRETRAINED_MODELS:
        raise ValueError('unknown pretrained model %r, choose from %s' % (name, sorted(PRETRAINED_MODELS)))
    info = PRETRAINED_MODELS[name]
    cache_dir = cache_dir or default_cache_dir()

    # a copy in the repository models/ folder (git checkout) or in the cache is used as is
    h5file = None
    if not force_download:
        for folder in (os.path.join(REPO_DIR, info['repo_folder']), cache_dir):
            candidate = os.path.join(folder, info['filename'])
            if os.path.exists(candidate):
                h5file = candidate
                break
    if h5file is None:
        h5file = os.path.join(cache_dir, info['filename'])
        print('downloading %s to %s' % (name, h5file))
        download_file(info['url'], h5file, progress=progress)
        _check_hdf5(h5file, name)
        if info['sha256'] is not None and _sha256(h5file) != info['sha256']:
            os.remove(h5file)
            raise IOError('checksum mismatch for the download of %s' % name)
    if not torch:
        return h5file

    ptfile = os.path.splitext(h5file)[0] + '.pt'
    if force_download or not os.path.exists(ptfile) or os.path.getmtime(ptfile) < os.path.getmtime(h5file):
        from .convert_weights import convert
        try:
            ptfile = convert(h5file, ptfile=ptfile)
        except PermissionError:
            # e.g. a read-only repository folder: convert into the cache instead
            os.makedirs(cache_dir, exist_ok=True)
            ptfile = convert(h5file, ptfile=os.path.join(cache_dir, os.path.basename(ptfile)))
    return ptfile
