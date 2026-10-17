# -*- coding: utf-8 -*-
"""SPOT.Detection.pretrained: download, cache and conversion of the pretrained networks (no network access: the
downloads are served from local file:// URLs)."""

import os
import pathlib

import numpy as np
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('h5py')
import SPOT.Detection.pretrained as pretrained  # noqa: E402

SEG = 'attn_unet_organoid_segmentation'
SEG_H5 = os.path.join(pretrained.REPO_DIR, 'models', 'segment_CNN_model',
                      'organoid-bbox_attn_seg-unet-master-v2_64x64-8feats_v2.h5')


def _serve(monkeypatch, name, source, tmp_path):
    """ point the model at a file:// URL and hide the repository copy """
    info = dict(pretrained.PRETRAINED_MODELS[name], url=pathlib.Path(source).resolve().as_uri(), sha256=None)
    monkeypatch.setitem(pretrained.PRETRAINED_MODELS, name, info)
    monkeypatch.setattr(pretrained, 'REPO_DIR', str(tmp_path / 'no_repo'))


def test_repository_copy_used_without_download(monkeypatch, tmp_path):
    def no_download(*a, **k):
        raise AssertionError('should not download')
    monkeypatch.setattr(pretrained, 'download_file', no_download)
    assert pretrained.get_pretrained_weights(SEG, torch=False, cache_dir=str(tmp_path)) == SEG_H5


def test_download_cache_and_convert(monkeypatch, tmp_path):
    _serve(monkeypatch, SEG, SEG_H5, tmp_path)
    cache = tmp_path / 'cache'
    ptfile = pretrained.get_pretrained_weights(SEG, torch=True, cache_dir=str(cache), progress=False)
    assert ptfile == str(cache / 'organoid-bbox_attn_seg-unet-master-v2_64x64-8feats_v2.pt')
    assert (cache / 'organoid-bbox_attn_seg-unet-master-v2_64x64-8feats_v2.h5').read_bytes() == pathlib.Path(SEG_H5).read_bytes()
    assert not list(cache.glob('*.part'))

    # same weights as the shipped conversion
    shipped = torch.load(os.path.splitext(SEG_H5)[0] + '.pt', map_location='cpu')['state_dict']
    got = torch.load(ptfile, map_location='cpu')['state_dict']
    assert shipped.keys() == got.keys() and all(torch.equal(shipped[k], got[k]) for k in shipped)

    # cached: no second download
    def no_download(*a, **k):
        raise AssertionError('should not download')
    monkeypatch.setattr(pretrained, 'download_file', no_download)
    assert pretrained.get_pretrained_weights(SEG, torch=True, cache_dir=str(cache)) == ptfile


def test_non_hdf5_download_raises(monkeypatch, tmp_path):
    page = tmp_path / 'page.html'
    page.write_text('<html>Dropbox sign in</html>')
    _serve(monkeypatch, 'yolov3_organoid_detector', page, tmp_path)
    with pytest.raises(IOError, match='not an HDF5'):
        pretrained.get_pretrained_weights('yolov3_organoid_detector', cache_dir=str(tmp_path / 'cache'), progress=False)
    assert not list((tmp_path / 'cache').iterdir())  # the bad download is not left as a cached .h5


def test_deleted_dropbox_file_message_and_url_override(monkeypatch, tmp_path):
    page = tmp_path / 'page.html'
    page.write_text('<html><title>Dropbox - File Deleted - Simplify your life</title></html>')
    _serve(monkeypatch, 'yolov3_organoid_detector', page, tmp_path)
    with pytest.raises(IOError, match='has been deleted'):
        pretrained.get_pretrained_weights('yolov3_organoid_detector', torch=False, cache_dir=str(tmp_path / 'c1'),
                                          progress=False)
    # a working link passed with url= is used instead
    h5 = pretrained.get_pretrained_weights('yolov3_organoid_detector', torch=False, cache_dir=str(tmp_path / 'c2'),
                                           progress=False, url=pathlib.Path(SEG_H5).resolve().as_uri())
    assert pathlib.Path(h5).read_bytes() == pathlib.Path(SEG_H5).read_bytes()


def test_unknown_model():
    with pytest.raises(ValueError):
        pretrained.get_pretrained_weights('nope')


def test_segmentation_loader_defaults_to_pretrained():
    from SPOT.Detection.detection import load_torch_segmentation_model
    model = load_torch_segmentation_model(device='cpu')
    patches = np.random.default_rng(0).random((2, 64, 64, 3)).astype(np.float32)
    assert np.isfinite(model.predict(patches)).all()


def test_registry_points_to_zenodo_with_checksums():
    """ both models are hosted on Zenodo record 23438299 and verified with a sha256 (offline check of the registry) """
    for name, info in pretrained.PRETRAINED_MODELS.items():
        assert info['url'] == 'https://zenodo.org/records/23438299/files/%s?download=1' % info['filename'], name
        assert info['sha256'] is not None and len(info['sha256']) == 64, name
    # the repository copy of the segmentation model is the hosted file
    import hashlib
    assert hashlib.sha256(pathlib.Path(SEG_H5).read_bytes()).hexdigest() ==         pretrained.PRETRAINED_MODELS['attn_unet_organoid_segmentation']['sha256']
