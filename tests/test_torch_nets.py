# -*- coding: utf-8 -*-
"""
Numerical equivalence of the PyTorch networks (SPOT.Detection.unets.*_torch, SPOT.Detection.yolov3_torch) with the
original Keras networks, including the Keras .h5 -> .pt weight conversion (scripts/convert_keras_weights_to_torch.py).

Tests are skipped when torch, tensorflow or the model weights are not available. The pretrained YOLOv3 weights are a
separate download (README.md): place keras_YOLOv3_organoid_detector2.h5 in models/detect_CNN_model/ to enable
test_yolov3_pretrained_boxes_match_keras.
"""

import glob
import importlib.util
import os

import numpy as np
import pytest

torch = pytest.importorskip('torch')

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEG_H5 = os.path.join(REPO_DIR, 'models', 'segment_CNN_model', 'organoid-bbox_attn_seg-unet-master-v2_64x64-8feats_v2.h5')
SEG_PT = os.path.splitext(SEG_H5)[0] + '.pt'
YOLO_H5 = os.path.join(REPO_DIR, 'models', 'detect_CNN_model', 'keras_YOLOv3_organoid_detector2.h5')
VIDEO = os.path.join(REPO_DIR, 'data', 'organoids', 'fluorescent_murine_colon', 'KRAS_G12D_EYFP_2.wmv')

SEG_ATOL = 1e-4


def _converter():
    spec = importlib.util.spec_from_file_location('convert_keras_weights_to_torch',
                                                  os.path.join(REPO_DIR, 'scripts', 'convert_keras_weights_to_torch.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _tf():
    tf = pytest.importorskip('tensorflow')
    return tf


def _video_frames(n_frames=4):
    """ a few real frames (n, H, W, 3) float in [0, 255], else synthetic blobs """
    try:
        import cv2
        cap = cv2.VideoCapture(VIDEO)
        frames = []
        while len(frames) < n_frames:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame[..., ::-1])
        cap.release()
        if len(frames) > 0:
            return np.array(frames).astype(np.float64)
    except ImportError:
        pass
    rng = np.random.RandomState(0)
    yy, xx = np.mgrid[:480, :640]
    frames = []
    for _ in range(n_frames):
        img = np.zeros((480, 640))
        for _ in range(8):
            cy, cx, r = rng.uniform(40, 440), rng.uniform(40, 600), rng.uniform(10, 40)
            img += 200 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * r ** 2))
        frames.append(np.dstack([img] * 3))
    return np.clip(np.array(frames), 0, 255)


def _patches(n=48, size=64, seed=0):
    """ 64x64x3 patches like SPOT.Tracking.track.segment_organoid_bbox_track feeds the segmentation model """
    import skimage.exposure as skexposure
    import skimage.transform as sktform
    rng = np.random.RandomState(seed)
    vid = _video_frames()
    patches = []
    for _ in range(n):
        frame = vid[rng.randint(len(vid))][..., 2]
        h = rng.randint(24, 160)
        y, x = rng.randint(0, frame.shape[0] - h), rng.randint(0, frame.shape[1] - h)
        crop = sktform.resize(frame[y:y + h, x:x + h], (size, size), preserve_range=True)
        crop = skexposure.rescale_intensity(crop.astype(np.float32))
        patches.append(np.dstack([crop] * 3))
    patches.append(rng.rand(size, size, 3).astype(np.float32))  # noise
    patches.append(np.zeros((size, size, 3), dtype=np.float32))  # empty
    return np.array(patches, dtype=np.float32)


def test_unets_module_imports_and_torch_unet_shapes():
    import SPOT.Detection.unets as unets
    for builder in (unets.unet_torch, unets.att_unet_torch):
        model = builder(n_label=2, img_ch=3, depth=3, features=4).eval()
        with torch.no_grad():
            out = model(torch.rand(2, 3, 32, 48))
        assert out.shape == (2, 2, 32, 48)
        assert float(out.min()) >= 0 and float(out.max()) <= 1


@pytest.mark.skipif(not os.path.exists(SEG_H5), reason='segmentation .h5 weights not found')
def test_att_unet_segmentation_matches_keras(tmp_path):
    _tf()
    from SPOT.Detection.unets import att_unet
    import SPOT.Detection.detection as detection

    keras_model = att_unet(img_w=64, img_h=64, n_label=1, img_ch=3, depth=4, features=8, data_format='channels_last')
    keras_model.load_weights(SEG_H5)

    ptfile = _converter().convert(SEG_H5, ptfile=str(tmp_path / 'seg.pt'))
    torch_model = detection.load_torch_segmentation_model(ptfile, device='cpu')

    x = _patches()
    p_keras = keras_model.predict(x, batch_size=16, verbose=0)
    p_torch = torch_model.predict(x, batch_size=16)
    assert p_torch.shape == p_keras.shape == (len(x), 64, 64, 1)
    maxdiff = np.abs(p_torch - p_keras).max()
    print('att_unet segmentation max abs diff (cpu): %.3g' % maxdiff)
    assert maxdiff < SEG_ATOL

    if torch.cuda.is_available():
        p_cuda = detection.load_torch_segmentation_model(ptfile, device='cuda').predict(x)
        maxdiff = np.abs(p_cuda - p_keras).max()
        print('att_unet segmentation max abs diff (cuda, no tf32): %.3g' % maxdiff)
        assert maxdiff < SEG_ATOL


@pytest.mark.skipif(not os.path.exists(SEG_PT), reason='converted segmentation .pt not found')
def test_shipped_segmentation_pt_matches_fresh_conversion(tmp_path):
    pytest.importorskip('h5py')
    ptfile = _converter().convert(SEG_H5, ptfile=str(tmp_path / 'seg.pt'))
    fresh = torch.load(ptfile, map_location='cpu')
    shipped = torch.load(SEG_PT, map_location='cpu')
    assert shipped['arch'] == fresh['arch'] == 'att_unet'
    assert shipped['config'] == fresh['config'] == {'n_label': 1, 'img_ch': 3, 'depth': 4, 'features': 8}
    for k, v in fresh['state_dict'].items():
        assert torch.equal(v, shipped['state_dict'][k]), k


def test_unet_matches_keras(tmp_path):
    """ plain unet with random weights (the Keras unet is channels_first only; TF cannot run NCHW convs on CPU) """
    tf = _tf()
    if not tf.config.list_physical_devices('GPU'):
        pytest.skip('Keras unet is channels_first only, which TensorFlow cannot run on CPU')
    from SPOT.Detection.unets import unet
    import SPOT.Detection.unets as unets

    keras_model = unet(img_w=32, img_h=32, n_label=1, img_ch=3, depth=3, features=4)
    h5 = str(tmp_path / 'unet.h5')
    keras_model.save_weights(h5)
    ptfile = _converter().convert(h5)
    torch_model = unets.load_unet_torch(ptfile)
    x = np.random.RandomState(0).rand(4, 3, 32, 32).astype(np.float32)
    p_keras = keras_model.predict(x, verbose=0)
    with torch.no_grad():
        p_torch = torch_model(torch.from_numpy(x)).numpy()
    assert np.abs(p_torch - p_keras).max() < SEG_ATOL


######################################################################################################################
# YOLOv3


def _keras_yolov3(n_classes=1):
    """ Reference keras-yolo3 (experiencor) YOLOv3, the layout of keras_YOLOv3_organoid_detector2.h5 """
    from tensorflow.keras.layers import Conv2D, Input, BatchNormalization, LeakyReLU, ZeroPadding2D, UpSampling2D, add, concatenate
    from tensorflow.keras.models import Model

    def conv_block(inp, convs, skip=True):
        x = inp
        count = 0
        for conv in convs:
            if count == (len(convs) - 2) and skip:
                skip_connection = x
            count += 1
            if conv['stride'] > 1:
                x = ZeroPadding2D(((1, 0), (1, 0)))(x)
            x = Conv2D(conv['filter'], conv['kernel'], strides=conv['stride'], padding='valid' if conv['stride'] > 1 else 'same',
                       name='conv_' + str(conv['layer_idx']), use_bias=False if conv['bnorm'] else True)(x)
            if conv['bnorm']:
                x = BatchNormalization(epsilon=0.001, name='bnorm_' + str(conv['layer_idx']))(x)
            if conv['leaky']:
                x = LeakyReLU(alpha=0.1, name='leaky_' + str(conv['layer_idx']))(x)
        return add([skip_connection, x]) if skip else x

    def c(idx, f, k, s=1, bnorm=True, leaky=True):
        return {'filter': f, 'kernel': k, 'stride': s, 'bnorm': bnorm, 'leaky': leaky, 'layer_idx': idx}

    n_out = 3 * (5 + n_classes)
    input_image = Input(shape=(None, None, 3))
    x = conv_block(input_image, [c(0, 32, 3), c(1, 64, 3, 2), c(2, 32, 1), c(3, 64, 3)])
    x = conv_block(x, [c(5, 128, 3, 2), c(6, 64, 1), c(7, 128, 3)])
    x = conv_block(x, [c(9, 64, 1), c(10, 128, 3)])
    x = conv_block(x, [c(12, 256, 3, 2), c(13, 128, 1), c(14, 256, 3)])
    for i in range(7):
        x = conv_block(x, [c(16 + i * 3, 128, 1), c(17 + i * 3, 256, 3)])
    skip_36 = x
    x = conv_block(x, [c(37, 512, 3, 2), c(38, 256, 1), c(39, 512, 3)])
    for i in range(7):
        x = conv_block(x, [c(41 + i * 3, 256, 1), c(42 + i * 3, 512, 3)])
    skip_61 = x
    x = conv_block(x, [c(62, 1024, 3, 2), c(63, 512, 1), c(64, 1024, 3)])
    for i in range(3):
        x = conv_block(x, [c(66 + i * 3, 512, 1), c(67 + i * 3, 1024, 3)])
    x = conv_block(x, [c(75, 512, 1), c(76, 1024, 3), c(77, 512, 1), c(78, 1024, 3), c(79, 512, 1)], skip=False)
    yolo_82 = conv_block(x, [c(80, 1024, 3), c(81, n_out, 1, bnorm=False, leaky=False)], skip=False)
    x = conv_block(x, [c(84, 256, 1)], skip=False)
    x = UpSampling2D(2)(x)
    x = concatenate([x, skip_61])
    x = conv_block(x, [c(87, 256, 1), c(88, 512, 3), c(89, 256, 1), c(90, 512, 3), c(91, 256, 1)], skip=False)
    yolo_94 = conv_block(x, [c(92, 512, 3), c(93, n_out, 1, bnorm=False, leaky=False)], skip=False)
    x = conv_block(x, [c(96, 128, 1)], skip=False)
    x = UpSampling2D(2)(x)
    x = concatenate([x, skip_36])
    yolo_106 = conv_block(x, [c(99, 128, 1), c(100, 256, 3), c(101, 128, 1), c(102, 256, 3), c(103, 128, 1), c(104, 256, 3),
                              c(105, n_out, 1, bnorm=False, leaky=False)], skip=False)
    return Model(input_image, [yolo_82, yolo_94, yolo_106])


def _randomize_weights(keras_model, seed=0):
    """ non-trivial BatchNorm statistics so the conversion of every BN parameter is exercised, and output heads
    scaled up so the random detector gives well separated scores (few near-ties for NMS) """
    rng = np.random.RandomState(seed)
    for layer in keras_model.layers:
        if layer.name.startswith('bnorm_'):
            n = layer.get_weights()[0].shape[0]
            layer.set_weights([rng.uniform(0.5, 1.5, n), rng.uniform(-0.2, 0.2, n),
                               rng.uniform(-0.2, 0.2, n), rng.uniform(0.5, 2.0, n)])
        if layer.name in ('conv_81', 'conv_93', 'conv_105'):
            # per anchor channels are x, y, w, h, objectness, classes: keep boxes near anchor size, spread the scores
            kernel, bias = layer.get_weights()
            is_score = (np.arange(kernel.shape[-1]) % (kernel.shape[-1] // 3)) >= 4
            kernel = kernel * np.where(is_score, 8., 0.25)
            bias = np.where(is_score, rng.uniform(-3, 0, bias.shape), 0.)
            layer.set_weights([kernel, bias])


def _detections(outfolder):
    """ all bounding box txt files written by the detection runner -> {relative path: (labels, scores, int boxes)} """
    files = sorted(glob.glob(os.path.join(str(outfolder), '*', 'bbox_predictions_*.txt')))
    dets = {}
    for f in files:
        rows = [line.split('	') for line in open(f).read().splitlines()]
        dets[os.path.relpath(f, str(outfolder))] = ([r[0] for r in rows],
                                                     np.array([float(r[1]) for r in rows]),
                                                     np.array([[int(v) for v in r[2:]] for r in rows], dtype=int).reshape(-1, 4))
    return dets


def _assert_same_detections(det_keras, det_torch, score_atol=1e-4):
    """ identical boxes (labels, integer coordinates, order) after NMS; scores equal up to float32 rounding """
    assert sorted(det_keras) == sorted(det_torch)
    for f in det_keras:
        labels_k, scores_k, boxes_k = det_keras[f]
        labels_t, scores_t, boxes_t = det_torch[f]
        assert labels_k == labels_t, f
        np.testing.assert_array_equal(boxes_k, boxes_t, err_msg=f)
        np.testing.assert_allclose(scores_k, scores_t, rtol=0, atol=score_atol, err_msg=f)


def _run_both_detectors(keras_model, ptfile, vid, tmp_path, **kwargs):
    import SPOT.Detection.detection as detection
    from SPOT.Detection.yolov3_torch import load_yolov3_torch

    out_keras, out_torch = tmp_path / 'keras', tmp_path / 'torch'
    raw = {'keras': [], 'torch': []}
    torch_predict = detection._torch_predict_fn(load_yolov3_torch(ptfile), torch.device('cpu'))

    def keras_fn(x):
        out = keras_model.predict(x, verbose=0)
        raw['keras'].append([o.copy() for o in out])
        return out

    def torch_fn(x):
        out = torch_predict(x)
        raw['torch'].append([o.copy() for o in out])
        return out

    detection._run_YOLOv3_detection_RGB(vid, keras_fn, str(out_keras), **kwargs)
    detection._run_YOLOv3_detection_RGB(vid, torch_fn, str(out_torch), **kwargs)
    maxdiff = max(np.abs(a - b).max() for fk, ft in zip(raw['keras'], raw['torch']) for a, b in zip(fk, ft))
    scale = max(np.abs(a).max() for fk in raw['keras'] for a in fk)
    return _detections(out_keras), _detections(out_torch), maxdiff, scale


def test_yolov3_random_weights_match_keras(tmp_path):
    """ conversion + forward pass + decoding of the full Darknet-53 YOLOv3 with random weights """
    tf = _tf()
    tf.keras.utils.set_random_seed(0)
    keras_model = _keras_yolov3(n_classes=1)
    _randomize_weights(keras_model)
    h5 = str(tmp_path / 'yolo.h5')
    keras_model.save(h5)  # full model, as keras_YOLOv3_organoid_detector2.h5

    ptfile = _converter().convert(h5)
    vid = _video_frames(n_frames=2)[..., :1]  # one channel
    # obj_thresh is raised from the pipeline default so random weights give a manageable number of boxes
    det_keras, det_torch, maxdiff, scale = _run_both_detectors(keras_model, ptfile, vid, tmp_path, obj_thresh=0.5, imsize=(512, 512))
    print('YOLOv3 random weights: raw head max abs diff %.3g (max |output| %.3g)' % (maxdiff, scale))
    assert maxdiff < 1e-4 * max(1., scale)
    n_boxes = [len(d[0]) for d in det_keras.values()]
    print('YOLOv3 random weights: boxes after NMS per frame', n_boxes)
    assert len(det_keras) == len(vid)
    assert sum(n_boxes) > 0
    _assert_same_detections(det_keras, det_torch)


@pytest.mark.skipif(not os.path.exists(YOLO_H5), reason='keras_YOLOv3_organoid_detector2.h5 not found; download it (README.md) into models/detect_CNN_model/')
def test_yolov3_pretrained_boxes_match_keras(tmp_path):
    _tf()
    from tensorflow.keras.models import load_model

    keras_model = load_model(YOLO_H5)
    ptfile = _converter().convert(YOLO_H5, ptfile=str(tmp_path / 'yolo.pt'))
    vid = _video_frames(n_frames=4)
    det_keras, det_torch, maxdiff, scale = _run_both_detectors(keras_model, ptfile, vid, tmp_path)  # pipeline defaults
    print('YOLOv3 pretrained: raw head max abs diff %.3g (max |output| %.3g)' % (maxdiff, scale))
    print('YOLOv3 pretrained: boxes after NMS per frame', [len(d[0]) for d in det_keras.values()])
    assert len(det_keras) == len(vid) * vid.shape[-1]
    _assert_same_detections(det_keras, det_torch)
