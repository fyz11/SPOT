# -*- coding: utf-8 -*-
"""
Convert SPOT's Keras .h5 network weights to PyTorch .pt files, without TensorFlow (only h5py + torch).

    python scripts/convert_keras_weights_to_torch.py                       # convert the default SPOT models found
    python scripts/convert_keras_weights_to_torch.py path/to/model.h5 ...  # convert specific files

The .pt is written next to each .h5 (same basename). The conversion itself is SPOT.Detection.convert_weights; to
download and convert the pretrained SPOT models automatically use SPOT.Detection.pretrained.get_pretrained_weights.
"""

import argparse
import os
import sys

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_DIR)

from SPOT.Detection.convert_weights import convert  # noqa: E402

DEFAULT_MODELS = [os.path.join(REPO_DIR, 'models', 'segment_CNN_model', 'organoid-bbox_attn_seg-unet-master-v2_64x64-8feats_v2.h5'),
                  os.path.join(REPO_DIR, 'models', 'detect_CNN_model', 'keras_YOLOv3_organoid_detector2.h5')]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('h5files', nargs='*', help='Keras .h5 files (default: the SPOT models in models/)')
    parser.add_argument('--arch', choices=['att_unet', 'unet', 'yolov3'], default=None, help='override architecture detection')
    args = parser.parse_args()

    h5files = args.h5files or DEFAULT_MODELS
    for h5file in h5files:
        if not os.path.exists(h5file):
            print('skipping %s (not found)' % h5file)
            if 'keras_YOLOv3_organoid_detector2' in h5file:
                print("  download it with SPOT.Detection.pretrained.get_pretrained_weights('yolov3_organoid_detector')")
            continue
        ptfile = convert(h5file, arch=args.arch)
        print('converted %s -> %s' % (h5file, ptfile))


if __name__ == '__main__':
    main()
