# ------------------------------------------------------------------------------
# Reference: https://github.com/jamycheung/DELIVER/blob/main/semseg/datasets/deliver.py
# ------------------------------------------------------------------------------

import json
import logging
import os
import glob

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.builtin_meta import CITYSCAPES_CATEGORIES
from detectron2.utils.file_io import PathManager

"""
This file contains functions to register the DELIVER panoptic dataset to the DatasetCatalog.
"""

# num_classes: 25
CLASSES = ["Building", "Fence", "Other", "Pedestrian", "Pole", "RoadLine", "Road", "SideWalk", "Vegetation",
            "Cars", "Wall", "TrafficSign", "Sky", "Ground", "Bridge", "RailTrack", "GroundRail",
            "TrafficLight", "Static", "Dynamic", "Water", "Terrain", "TwoWheeler", "Bus", "Truck"]

# 5 thing classes
THING_CLASSES = ["Pedestrian", "Cars", "TwoWheeler", "Bus", "Truck"] #, "Static", "Dynamic"

PALETTE = [[70, 70, 70],
        [100, 40, 40],
        [55, 90, 80],
        [220, 20, 60],
        [153, 153, 153],
        [157, 234, 50],
        [128, 64, 128],
        [244, 35, 232],
        [107, 142, 35],
        [0, 0, 142],
        [102, 102, 156],
        [220, 220, 0],
        [70, 130, 180],
        [81, 0, 81],
        [150, 100, 100],
        [230, 150, 140],
        [180, 165, 180],
        [250, 170, 30],
        [110, 190, 160],
        [170, 120, 50],
        [45, 60, 150],
        [145, 170, 100],
        [  0,  0, 230],
        [  0, 60, 100],
        [  0,  0, 70],
        ]


logger = logging.getLogger(__name__)


def get_deliver_panoptic_files(image_dir, gt_dir, json_info):
    image_id_to_idx = get_image_id_to_idx(json_info)
    files = []
    for ann in json_info["annotations"]:
        image_file = os.path.join(image_dir, ann["file_name"].replace("_gt_panoptic.png", "_frame_camera.png"))
        label_file = os.path.join(gt_dir, ann["file_name"])
        segments_info = ann["segments_info"]
        image_idx = image_id_to_idx[ann['image_id']]
        scene_info = json_info['images'][image_idx]
        files.append((image_file, label_file, segments_info, scene_info))

    assert len(files), "No images found in {}".format(image_dir)
    assert PathManager.isfile(files[0][0]), files[0][0]
    assert PathManager.isfile(files[0][1]), files[0][1]
    return files

def get_image_id_to_idx(json_info):
    image_id_to_idx = {}
    for idx, image_info in enumerate(json_info['images']):
        if image_info['id'] in image_id_to_idx:
            raise ValueError(f"Duplicate image ID found: {image_info['id']}. \
                Creating lookup table with duplicates is not supported.")
        image_id_to_idx[image_info['id']] = idx
    return image_id_to_idx

def load_deliver_semantic(files, meta):

    def _convert_category_id(segment_info, meta):
        if segment_info["category_id"] in meta["load_deliver_semantic"]:
            segment_info["category_id"] = meta["load_deliver_semantic"][
                segment_info["category_id"]
            ]
        else:
            segment_info["category_id"] = meta["stuff_dataset_id_to_contiguous_id"][
                segment_info["category_id"]
            ]
        return segment_info

    ret = []
    for file in files:
        rgb = str(file)
        gt_depth_path = rgb.replace('/img', '/depth').replace('_rgb', '_depth')
        x1 = rgb.replace('/img', '/hha').replace('_rgb', '_depth')
        x2 = rgb.replace('/img', '/lidar').replace('_rgb', '_lidar')
        # x2 = rgb.replace('/img', '/lidar').replace('_rgb', '_lidar').replace('.png', '_color.png')
        x3 = rgb.replace('/img', '/event').replace('_rgb', '_event')
        lbl_path = rgb.replace('/img', '/semantic').replace('_rgb', '_semantic')

        case_present = False
        for case in ['motionblur', 'overexposure', 'underexposure', 'lidarjitter', 'eventlowres']:
            if case in rgb:
                case_present = case

        scene_info = {'condition': file.split('/')[-4], 'case': case_present}

        image_id = "-".join(rgb.split('/')[-2:]).split('.png')[0]

        ret.append(
            {
                "file_name": rgb,
                "image_id": image_id,
                "sem_seg_file_name": lbl_path,
                "gt_depth_file_name": gt_depth_path,
                "depth_file_name": x1,
                "lidar_file_name": x2,
                "event_file_name": x3,
                "scene_info": scene_info,
            }
        )
    assert len(ret), f"No images found in DeLiVER dataset folder: {files}"
    assert PathManager.isfile(
        ret[0]["sem_seg_file_name"]
    ), "semantic segmentation file not found: {}".format(ret[0]["sem_seg_file_name"])
    return ret


_RAW_DELIVER_SEMANTIC_SPLITS = {
    "deliver_semantic_train": (
        "train",
        "deliver",
        None, #"debug",
    ),
    "deliver_semantic_val": (
        "val",
        "deliver",
        None, #"debug",
    ),
    "deliver_semantic_test": (
        "test",
        "deliver",
        None,
    ),
}


def register_all_deliver_semantic(root):
    meta = {}
    thing_classes = [k for k in CLASSES if k in THING_CLASSES]
    thing_colors = [PALETTE[CLASSES.index(k)] for k in thing_classes]
    stuff_classes = [k for k in CLASSES] # if k not in THING_CLASSES    # seams like I need to have all names in
    stuff_colors = [PALETTE[CLASSES.index(k)] for k in stuff_classes]

    meta["thing_classes"] = thing_classes
    meta["thing_colors"] = thing_colors
    meta["stuff_classes"] = stuff_classes
    meta["stuff_colors"] = stuff_colors

    thing_dataset_id_to_contiguous_id = {}
    stuff_dataset_id_to_contiguous_id = {}

    # enumerate(CLASSES):
    for idx, category in enumerate(CLASSES):
        if category in THING_CLASSES:
            thing_dataset_id_to_contiguous_id[idx] = idx

        # in order to use sem_seg evaluator
        stuff_dataset_id_to_contiguous_id[idx] = idx

    meta["thing_dataset_id_to_contiguous_id"] = thing_dataset_id_to_contiguous_id
    meta["stuff_dataset_id_to_contiguous_id"] = stuff_dataset_id_to_contiguous_id

    for key, (split, dataset_folder, case) in _RAW_DELIVER_SEMANTIC_SPLITS.items():
        dataset_root = os.path.join(root, dataset_folder)
        ignore_label = 255
        files = sorted(glob.glob(os.path.join(*[dataset_root, 'img', '*', split, '*', '*.png'])))
        if not files:
            logger.warning(f"[WARNING] No images found in {dataset_root} for {split} split.")
        if case is not None:
            if case == 'debug':
                logger.warning(f"[Debug mode] Only using 100 images for the {split} split.")
                files = files[:100]
            else:
                assert case in ['cloud', 'fog', 'night', 'rain', 'sun', 'motionblur', 'overexposure', 'underexposure', 'lidarjitter', 'eventlowres'], "Case name not available."
                files = [f for f in files if case in f]
                logger.warning(f"Found {len(files)} {split} {case} images.")

        if key in DatasetCatalog.list():
            DatasetCatalog.remove(key)

        DatasetCatalog.register(
            key, lambda x=files: load_deliver_semantic(x, meta)
        )

        MetadataCatalog.get(key).set(
            panoptic_root=dataset_root,
            image_root=os.path.join(dataset_root, 'img'),
            gt_dir=os.path.join(dataset_root, 'semantic'),
            evaluator_type="deliver_semantic_seg",
            ignore_label=ignore_label,
            # label_divisor=1,
            **meta,
            )

_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_deliver_semantic(_root)
