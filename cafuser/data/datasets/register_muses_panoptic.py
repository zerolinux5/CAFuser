import json
import logging
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.builtin_meta import CITYSCAPES_CATEGORIES
from detectron2.utils.file_io import PathManager


logger = logging.getLogger(__name__)


def get_muses_panoptic_files(image_dir, gt_dir, json_info):
    image_id_to_idx = get_image_id_to_idx(json_info)
    files = []
    if "annotations" in json_info:
        for ann in json_info["annotations"]:
            image_file = os.path.join(image_dir, ann["file_name"].replace("_gt_panoptic.png", "_frame_camera.png"))
            label_file = os.path.join(gt_dir, ann["file_name"])
            segments_info = ann["segments_info"]
            image_idx = image_id_to_idx[ann['image_id']]
            scene_info = json_info['images'][image_idx]
            files.append((image_file, label_file, segments_info, scene_info))
    else:
        logger.warning("No annotations found in json file. Using only image files.")
        for image_info in json_info["images"]:
            image_file = os.path.join(image_dir, image_info["file_name"].replace("_gt_panoptic.png", "_frame_camera.png"))
            files.append((image_file, None, None, image_info))

    assert len(files), "No images found in {}".format(image_dir)
    assert PathManager.isfile(files[0][0]), files[0][0]
    return files

def get_image_id_to_idx(json_info):
    image_id_to_idx = {}
    for idx, image_info in enumerate(json_info['images']):
        if image_info['id'] in image_id_to_idx:
            raise ValueError(f"Duplicate image ID found: {image_info['id']}. \
                Creating lookup table with duplicates is not supported.")
        image_id_to_idx[image_info['id']] = idx
    return image_id_to_idx

def load_muses_panoptic(image_dir, gt_dir, gt_json, meta):
    """
    Args:
        image_dir (str): path to the raw dataset. e.g., "~/muses/frame_camera".
        gt_dir (str): path to the raw annotations. e.g.,
            "~/muses/gt_panoptic/train".
        gt_json (str): path to the json file. e.g.,
            "~/muses/gt_panoptic/train.json".
        meta (dict): dictionary containing "thing_dataset_id_to_contiguous_id"
            and "stuff_dataset_id_to_contiguous_id" to map category ids to
            contiguous ids for training.

    Returns:
        list[dict]: a list of dicts in Detectron2 standard format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )
    """

    def _convert_category_id(segment_info, meta):
        if segment_info["category_id"] in meta["thing_dataset_id_to_contiguous_id"]:
            segment_info["category_id"] = meta["thing_dataset_id_to_contiguous_id"][
                segment_info["category_id"]
            ]
        else:
            segment_info["category_id"] = meta["stuff_dataset_id_to_contiguous_id"][
                segment_info["category_id"]
            ]
        return segment_info

    assert os.path.exists(
        gt_json
    ), "json file does not exist"  # noqa

    with open(gt_json) as f:
        json_info = json.load(f)
    
    files = get_muses_panoptic_files(image_dir, gt_dir, json_info)
    ret = []
    for image_file, label_file, segments_info, scene_info in files:
        if label_file:
            sem_label_file = label_file.replace("/gt_panoptic", "/gt_semantic")
            sem_label_file = sem_label_file.replace("_gt_panoptic.png", "_gt_labelTrainIds.png")
            segments_info = [_convert_category_id(x, meta) for x in segments_info]
        else:
            sem_label_file = None
            segments_info = None

        ret.append(
            {
                "file_name": image_file,
                "image_id": "_".join(
                    os.path.splitext(os.path.basename(image_file))[0].split("_")[:3]
                ),
                "sem_seg_file_name": sem_label_file,
                "pan_seg_file_name": label_file,
                "segments_info": segments_info,
                "scene_info": scene_info,
            }
        )

    assert len(ret), f"No images found in {image_dir}!"
    if ret[0]["sem_seg_file_name"]:
        assert PathManager.isfile(ret[0]["sem_seg_file_name"]), "no semseg file"  # noqa
    if ret[0]["pan_seg_file_name"]:
        assert PathManager.isfile(ret[0]["pan_seg_file_name"]), "no panoptic file"  # noqa
    return ret


_RAW_MUSES_PANOPTIC_SPLITS = {
    "muses_panoptic_train": (
        "muses/frame_camera",
        "muses/gt_panoptic",
        "muses/gt_panoptic/train.json",
    ),
    "muses_panoptic_val": (
        "muses/frame_camera",
        "muses/gt_panoptic",
        "muses/gt_panoptic/val.json",
    ),
    "muses_panoptic_test": (
        "muses/frame_camera",
        None,
        "muses/gt_panoptic/test_image_info.json",
    ),
}


def register_all_muses_panoptic(root):
    meta = {}

    thing_classes = [k["name"] for k in CITYSCAPES_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in CITYSCAPES_CATEGORIES if k["isthing"] == 1]
    stuff_classes = [k["name"] for k in CITYSCAPES_CATEGORIES]
    stuff_colors = [k["color"] for k in CITYSCAPES_CATEGORIES]

    meta["thing_classes"] = thing_classes
    meta["thing_colors"] = thing_colors
    meta["stuff_classes"] = stuff_classes
    meta["stuff_colors"] = stuff_colors

    thing_dataset_id_to_contiguous_id = {}
    stuff_dataset_id_to_contiguous_id = {}

    for k in CITYSCAPES_CATEGORIES:
        if k["isthing"] == 1:
            thing_dataset_id_to_contiguous_id[k["id"]] = k["trainId"]

        # in order to use sem_seg evaluator
        stuff_dataset_id_to_contiguous_id[k["id"]] = k["trainId"]

    meta["thing_dataset_id_to_contiguous_id"] = thing_dataset_id_to_contiguous_id
    meta["stuff_dataset_id_to_contiguous_id"] = stuff_dataset_id_to_contiguous_id

    for key, (image_dir, gt_dir, gt_json) in _RAW_MUSES_PANOPTIC_SPLITS.items():
        image_dir = os.path.join(root, image_dir)
        gt_dir = os.path.join(root, gt_dir) if gt_dir else None
        gt_json = os.path.join(root, gt_json)

        if key in DatasetCatalog.list():
            DatasetCatalog.remove(key)

        DatasetCatalog.register(
            key, lambda x=image_dir, y=gt_dir, z=gt_json: load_muses_panoptic(x, y, z, meta)
        )
        MetadataCatalog.get(key).set(
            panoptic_root=gt_dir,
            image_root=image_dir,
            panoptic_json=gt_json,
            gt_dir=gt_dir,
            evaluator_type="muses_panoptic_seg",
            ignore_label=255,
            label_divisor=1000,
            **meta,
        )

_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_muses_panoptic(_root)
