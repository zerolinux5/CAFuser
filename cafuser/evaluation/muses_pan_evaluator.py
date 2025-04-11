import contextlib
import io
import itertools
import json
import logging
import numpy as np
import os
import tempfile
from collections import OrderedDict
from typing import Optional
from PIL import Image
from tabulate import tabulate
from panopticapi.utils import IdGenerator

from detectron2.data import MetadataCatalog
from detectron2.utils import comm
from detectron2.utils.file_io import PathManager

from oneformer.evaluation.evaluator import DatasetEvaluator

logger = logging.getLogger(__name__)


class MUSESPanopticEvaluator(DatasetEvaluator):
    """
    Evaluate Panoptic Quality metrics on MUSES using PanopticAPI.
    It saves panoptic segmentation prediction in `output_dir`

    It contains a synchronize call and has to be called from all workers.
    """

    def __init__(self, dataset_name: str, output_dir: Optional[str] = None, write_out_confidence: bool = False, 
                        save_colored_pred: bool = False, inference_only: bool = False):
        """
        Args:
            dataset_name: name of the dataset
            output_dir: output directory to save results for evaluation.
        """
        self._metadata = MetadataCatalog.get(dataset_name)
        self._thing_contiguous_id_to_dataset_id = {
            v: k for k, v in self._metadata.thing_dataset_id_to_contiguous_id.items()
        }
        self._stuff_contiguous_id_to_dataset_id = {
            v: k for k, v in self._metadata.stuff_dataset_id_to_contiguous_id.items()
        }

        self._output_dir = output_dir
        if self._output_dir is not None:
            PathManager.mkdirs(self._output_dir)

        self.write_out_confidence = write_out_confidence
        if self.write_out_confidence:
            os.makedirs(os.path.join(self._output_dir, 'panoptic','classConfidence'), exist_ok=True)
            os.makedirs(os.path.join(self._output_dir, 'panoptic', 'instanceConfidence'), exist_ok=True)

        self.save_colored_pred = save_colored_pred if self._output_dir is not None else False
        if self.save_colored_pred:
            # Initialize IdGenerator with categories from metadata 
            self.categories_dict = {
                i: dict(id=i, isthing=this_cls in self._metadata.thing_classes, name=this_cls, color=self._metadata.stuff_colors[i]) 
                for i, this_cls in enumerate(self._metadata.stuff_classes)
            }

        self.inference_only = inference_only

    def reset(self):
        self._predictions = []

    def _convert_category_id(self, segment_info):
        isthing = segment_info.pop("isthing", None)
        if isthing is None:
            # the model produces panoptic category id directly. No more conversion needed
            return segment_info
        if isthing is True:
            segment_info["category_id"] = self._thing_contiguous_id_to_dataset_id[
                segment_info["category_id"]
            ]
        else:
            segment_info["category_id"] = self._stuff_contiguous_id_to_dataset_id[
                segment_info["category_id"]
            ]
        return segment_info

    def process(self, inputs, outputs):
        from panopticapi.utils import id2rgb

        for input, output in zip(inputs, outputs):
            panoptic_img, panoptic_class_conf, panoptic_instance_conf, segments_info = output["panoptic_seg"]
            panoptic_img = panoptic_img.cpu().numpy()
            if self.write_out_confidence:
                panoptic_class_conf = 255 * panoptic_class_conf.cpu().numpy()
                panoptic_instance_conf = 255 * panoptic_instance_conf.cpu().numpy()
            if segments_info is None:
                # If "segments_info" is None, we assume "panoptic_img" is a
                # H*W int32 image storing the panoptic_id in the format of
                # category_id * label_divisor + instance_id. We reserve -1 for
                # VOID label, and add 1 to panoptic_img since the official
                # evaluation script uses 0 for VOID label.
                label_divisor = self._metadata.label_divisor
                segments_info = []
                for panoptic_label in np.unique(panoptic_img):
                    if panoptic_label == -1:
                        # VOID region.
                        continue
                    pred_class = panoptic_label // label_divisor
                    isthing = (
                        pred_class in self._metadata.thing_dataset_id_to_contiguous_id.values()
                    )
                    segments_info.append(
                        {
                            "id": int(panoptic_label) + 1,
                            "category_id": int(pred_class),
                            "isthing": bool(isthing),
                        }
                    )
                # Official evaluation script uses 0 for VOID label.
                panoptic_img += 1

            if self.save_colored_pred:
                # convert segments_info with self.id_generator to colorfull image
                color_img = np.zeros((panoptic_img.shape[0], panoptic_img.shape[1], 3), dtype=np.uint8)
                id_generator = IdGenerator(self.categories_dict)
                for seg in segments_info:
                    category_id = seg['category_id']
                    cityscapes_id, cityscapes_color = id_generator.get_id_and_color(category_id)
                    color_img[panoptic_img == seg['id']] = cityscapes_color
                # save the color image
                output_dir = os.path.join(self._output_dir, 'panoptic', 'cityscapes_colors')
                os.makedirs(output_dir, exist_ok=True)
                save_path = os.path.join(output_dir, os.path.basename(input["file_name"]))
                color_img_pil = Image.fromarray(color_img) # Reverse the last channel to convert RGB to BGR
                color_img_pil.save(save_path)

            file_name = os.path.basename(input["file_name"])
            file_name_png = os.path.splitext(file_name)[0] + ".png"
            segments_info = [self._convert_category_id(x) for x in segments_info]
            with io.BytesIO() as out:
                Image.fromarray(id2rgb(panoptic_img)).save(out, format="PNG")
                png_string = out.getvalue()
            if self.write_out_confidence:
                Image.fromarray(panoptic_class_conf.astype(np.uint8)).save(os.path.join(
                    output_dir, 'panoptic', 'classConfidence', file_name.replace('.png', '_class_confidence.png')))
                Image.fromarray(panoptic_instance_conf.astype(np.uint8)).save(os.path.join(
                    output_dir, 'panoptic', 'instanceConfidence', file_name.replace('.png', '_instance_confidence.png')))
            self._predictions.append(
                {
                    "image_id": input["image_id"],
                    "file_name": file_name_png,
                    "png_string": png_string,
                    "segments_info": segments_info,
                }
            )

    def evaluate(self):
        comm.synchronize()

        self._predictions = comm.gather(self._predictions)
        self._predictions = list(itertools.chain(*self._predictions))
        if not comm.is_main_process():
            return

        if self.inference_only:
            output_dir = self._output_dir
            pred_folder = os.path.join(output_dir, 'panoptic', 'labelIds')
    
            logger.info("Writing panoptic predictions to: {}".format(pred_folder))
            os.makedirs(pred_folder, exist_ok=True)

            for p in self._predictions:
                with open(os.path.join(pred_folder, p["file_name"]), "wb") as f:
                    f.write(p.pop("png_string"))

            # Save JSON
            predictions_json = os.path.join(pred_folder, "predictions.json")
            with open(predictions_json, "w") as f:
                json.dump({"annotations": self._predictions}, f)

            return OrderedDict()

        # PanopticApi requires local files
        gt_json = PathManager.get_local_path(self._metadata.panoptic_json)
        gt_folder = PathManager.get_local_path(self._metadata.panoptic_root)

        with tempfile.TemporaryDirectory(prefix="panoptic_eval") as pred_dir:
            output_dir = self._output_dir or pred_dir
            logger.info("Writing all panoptic predictions to {} ...".format(output_dir))
            pred_folder=os.path.join(output_dir, 'panoptic', 'labelIds')
            os.makedirs(pred_folder, exist_ok=True)
            for p in self._predictions:
                with open(os.path.join(output_dir, 'panoptic', 'labelIds', p["file_name"]), "wb") as f:
                    f.write(p.pop("png_string"))

            with open(gt_json, "r") as f:
                json_data = json.load(f)
            json_data["annotations"] = self._predictions

            predictions_json = os.path.join(pred_folder, "predictions.json")
            with PathManager.open(predictions_json, "w") as f:
                f.write(json.dumps(json_data))

            from panopticapi.evaluation import pq_compute

            with contextlib.redirect_stdout(io.StringIO()):
                pq_res = pq_compute(
                    gt_json,
                    PathManager.get_local_path(predictions_json),
                    gt_folder=gt_folder,
                    pred_folder=pred_folder,
                )

        res = {}
        res["PQ"] = 100 * pq_res["All"]["pq"]
        res["SQ"] = 100 * pq_res["All"]["sq"]
        res["RQ"] = 100 * pq_res["All"]["rq"]
        res["PQ_th"] = 100 * pq_res["Things"]["pq"]
        res["SQ_th"] = 100 * pq_res["Things"]["sq"]
        res["RQ_th"] = 100 * pq_res["Things"]["rq"]
        res["PQ_st"] = 100 * pq_res["Stuff"]["pq"]
        res["SQ_st"] = 100 * pq_res["Stuff"]["sq"]
        res["RQ_st"] = 100 * pq_res["Stuff"]["rq"]

        results = OrderedDict({"panoptic_seg": res})
        _print_panoptic_results(pq_res)

        return results


def _print_panoptic_results(pq_res):
    headers = ["", "PQ", "SQ", "RQ", "#categories"]
    data = []
    for name in ["All", "Things", "Stuff"]:
        row = [name] + [pq_res[name][k] * 100 for k in ["pq", "sq", "rq"]] + [pq_res[name]["n"]]
        data.append(row)
    table = tabulate(
        data, headers=headers, tablefmt="pipe", floatfmt=".3f", stralign="center", numalign="center"
    )
    logger.info("Panoptic Evaluation Results:\n" + table)


if __name__ == "__main__":
    from detectron2.utils.logger import setup_logger

    logger = setup_logger()
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-json")
    parser.add_argument("--gt-dir")
    parser.add_argument("--pred-json")
    parser.add_argument("--pred-dir")
    args = parser.parse_args()

    from panopticapi.evaluation import pq_compute

    with contextlib.redirect_stdout(io.StringIO()):
        pq_res = pq_compute(
            args.gt_json, args.pred_json, gt_folder=args.gt_dir, pred_folder=args.pred_dir
        )
        _print_panoptic_results(pq_res)