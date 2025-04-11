import logging
import os
import numpy as np
from PIL import Image
from detectron2.evaluation import SemSegEvaluator

from detectron2.data import MetadataCatalog

class MUSESSemSegEvaluator(SemSegEvaluator):
    """
    Evaluator for MUSES semantic segmentation task.
    This class wraps the Detectron2 SemSegEvaluator to handle the inference only
    case for the MUSES test set. To get a folder to be uploaded to the benchmark.
    """

    def __init__(self, dataset_name, distributed=True, output_dir=None, inference_only=False, save_colored_pred=False):
        super().__init__(dataset_name, distributed, output_dir)
        self.inference_only = inference_only
        if self.inference_only:
            self.label_dir = os.path.join(self._output_dir, "semantic", "labelTrainIds")
            os.makedirs(self.label_dir, exist_ok=True)

            # Initialize IdGenerator with categories from metadata
            self.cityscapes_colors = save_colored_pred
            if self.cityscapes_colors:
                self._metadata = MetadataCatalog.get(dataset_name)
                self.categories_dict = {
                    i: dict(id=i, isthing=this_cls in self._metadata.thing_classes, name=this_cls, color=self._metadata.stuff_colors[i])
                    for i, this_cls in enumerate(self._metadata.stuff_classes)
                }
                self.label_dir_color = os.path.join(self._output_dir, "semantic", "cityscapes_colors")
                os.makedirs(self.label_dir_color, exist_ok=True)

    def process(self, inputs, outputs):
        if not self.inference_only:
            return super().process(inputs, outputs)

        for input, output in zip(inputs, outputs):
            output = output["sem_seg"].argmax(dim=0).to(self._cpu_device)
            pred = np.array(output, dtype=int)
            self.save_prediction_as_png(pred, input["file_name"])

    def save_prediction_as_png(self, pred, file_name):
        base_name = os.path.basename(file_name)
        save_filename = base_name.replace('_frame_camera', '')
        save_path = os.path.join(self.label_dir, save_filename)
        pred_image = Image.fromarray(pred.astype(np.uint8))
        pred_image.save(save_path)

        if self.cityscapes_colors:
            pred_color = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
            for i in range(len(self.categories_dict)):
                pred_color[pred == i] = self.categories_dict[i]["color"]
            pred_color_image = Image.fromarray(pred_color)
            pred_color_image.save(os.path.join(self.label_dir_color, save_filename))


    def evaluate(self):
        if not self.inference_only:
            return super().evaluate()

        self._logger.info(f"Writing semantic predictions to: {self.label_dir}")