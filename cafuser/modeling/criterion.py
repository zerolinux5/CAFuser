# ------------------------------------------------------------------------------
# Reference: https://github.com/SHI-Labs/OneFormer/blob/main/oneformer/modeling/criterion.py
# ------------------------------------------------------------------------------

import logging

import torch
import torch.nn.functional as F
from torch import nn

from detectron2.utils.comm import get_world_size
from detectron2.projects.point_rend.point_features import (
    get_uncertain_point_coords_with_randomness,
    point_sample,
)

from oneformer.utils.misc import is_dist_avail_and_initialized, nested_tensor_from_tensor_list
from oneformer.utils import box_ops
from oneformer.modeling.criterion import dist_collect, dice_loss, sigmoid_ce_loss, calculate_uncertainty
import torch.distributed as dist
import diffdist.functional as diff_dist
import numpy as np

sigmoid_ce_loss_jit = torch.jit.script(
    sigmoid_ce_loss
)

dice_loss_jit = torch.jit.script(
    dice_loss
)

class SetCriterion(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses,
                 num_points, oversample_ratio, importance_sample_ratio, contrast_temperature=None, condition_temperature=None):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)
        self.cross_entropy = nn.CrossEntropyLoss()

        # pointwise mask loss parameters
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio
        self.contrast_temperature = contrast_temperature
        if self.contrast_temperature is not None:
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / contrast_temperature))
            self.contrast_logit_scale = self.logit_scale
        self.condition_temperature = condition_temperature
        if self.condition_temperature is not None:
            self.condition_logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / condition_temperature))
    
    def generate_labels(self, batch_size, image_x):
        if is_dist_avail_and_initialized():
            return torch.arange(batch_size, dtype=torch.long, device=image_x.device) + batch_size * dist.get_rank()
        else:
            return torch.arange(batch_size, dtype=torch.long, device=image_x.device)

    def normalize_features(self, features):
        return F.normalize(features.flatten(1), dim=-1)

    def compute_logits(self, image_x, text_x):
        if is_dist_avail_and_initialized():
            logits_per_img = image_x @ dist_collect(text_x).t()
            logits_per_text = text_x @ dist_collect(image_x).t()
        else:
            logits_per_img = image_x @ text_x.t()
            logits_per_text = text_x @ image_x.t()
        return logits_per_img, logits_per_text

    def compute_loss(self, logits_per_img, logits_per_text, labels, logit_scale):
        logit_scale = torch.clamp(logit_scale.exp(), max=100)
        loss_img = self.cross_entropy(logits_per_img * logit_scale, labels)
        loss_text = self.cross_entropy(logits_per_text * logit_scale, labels)
        return loss_img + loss_text

    def loss_condition(self, outputs, targets, indices, num_masks):
        assert "condition_contrastive_logits" in outputs
        assert "condition_texts" in outputs
        
        image_x = outputs["condition_contrastive_logits"].float()
        text_x = outputs["condition_texts"]

        batch_size = image_x.shape[0]
        labels = self.generate_labels(batch_size, image_x)
        
        image_x = self.normalize_features(image_x)
        text_x = self.normalize_features(text_x)

        logits_per_img, logits_per_text = self.compute_logits(image_x, text_x)
        loss_condition = self.compute_loss(logits_per_img, logits_per_text, labels, self.condition_logit_scale)

        return {"loss_condition": loss_condition}

    def loss_contrastive(self, outputs, targets, indices, num_masks):
        assert "contrastive_logits" in outputs
        assert "texts" in outputs
        
        image_x = outputs["contrastive_logits"].float()
        text_x = outputs["texts"]

        batch_size = image_x.shape[0]
        labels = self.generate_labels(batch_size, image_x)
        
        image_x = self.normalize_features(image_x)
        text_x = self.normalize_features(text_x)

        logits_per_img, logits_per_text = self.compute_logits(image_x, text_x)
        loss_contrastive = self.compute_loss(logits_per_img, logits_per_text, labels, self.contrast_logit_scale)

        return {"loss_contrastive": loss_contrastive}

    def loss_modality(self, outputs, targets, indices, num_masks):
        assert "modality_logits" in outputs
        assert "modality_labels" in outputs

        modality_logits = outputs["modality_logits"]
        modality_labels = outputs["modality_labels"]

        loss_modality = 0
        for modality in modality_logits.keys():
            for level in modality_logits[modality].keys():
                loss_modality += F.cross_entropy(modality_logits[modality][level], modality_labels[modality])

        return {"loss_modality": loss_modality}


    def loss_labels(self, outputs, targets, indices, num_masks):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"].float()

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o
        
        ce_weight = torch.full(
            src_logits.shape[:2], self.eos_coef, dtype=torch.float32, device=src_logits.device
        )
        ce_weight[idx] = torch.tensor(1.).to(target_classes.device)

        # Deprecated: loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight, reduce=False, reduction="none")
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight, reduction="none")

        loss_ce = loss_ce.sum(1) / ce_weight.sum()
        loss_ce = loss_ce.sum()
        losses = {"loss_ce": loss_ce}
        return losses
    
    def loss_masks(self, outputs, targets, indices, num_masks):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # No need to upsample predictions as we are using normalized coordinates :)
        # N x 1 x H x W
        src_masks = src_masks[:, None]
        target_masks = target_masks[:, None]

        with torch.no_grad():
            # sample point_coords
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                lambda logits: calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )
            # get gt labels
            point_labels = point_sample(
                target_masks,
                point_coords,
                align_corners=False,
            ).squeeze(1)

        point_logits = point_sample(
            src_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)

        losses = {
            "loss_mask": sigmoid_ce_loss_jit(point_logits, point_labels, num_masks),
            "loss_dice": dice_loss_jit(point_logits, point_labels, num_masks),
        }

        del src_masks
        del target_masks
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_masks):
        loss_map = {
            'labels': self.loss_labels,
            'masks': self.loss_masks,
            'contrastive': self.loss_contrastive,
            'condition': self.loss_condition,
            'modality': self.loss_modality,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_masks)

    def forward(self, outputs, targets):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_masks = sum(len(t["labels"]) for t in targets)
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_masks))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == "contrastive" or loss == "condition" or loss == "modality": 
                        continue
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_masks)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses

    def __repr__(self):
        head = "Criterion " + self.__class__.__name__
        body = [
            "matcher: {}".format(self.matcher.__repr__(_repr_indent=8)),
            "losses: {}".format(self.losses),
            "weight_dict: {}".format(self.weight_dict),
            "num_classes: {}".format(self.num_classes),
            "eos_coef: {}".format(self.eos_coef),
            "num_points: {}".format(self.num_points),
            "oversample_ratio: {}".format(self.oversample_ratio),
            "importance_sample_ratio: {}".format(self.importance_sample_ratio),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
