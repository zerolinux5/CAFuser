import cv2
import numpy as np
import torch
from .processing.utils import load_muses_calibration_data, load_meta_data
from .processing.lidar_processing import load_lidar_projection
from .processing.radar_processing import load_radar_projection
from .processing.event_camera_processing import load_event_camera_projection
from matplotlib import colormaps as cm
import random
import copy
import os

class Augmentations:
    def __init__(self, aug_cfg):
        random.seed(aug_cfg.SEED)
        np.random.seed(aug_cfg.SEED)
        self.aug_cfg = aug_cfg
        self.apply_patch_augmentation = self.aug_cfg.PATCH_AUGMENTATION.ENABLED
        self.apply_extreme_augmentation = self.aug_cfg.EXTREME_AUGMENTATION.ENABLED
        self.apply_mixup = self.aug_cfg.MIXUP.ENABLED
        self.apply_cutmix = self.aug_cfg.CUTMIX.ENABLED

    def patch_augmentation(self, modality_images):
        """
        Apply patch augmentation to the input modality images.
        For scale in SCALES:
            - Randomly select a patch of size (scale * h, scale * w)
            - Replace the patch with Gaussian noise
        """
        for modality in modality_images:
            if random.random() > self.aug_cfg.PATCH_AUGMENTATION.PROB_THRESHOLD:
                continue
            scales = self.aug_cfg.PATCH_AUGMENTATION.SCALES
            for scale in scales:
                h, w = modality_images[modality].shape[:2]
                patch_h, patch_w = int(h * scale), int(w * scale)
                y, x = random.randint(0, h - patch_h), random.randint(0, w - patch_w)
                # Generate Gaussian noise
                noise = np.random.random((patch_h, patch_w, modality_images[modality].shape[-1])) * 255
                noise = np.round(noise)
                # Replace patch with Gaussian noise
                modality_images[modality][y:y + patch_h, x:x + patch_w] = noise.astype(modality_images[modality].dtype)
        return modality_images

    def extreme_augmentation(self, modality_images):
        for modality in modality_images:
            if random.random() > self.aug_cfg.EXTREME_AUGMENTATION.PROB_THRESHOLD:
                continue
            h, w = modality_images[modality].shape[:2]
            num_patches = self.aug_cfg.EXTREME_AUGMENTATION.NUM_PATCHES
            patch_size = self.aug_cfg.EXTREME_AUGMENTATION.PATCH_SIZE
            noisy_image = np.random.random((modality_images[modality].shape)) * 255
            noisy_image = np.round(noisy_image).astype(modality_images[modality].dtype)
            for _ in range(num_patches):
                patch_h, patch_w = int(h * patch_size), int(w * patch_size)
                y, x = random.randint(0, h - patch_h), random.randint(0, w - patch_w)
                noisy_image[y:y+patch_h, x:x+patch_w] = modality_images[modality][y:y+patch_h, x:x+patch_w]
            modality_images[modality] = noisy_image
        return modality_images

    def mixup(self, modality_images):
        alpha = self.aug_cfg.MIXUP.ALPHA
        modalities = list(modality_images.keys())
        augmented_images = copy.deepcopy(modality_images)

        # Determine number of pairs to mix
        num_modalities = len(modalities)
        if num_modalities < 2:
            return augmented_images  # Nothing to mix

        # Shuffle modalities and pair them
        random.shuffle(modalities)
        pairs = []
        for i in range(0, num_modalities - 1, 2):
            pairs.append((modalities[i], modalities[i + 1]))

        # Apply MixUp to each pair
        for (mod1, mod2) in pairs:
            if random.random() > self.aug_cfg.MIXUP.PROB_THRESHOLD:
                continue
            lam = np.random.beta(alpha, alpha)
            augmented_images[mod1] = lam * modality_images[mod1] + (1 - lam) * modality_images[mod2]
            augmented_images[mod2] = lam * modality_images[mod2] + (1 - lam) * modality_images[mod1]

        return augmented_images

    def cutmix(self, modality_images):
        """
        CutMix augmentation.
        Randomly selects two modalities and mixes them. The mixing is done by cutting a random patch from one modality
        and replacing it with a patch from the other modality. The mixing is done for all pairs of modalities.
        """
        modalities = list(modality_images.keys())
        augmented_images = copy.deepcopy(modality_images)
        scales = self.aug_cfg.CUTMIX.SCALES

        # Determine number of pairs to apply CutMix
        num_modalities = len(modalities)
        if num_modalities < 2:
            return augmented_images  # Nothing to cutmix

        # Shuffle modalities and pair them
        random.shuffle(modalities)
        pairs = []
        for i in range(0, num_modalities - 1, 2):
            pairs.append((modalities[i], modalities[i + 1]))

        # Apply CutMix to each pair
        for (mod1, mod2) in pairs:
            if random.random() > self.aug_cfg.CUTMIX.PROB_THRESHOLD:
                continue
            for scale in scales:
                h, w = modality_images[mod1].shape[:2]
                cut_h, cut_w = int(h * scale), int(w * scale)
                y, x = random.randint(0, h - cut_h), random.randint(0, w - cut_w)
                augmented_images[mod1][y:y + cut_h, x:x + cut_w] = modality_images[mod2][y:y + cut_h, x:x + cut_w]
                augmented_images[mod2][y:y + cut_h, x:x + cut_w] = modality_images[mod1][y:y + cut_h, x:x + cut_w]
        return augmented_images

    def __call__(self, modality_images):
        types = [modality_images[modality].dtype for modality in modality_images]
        if self.apply_patch_augmentation:
            modality_images = self.patch_augmentation(modality_images)
        if self.apply_extreme_augmentation:
            modality_images = self.extreme_augmentation(modality_images)
        if self.apply_mixup:
            modality_images = self.mixup(modality_images)
        if self.apply_cutmix:
            modality_images = self.cutmix(modality_images)
        for img_type, modality in zip(types, modality_images.keys()):
            modality_images[modality] = modality_images[modality].astype(img_type)
        return modality_images


class MUSES_loader:
    """
    Helper class to load the MUSES dataset.

    Args:
        modalities (dict): Dictionary with the modalities to load.
        img_format (str): Image format to use.
        load_projected (bool): Whether to load the projected images.
        calib_data (dict): Calibration data.
        relevant_meta_data (dict): Relevant metadata.
        muses_data_root (str): Root of the MUSES dataset.
    """
    def __init__(self,
                modalities_cfg=None,
                muses_data_root=None,
                is_train=True,
                img_format="RGB",
                target_shape=(1080,1920,3),
                missing_mod = [None]):

        self.modalities = modalities_cfg
        self.main_modality = self.modalities.MAIN_MODALITY.upper()
        modality_order = self.modalities.ORDER
        if modality_order != ["CAMERA", "LIDAR", "EVENT_CAMERA", "RADAR", "REF_IMAGE"]:
            raise NotImplementedError

        self.load_projected = {modality: self.modalities[modality].LOAD_PROJECTED for modality in self.modalities if modality in self.modalities.ORDER}
        self.muses_data_root = muses_data_root
        self.meta_data = load_meta_data(self.muses_data_root)
        self.calib_data = load_muses_calibration_data(self.muses_data_root)
        self.is_train = is_train
        self.img_format = img_format
        self.target_shape = target_shape
        if self.is_train:
            self.augmentations = Augmentations(self.modalities.AUGMENTATIONS)

        self.missing_mod = missing_mod
        assert self.modalities[self.main_modality].LOAD, f"Main modality {self.main_modality} is not loaded."

    def adapt_mod_file_name(self, modality, orig_file_name, projected=False):
        if projected:
            modality_folder = self.modalities[modality].PROJECTED.FOLDER
            extension = self.modalities[modality].PROJECTED.EXTENSION
        else:
            extension = self.modalities[modality].EXTENSION
            modality_folder = self.modalities[modality].FOLDER
        file_name = orig_file_name.replace(self.modalities.CAMERA.FOLDER, modality_folder)
        file_name = file_name.replace(self.modalities.CAMERA.EXTENSION, extension)
        return file_name

    def load_modality_from_raw(self, modality, modality_file_name, scene_meta_data, dtype=np.float32):
        if modality == "LIDAR":
            modality_image = load_lidar_projection(modality_file_name, self.calib_data, scene_meta_data,
                                                   motion_compensation=self.modalities[modality].MOTION_COMPENSATION,
                                                   muses_root=self.muses_data_root,
                                                   enlarge_lidar_points=self.modalities[modality].DILATION.ENABLED,
                                                   dialtion_kernal=self.modalities[modality].DILATION.KERNAL)
        elif modality == "RADAR":
            modality_image = load_radar_projection(modality_file_name, self.calib_data, scene_meta_data,
                                                   motion_compensation=self.modalities[modality].MOTION_COMPENSATION,
                                                   muses_root=self.muses_data_root,
                                                   enlarge_radar_points=self.modalities[modality].DILATION.ENABLED,
                                                   dialtion_kernal=self.modalities[modality].DILATION.KERNAL,
                                                   intensity_threshold=self.modalities[modality].INTENSITY_THRESHOLD,
                                                   max_distance=self.modalities[modality].MAX_DISTANCE)
        elif modality == "EVENT_CAMERA":
            modality_image = load_event_camera_projection(modality_file_name, self.calib_data,
                                                   enlarge_event_camera_points=self.modalities[modality].DILATION.ENABLED,
                                                   dialtion_kernal=self.modalities[modality].DILATION.KERNAL)
        elif modality == "CAMERA" or modality == "REF_IMAGE":
            modality_image = cv2.imread(modality_file_name, cv2.IMREAD_UNCHANGED)
        else:
            raise ValueError(f"Unknown modality {modality}.")
        return modality_image.astype(dtype)

    def load_projected_modality(self, modality, modality_file_name):
        if self.img_format == "RGB":
            if not os.path.exists(modality_file_name):
                raise FileNotFoundError(f"File {modality_file_name} does not exist.")
            modality_image = cv2.imread(modality_file_name, cv2.IMREAD_UNCHANGED)
        else:
            raise NotImplementedError(f"Image format {self.img_format} not implemented for muses.")
        if modality not in ["CAMERA", "REF_IMAGE"]:
            modality_image = modality_image.astype(np.float32)
        if self.modalities[modality].PROJECTED.SCALE_FACTOR:
            modality_image /= self.modalities[modality].PROJECTED.SCALE_FACTOR
        if self.modalities[modality].PROJECTED.SHIFT_FACTOR:
            modality_image -= self.modalities[modality].PROJECTED.SHIFT_FACTOR
        return modality_image

    def get_modality_file_name(self, dataset_dict, modality, projected=False):
        if modality == "CAMERA":
            modality_file_name = dataset_dict["file_name"]
        elif modality == "REF_IMAGE":
            if "clear/day" in dataset_dict["file_name"]:
                modality_file_name = dataset_dict["file_name"]
            else:
                modality_file_name = self.adapt_mod_file_name(modality, dataset_dict["file_name"], projected)
        else:
            modality_file_name = self.adapt_mod_file_name(modality, dataset_dict["file_name"], projected)
        return modality_file_name

    def should_drop_modality(self, modality):
        drop_prob = self.modalities[modality].RANDOM_DROP
        return np.random.rand() < drop_prob

    def get_condition_meta_data(self, dataset_dict):
        scene_name = dataset_dict['image_id']
        scene_meta_data = self.meta_data.get(scene_name)

        condition_meta_data = {}
        
        condition_mapping = {
            'rain': 'rainy',
            'fog': 'foggy',
            'snow': 'snowy',
            'clear': 'clear'
        }
        
        ground_condition_mapping = {
            'dry': 'dry',
            'wet': 'wet',
            'snow': 'snowy',
        }

        sun_level_mapping = {
            'sunlight': 'sunny',
            'overcast': 'overcast',
            'none': 'none',
            'nan': 'none'
        }
        
        # Extract and map conditions
        condition = condition_mapping.get(scene_meta_data['weather'].lower(), scene_meta_data['weather'].lower())
        ground_condition = ground_condition_mapping.get(scene_meta_data['ground_condition'], scene_meta_data['ground_condition'].lower())
        time_of_day = scene_meta_data['time_of_day'].lower()
        strength = scene_meta_data.get('precipitation_level', 'None').lower()
        precipitation = scene_meta_data.get('precipitation_tag', 'None').lower()
        sun_level = sun_level_mapping.get(str(scene_meta_data.get('sun_level', 'none')).lower())
        if sun_level == 'none':
            if time_of_day == 'night':
                sun_level = 'dark'
            else:
                tunnel = scene_meta_data.get("tunnel")
                if tunnel: 
                    sun_level = 'hidden'
                else:
                    raise Exception(f'This should not happen, need a sun level when at daytime. Scenes: {dataset_dict["image_id"]}')
                

        if strength == 'none' and precipitation == 'none':  
            precipitation_text = 'no precipitation'
        else:
            precipitation_text = f'{strength} {precipitation}'
        
        # Construct the full text description
        text = f"A {condition} driving scene at {time_of_day}time with {precipitation_text}, a {ground_condition} ground and a {sun_level} sky."

        # Populate the condition_meta_data dictionary
        condition_meta_data['condition'] = condition
        condition_meta_data['time_of_day'] = time_of_day
        condition_meta_data['strength'] = strength
        condition_meta_data['precipitation'] = precipitation
        condition_meta_data['ground_condition'] = ground_condition
        condition_meta_data['precipitation_text'] = precipitation_text
        condition_meta_data['sun_level'] = sun_level
        condition_meta_data['text'] = text
        
        return condition_meta_data
        

    def __call__(self, dataset_dict):
        if not all(self.load_projected.values()):
            scene_meta_data = self.meta_data[dataset_dict['image_id']]
        modality_images = {}
        for modality in self.modalities.ORDER:
            if self.modalities[modality].LOAD:
                if self.should_drop_modality(modality) and self.is_train:
                    modality_image = np.zeros(self.target_shape, dtype=np.float32)
                elif not self.is_train and modality in self.missing_mod:
                    modality_image = np.zeros(self.target_shape, dtype=np.float32)
                else:
                    modality_file_name = self.get_modality_file_name(dataset_dict, modality, self.load_projected[modality])
                    if self.load_projected[modality]:
                        modality_image = self.load_projected_modality(modality, modality_file_name)
                    else:
                        modality_image = self.load_modality_from_raw(modality, modality_file_name, scene_meta_data)
                modality_images.update({modality: modality_image})
        for modality in modality_images:
            assert modality_images[modality].shape == modality_images[self.main_modality].shape, \
                (f"Loaded modality images have different shapes: {modality_images[modality].shape} vs "
                    f"{modality_images[self.main_modality].shape}.")

        # Apply augmentations
        if self.is_train:
            modality_images = self.augmentations(modality_images)

        return modality_images