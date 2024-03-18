# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import cv2
import supervision as sv
import numpy as np
from tqdm import tqdm
from inference.models import YOLOWorld
from ultralytics import SAM
import torch
import torchvision
from pathlib import Path
from typing import List, Optional, Tuple

from home_robot.core.abstract_perception import PerceptionModule
from home_robot.core.interfaces import Observations


PARENT_DIR = Path(__file__).resolve().parent
MOBILE_SAM_CHECKPOINT_PATH = str(PARENT_DIR / "pretrained_wt" / "mobile_sam.pt")



class YOLOPerception(PerceptionModule):
    def __init__(
        self,
        yolo_model_id="yolo_world/l",
        checkpoint_file: str = MOBILE_SAM_CHECKPOINT_PATH,
        sem_gpu_id=0,
        verbose: bool = False,
        confidence_threshold: Optional[float] = None,
        
    ):
        """Loads a YOLO model for object detection and instance segmentation

        Arguments:
            yolo_model_id: one of "yolo_world/l" or "yolo_world/s" for large or small
            checkpoint_file: path to model checkpoint
            sem_gpu_id: GPU ID to load the model on, -1 for CPU
            verbose: whether to print out debug information
        """
        self.verbose = verbose
        if checkpoint_file is None:
            checkpoint_file = str(
                Path(__file__).resolve().parent
                / "pretrained_wt/mobile_sam.pt"
            )
        if self.verbose:
            print(
                f"Loading YOLO model from {yolo_model_id} and MobileSAM with checkpoint={checkpoint_file}"   
            )

        self.cpu_device = torch.device("cpu")
        self.model = YOLOWorld(yolo_model_id)
         #check format of vocabulary
        self.confidence_threshold = confidence_threshold
        self.sam_model = SAM(checkpoint_file)
        

    def predict(
        self,
        obs: Observations,
        vocab: List[str],
        depth_threshold: Optional[float] = None,
        draw_instance_predictions: bool = True,
    ) -> Observations:
        """
        Arguments:
            obs.rgb: image of shape (H, W, 3) (in RGB order - Detic expects BGR)
            obs.depth: depth frame of shape (H, W), used for depth filtering
            depth_threshold: if specified, the depth threshold per instance

        Returns:
            obs.semantic: segmentation predictions of shape (H, W) with
             indices in [0, num_sem_categories - 1]
            obs.task_observations["semantic_frame"]: segmentation visualization
             image of shape (H, W, 3)
        """
        self.model.set_classes(vocab)
        nms_threshold=0.8
        if draw_instance_predictions:
            raise NotImplementedError
        # Predict classes and hyper-param for GroundingDINO
        height, width, _ = obs.rgb.shape

        # convert to uint8 instead of silently failing by returning no instances
        image = obs.rgb
        if not image.dtype == np.uint8:
            if image.max() <= 1.0:
                image = image * 255.0
            image = image.astype(np.uint8)

        depth = obs.depth
        height, width, _ = image.shape

        results = self.model.infer(image, self.confidence_threshold)
        detections = sv.Detections.from_inference(results).with_nms(threshold=0.1)

        nms_idx = (
            torchvision.ops.nms(
                torch.from_numpy(detections.xyxy),
                torch.from_numpy(detections.confidence),
                nms_threshold,
            )
            .numpy()
            .tolist()
        )

        detections.xyxy = detections.xyxy[nms_idx]
        detections.confidence = detections.confidence[nms_idx]
        detections.class_id = detections.class_id[nms_idx]

        # convert detections to masks
        detections.mask = self.segment(image=image, xyxy=detections.xyxy)

        if depth_threshold is not None and obs.depth is not None:
            detections.mask = np.array(
                [
                    self.filter_depth(mask, obs.depth, depth_threshold)
                    for mask in detections.mask
                ]
            )

        semantic_map, instance_map = self.overlay_masks(
            detections.mask, detections.class_id, (height, width)
        )

        # obs.semantic = semantic_map.astype(int)
        # obs.instance = instance_map.astype(int)
        # if obs.task_observations is None:
        #     obs.task_observations = dict()
        # obs.task_observations["instance_map"] = instance_map
        # obs.task_observations["instance_classes"] = detections.class_id
        # obs.task_observations["instance_scores"] = detections.confidence
        # obs.task_observations["semantic_frame"] = None
        return instance_map
    

    # Prompting SAM with detected boxes
    def segment(self, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
        """
        Get masks for all detected bounding boxes using SAM
        Arguments:
            image: image of shape (H, W, 3)
            xyxy: bounding boxes of shape (N, 4) in (x1, y1, x2, y2) format
        Returns:
            masks: masks of shape (N, H, W)
        """
        result_masks = []
        for box in xyxy:
            sam_output= self.sam_model.predict(source=image,
                bboxes=box)
            mask_object = sam_output[0].masks
            mask_tensor = mask_object.data.cpu()
            masks = mask_tensor.numpy()
            masks = masks.squeeze(axis=0)
            result_masks.append(masks)

        return np.array(result_masks)
    
    def overlay_masks(
        masks: np.ndarray, class_idcs: np.ndarray, shape: Tuple[int, int]
        ) -> np.ndarray:
        """Overlays the masks of objects
        Determines the order of masks based on mask size
        """
        mask_sizes = [np.sum(mask) for mask in masks]
        sorted_mask_idcs = np.argsort(mask_sizes)

        semantic_mask = np.zeros(shape)
        instance_mask = -np.ones(shape)
        for i_mask in sorted_mask_idcs[::-1]:  # largest to smallest
            semantic_mask[masks[i_mask].astype(bool)] = class_idcs[i_mask]
            instance_mask[masks[i_mask].astype(bool)] = i_mask

        return semantic_mask, instance_mask
    def filter_depth(
        mask: np.ndarray, depth: np.ndarray, depth_threshold: Optional[float] = None
    ) -> np.ndarray:
        """Filter object mask by depth.

        Arguments:
            mask: binary object mask of shape (height, width)
            depth: depth map of shape (height, width)
            depth_threshold: restrict mask to (depth median - threshold, depth median + threshold)
        """
        md = np.median(depth[mask == 1])  # median depth
        if md == 0:
            # Remove mask if more than half of points has invalid depth
            filter_mask = np.ones_like(mask, dtype=bool)
        elif depth_threshold is not None:
            # Restrict objects to depth_threshold
            filter_mask = (depth >= md + depth_threshold) | (depth <= md - depth_threshold)
        else:
            filter_mask = np.zeros_like(mask, dtype=bool)
        mask_out = mask.copy()
        mask_out[filter_mask] = 0.0

        return mask_out





