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
from ultralytics import YOLO
from home_robot.core.abstract_perception import PerceptionModule
from home_robot.core.interfaces import Observations
import torch


PARENT_DIR = Path(__file__).resolve().parent
MOBILE_SAM_CHECKPOINT_PATH = str(PARENT_DIR / "pretrained_wt" / "mobile_sam.pt")
CLASSES = [
    "action_figure", "android_figure", "apple", "backpack", "baseballbat",
    "basket", "basketball", "bath_towel", "battery_charger", "board_game",
    "book", "bottle", "bowl", "box", "bread", "bundt_pan", "butter_dish",
    "c-clamp", "cake_pan", "can", "can_opener", "candle", "candle_holder",
    "candy_bar", "canister", "carrying_case", "casserole", "cellphone", "clock",
    "cloth", "credit_card", "cup", "cushion", "dish", "doll", "dumbbell", "egg",
    "electric_kettle", "electronic_cable", "file_sorter", "folder", "fork",
    "gaming_console", "glass", "hammer", "hand_towel", "handbag", "hard_drive",
    "hat", "helmet", "jar", "jug", "kettle", "keychain", "knife", "ladle", "lamp",
    "laptop", "laptop_cover", "laptop_stand", "lettuce", "lunch_box",
    "milk_frother_cup", "monitor_stand", "mouse_pad", "multiport_hub",
    "newspaper", "pan", "pen", "pencil_case", "phone_stand", "picture_frame",
    "pitcher", "plant_container", "plant_saucer", "plate", "plunger", "pot",
    "potato", "ramekin", "remote", "salt_and_pepper_shaker", "scissors",
    "screwdriver", "shoe", "soap", "soap_dish", "soap_dispenser", "spatula",
    "spectacles", "spicemill", "sponge", "spoon", "spray_bottle", "squeezer",
    "statue", "stuffed_toy", "sushi_mat", "tape", "teapot", "tennis_racquet",
    "tissue_box", "toiletry", "tomato", "toy_airplane", "toy_animal", "toy_bee",
    "toy_cactus", "toy_construction_set", "toy_fire_truck", "toy_food",
    "toy_fruits", "toy_lamp", "toy_pineapple", "toy_rattle", "toy_refrigerator",
    "toy_sink", "toy_sofa", "toy_swing", "toy_table", "toy_vehicle", "tray",
    "utensil_holder_cup", "vase", "video_game_cartridge", "watch", "watering_can",
    "wine_bottle", "bathtub", "bed", "bench", "cabinet", "chair", "chest_of_drawers",
    "couch", "counter", "filing_cabinet", "hamper", "serving_cart", "shelves",
    "shoe_rack", "sink", "stand", "stool", "table", "toilet", "trunk", "wardrobe",
    "washer_dryer"
]



class YOLOPerception(PerceptionModule):
    def __init__(
        self,
        checkpoint_file: Optional[str] = MOBILE_SAM_CHECKPOINT_PATH,
        sem_gpu_id=0,
        verbose: bool = False,
        confidence_threshold: Optional[float] = 0.7,
        
    ):
        """Loads a YOLO model for object detection and instance segmentation

        Arguments:
            yolo_model_id: one of "yolo_world/l" or "yolo_world/s" for large or small
            checkpoint_file: path to model checkpoint
            sem_gpu_id: GPU ID to load the model on, -1 for CPU
            verbose: whether to print out debug information
        """
        yolo_model_id="yolo_world/s",
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

        self.model = YOLO(model='yolov8s-world.pt')
        vocab = CLASSES
        self.model.set_classes(vocab)
        # Freeze the YOLO model's parameters
        for param in self.model.parameters():
            param.requires_grad = False
        # self.model.save("custom_yolov8s.pt")
        # self.model = YOLO(model='custom_yolov8s.pt')

         #check format of vocabulary
        self.confidence_threshold = confidence_threshold
        self.sam_model = SAM(checkpoint_file)
        # Freeze the SAM model's parameters
        for param in self.sam_model.parameters():
            param.requires_grad = False
        

    def predict(
        self,
        obs: Observations,
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

        nms_threshold=0.8
        # if draw_instance_predictions:
        #     raise NotImplementedError
        
        # convert to uint8 instead of silently failing by returning no instances
        images_tensor = obs["head_rgb"] #get images list from obeservation

        images = [images_tensor[i].cpu().numpy() for i in range(images_tensor.size(0))]   

        # if not image.dtype == np.uint8:
        #     if image.max() <= 1.0:
        #         image = image * 255.0
        #     image = image.astype(np.uint8)

        height, width, _ = images[0].shape
        results = self.model(images,  conf = self.confidence_threshold, stream = True)

        semantic_masks=[]

        for result, img in zip(results, images):

            boxes = [iresult.boxes.xyxy.tolist() for iresult in result]  # List of bounding boxes for each image
            batch_boxes = [box for boxes_list in boxes for box in boxes_list]  # Flatten the list of boxes

            input_boxes = np.array(batch_boxes)
            if len(input_boxes) == 0:
                height, width, _ = img.shape
                semantic_mask = np.zeros((height, width, 1))
            else:
                class_id=result.boxes.cls.cpu().numpy()
                sam_outputs = self.sam_model.predict(stream=True, source=img, bboxes=input_boxes, points=None, labels=None)
                sam_output=next(sam_outputs)
                result_masks=sam_output.masks
                height, width, _ = img.shape
                semantic_mask, instance_mask = self.overlay_masks(
                    result_masks.data, class_id, (height, width)
                )
        semantic_masks.append(semantic_mask)

        # obs.semantic = semantic_map.astype(int)
        # obs.instance = instance_map.astype(int)
        # if obs.task_observations is None:
        #     obs.task_observations = dict()
        # obs.task_observations["instance_map"] = instance_map
        # obs.task_observations["instance_classes"] = detections.class_id
        # obs.task_observations["instance_scores"] = detections.confidence
        # obs.task_observations["semantic_frame"] = None
        return semantic_masks
    

    # Prompting SAM with detected boxes
    # def segment(self, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    #     """
    #     Get masks for all detected bounding boxes using SAM
    #     Arguments:
    #         image: image of shape (H, W, 3)
    #         xyxy: bounding boxes of shape (N, 4) in (x1, y1, x2, y2) format
    #     Returns:
    #         masks: masks of shape (N, H, W)
    #     """
    #     result_masks = []
    #     for box in xyxy:
    #         sam_output= self.sam_model.predict(source=image,
    #             bboxes=box)
    #         mask_object = sam_output[0].masks
    #         mask_tensor = mask_object.data.cpu()
    #         masks = mask_tensor.numpy()
    #         masks = masks.squeeze(axis=0)
    #         result_masks.append(masks)

    #     return np.array(result_masks)
    
    def overlay_masks(self,
        masks: np.ndarray, class_idcs: np.ndarray, shape: Tuple[int, int]
    ) -> np.ndarray:
        """Overlays the masks of objects
        Masks are overlaid based on the order of class_idcs.
        """
        semantic_mask = np.zeros((*shape, 1))
        instance_mask = np.zeros(shape)

        for mask_idx, class_idx in enumerate(class_idcs):
            if mask_idx < len(masks):
                mask = masks[mask_idx].cpu().numpy()
                semantic_mask[mask.astype(bool)] = class_idx 
                instance_mask[mask.astype(bool)] = mask_idx
            else:
                break

        return semantic_mask, instance_mask
    

    # def filter_depth(
    #     mask: np.ndarray, depth: np.ndarray, depth_threshold: Optional[float] = None
    # ) -> np.ndarray:
    #     """Filter object mask by depth.

    #     Arguments:
    #         mask: binary object mask of shape (height, width)
    #         depth: depth map of shape (height, width)
    #         depth_threshold: restrict mask to (depth median - threshold, depth median + threshold)
    #     """
    #     md = np.median(depth[mask == 1])  # median depth
    #     if md == 0:
    #         # Remove mask if more than half of points has invalid depth
    #         filter_mask = np.ones_like(mask, dtype=bool)
    #     elif depth_threshold is not None:
    #         # Restrict objects to depth_threshold
    #         filter_mask = (depth >= md + depth_threshold) | (depth <= md - depth_threshold)
    #     else:
    #         filter_mask = np.zeros_like(mask, dtype=bool)
    #     mask_out = mask.copy()
    #     mask_out[filter_mask] = 0.0

    #     return mask_out





