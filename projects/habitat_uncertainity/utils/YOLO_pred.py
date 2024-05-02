# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import cv2
import time
import supervision as sv
import numpy as np

import torch
from pathlib import Path
from typing import List, Optional, Tuple
from ultralytics import YOLO
from ultralytics import SAM
from home_robot.core.abstract_perception import PerceptionModule
from home_robot.core.interfaces import Observations
import torch
from habitat.core.logging import logger
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
        confidence_threshold: Optional[float] = 0.35,
        
    ):
        """Loads YOLO-open world model for object detection and mobileSAM for segmentation

        Arguments:
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
                f"Loading YOLO model yolov8s-world and MobileSAM with checkpoint={checkpoint_file}"   
            )
        self.model = YOLO(model='yolov8s-world.pt')
        vocab = CLASSES
        self.model.set_classes(vocab)
        # Freeze the YOLO model's parameters
        for param in self.model.parameters():
            param.requires_grad = False

        self.confidence_threshold = confidence_threshold
        self.sam_model = SAM(checkpoint_file)
        # Freeze the SAM model's parameters
        for param in self.sam_model.parameters():
            param.requires_grad = False
        self.model.cuda()
        self.sam_model.cuda()
        torch.cuda.empty_cache()

        

    def predict(
        self,
        obs: Observations,
        depth_threshold: Optional[float] = None,
        draw_instance_predictions: bool = True,
    ) -> Observations:
        """
        Arguments:
            obs.rgb: image of shape (B, H, W, 3) (in RGB order)
            depth_threshold: if specified, the depth threshold per instance

        Returns:
            semantic_masks: segmentation predictions of shape (B, H, W, 1) with each pixel representing class names

        """
        torch.cuda.empty_cache()
                
        images_tensor = obs["head_rgb"] 
        images = [images_tensor[i].cpu().numpy() for i in range(images_tensor.size(0))]   

        height, width, _ = images[0].shape
        #Inference from Yolo-open world model 
        results = self.model(images, conf=self.confidence_threshold, stream=True, verbose=False)

        semantic_masks = []

        for idx, result in enumerate(results):
            img = images[idx] 
            boxes = [iresult.boxes.xyxy.tolist() for iresult in result]
            batch_boxes = [box for boxes_list in boxes for box in boxes_list]

            input_boxes = np.array(batch_boxes)
            if len(input_boxes) == 0:
                height, width, _ = img.shape
                semantic_mask = np.zeros((160, 120, 1))
            else:
                class_ids = result.boxes.cls.cpu().numpy()
                #Inference from mobileSAM model
                sam_outputs = self.sam_model.predict(stream=True, source=img, bboxes=input_boxes, points=None, labels=None, verbose=False)
                sam_output=next(sam_outputs)
                result_masks=sam_output.masks
                height, width, _ = img.shape
                semantic_mask, instance_mask = self.overlay_masks(
                    result_masks.data, class_ids, (height, width)
                )

            torch.cuda.empty_cache()
            semantic_masks.append(semantic_mask)


        semantic_masks = np.array(semantic_masks)
        torch.cuda.empty_cache()
        return semantic_masks
    
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
        del masks, class_idcs, shape

        return semantic_mask, instance_mask




