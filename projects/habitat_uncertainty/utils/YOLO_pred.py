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
from home_robot.core.abstract_perception import PerceptionModule
from home_robot.core.interfaces import Observations
import torch
from habitat.core.logging import logger
# from nvitop import Device
# import gc
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
        confidence_threshold: Optional[float] = 0.02,
        
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

        self.confidence_threshold = confidence_threshold

        self.model.cuda()
        torch.cuda.empty_cache()

    def create_gaussian_mask(self, height, width, boxes, max_sigma=25):
        if len(boxes) > 0:
            # print("Detected")
            boxes=boxes[0]
            x_min, y_min, x_max, y_max = boxes
            center = ((boxes[0] + boxes[2]) // 2, (boxes[1] + boxes[3]) // 2)    

            size = (x_max - x_min, y_max - y_min)
            sigma_x = min(size[0] / 8, max_sigma)
            sigma_y = min(size[1] / 8, max_sigma)

            y, x = np.ogrid[0:height, 0:width]
            distance = np.sqrt((x - center[0])**2 / (2 * (sigma_x**2) + 1e-6) + (y - center[1])**2 / (2 * (sigma_y**2) + 1e-6))        
            gaussian = np.exp(-distance)
            return np.expand_dims(gaussian / gaussian.max(), axis=2) 
        else:
            # print("Nothing Detected")
            return np.zeros((160, 120, 1))

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
        torch.cuda.empty_cache()
        # start_time = time.time()  
        nms_threshold=0.8
            
        images_tensor = obs["head_rgb"] 
        obj_class_ids = obs["yolo_object_sensor"].cpu().numpy().flatten()
        rec_class_ids = obs["yolo_start_receptacle_sensor"].cpu().numpy().flatten()
        batch_size = images_tensor.shape[0]
        images = [images_tensor[i].cpu().numpy() for i in range(images_tensor.size(0))]   

        height, width, _ = images[0].shape
        results = list(self.model(images, conf=self.confidence_threshold, stream=True, verbose=False))
        obj_semantic_masks = []
        rec_semantic_masks = []

        for idx, result in enumerate(results):
            class_ids = result.boxes.cls.cpu().numpy()
            input_boxes = result.boxes.xyxy.cpu().numpy()

            obj_mask_idx = np.isin(class_ids, obj_class_ids[idx])
            rec_mask_idx = np.isin(class_ids, rec_class_ids[idx])

            obj_boxes = input_boxes[obj_mask_idx]
            rec_boxes = input_boxes[rec_mask_idx]

            obj_semantic_mask = self.create_gaussian_mask(height, width, obj_boxes)
            rec_semantic_mask = self.create_gaussian_mask(height, width, rec_boxes)
            
            
            # combined_image = np.concatenate([obj_semantic_mask * 255, rec_semantic_mask * 255], axis=1)
            cv2.imwrite('obj_mask{}.png'.format(idx), [obj_semantic_mask * 255])
            cv2.imwrite('rec_mask{}.png'.format(idx), [rec_semantic_mask * 255])
            cv2.imwrite('orig_image{}.png'.format(idx), cv2.cvtColor(images[idx], cv2.COLOR_BGR2RGB) )


            # obj_semantic_mask = cv2.resize(obj_semantic_mask, (120, 160), interpolation=cv2.INTER_NEAREST)
            # rec_semantic_mask = cv2.resize(rec_semantic_mask, (120, 160), interpolation=cv2.INTER_NEAREST)
            
            # obj_semantic_mask = np.expand_dims(obj_semantic_mask, axis=-1)
            # rec_semantic_mask = np.expand_dims(rec_semantic_mask, axis=-1)
            obj_semantic_masks.append(obj_semantic_mask)
            rec_semantic_masks.append(rec_semantic_mask)

        torch.cuda.empty_cache()
        obj_semantic_masks = np.array(obj_semantic_masks)
        rec_semantic_masks = np.array(rec_semantic_masks)

        return obj_semantic_masks, rec_semantic_masks






