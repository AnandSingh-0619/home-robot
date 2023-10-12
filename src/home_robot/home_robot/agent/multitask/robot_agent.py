# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from home_robot.core.robot import RobotClient
from home_robot.mapping.instance import Instance
from home_robot.mapping.voxel import (
    SparseVoxelMap,
    SparseVoxelMapNavigationSpace,
    plan_to_frontier,
)
from home_robot.motion import RRTConnect, Shortcut
from home_robot.perception.encoders import get_encoder


class RobotAgent:
    """Basic demo code. Collects everything that we need to make this work."""

    def __init__(
        self,
        robot: RobotClient,
        semantic_sensor,
        parameters: Dict[str, Any],
    ):
        self.robot = robot
        self.semantic_sensor = semantic_sensor
        self.normalize_embeddings = True
        self.pos_err_threshold = parameters["trajectory_pos_err_threshold"]
        self.rot_err_threshold = parameters["trajectory_rot_err_threshold"]

        self.encoder = get_encoder(parameters["encoder"], parameters["encoder_args"])

        # Wrapper for SparseVoxelMap which connects to ROS
        self.voxel_map = SparseVoxelMap(
            resolution=parameters["voxel_size"],
            local_radius=parameters["local_radius"],
            obs_min_height=parameters["obs_min_height"],
            obs_max_height=parameters["obs_max_height"],
            min_depth=parameters["min_depth"],
            max_depth=parameters["max_depth"],
            obs_min_density=parameters["obs_min_density"],
            smooth_kernel_size=parameters["smooth_kernel_size"],
            encoder=self.encoder,
        )

        # Create planning space
        self.space = SparseVoxelMapNavigationSpace(
            self.voxel_map,
            self.robot.get_robot_model(),
            step_size=parameters["step_size"],
            rotation_step_size=parameters["rotation_step_size"],
            dilate_frontier_size=parameters[
                "dilate_frontier_size"
            ],  # 0.6 meters back from every edge = 12 * 0.02 = 0.24
            dilate_obstacle_size=parameters["dilate_obstacle_size"],
        )

        # Dictionary storing attempts to visit each object
        self._object_attempts = {}

        # Create a simple motion planner
        self.planner = Shortcut(RRTConnect(self.space, self.space.is_valid))

    def update(self, visualize_map=False):
        """Step the data collector. Get a single observation of the world. Remove bad points, such as those from too far or too near the camera. Update the 3d world representation."""
        obs = self.robot.get_observation()

        # Semantic prediction
        obs = self.semantic_sensor.predict(obs)

        # Add observation - helper function will unpack it
        self.voxel_map.add_obs(
            obs,
            K=self.robot.get_camera_intrinsics(),
        )
        if visualize_map:
            # Now draw 2d
            self.voxel_map.get_2d_map(debug=True)

    def move_to_any_instance(self, matches: List[Tuple[int, Instance]]):
        """Check instances and find one we can move to"""

        self.robot.move_to_nav_posture()
        start = self.robot.get_base_pose()
        start_is_valid = self.space.is_valid(start)
        start_is_valid_retries = 5
        while not start_is_valid and start_is_valid_retries > 0:
            print(f"Start {start} is not valid. back up a bit.")
            self.robot.nav.navigate_to([-0.1, 0, 0], relative=True)
            # Get the current position in case we are still invalid
            start = self.robot.get_base_pose()
            start_is_valid = self.space.is_valid(start)
            start_is_valid_retries -= 1
        res = None

        # Just terminate here - motion planning issues apparently!
        if not start_is_valid:
            raise RuntimeError("Invalid start state!")

        # Find and move to one of these
        for i, match in matches:
            print("Checking instance", i)
            # TODO: this is a bad name for this variable
            for j, view in enumerate(match.instance_views):
                print(i, j, view.cam_to_world)
            print("at =", match.bounds)
            res = None
            mask = self.voxel_map.mask_from_bounds(match.bounds)
            for goal in self.space.sample_near_mask(mask, radius_m=0.7):
                goal = goal.cpu().numpy()
                print("       Start:", start)
                print("Sampled Goal:", goal)
                show_goal = np.zeros(3)
                show_goal[:2] = goal[:2]
                goal_is_valid = self.space.is_valid(goal)
                print("Start is valid:", start_is_valid)
                print(" Goal is valid:", goal_is_valid)
                if not goal_is_valid:
                    print(" -> resample goal.")
                    continue

                # plan to the sampled goal
                res = self.planner.plan(start, goal)
                print("Found plan:", res.success)
                if res.success:
                    break
            else:
                # TODO: remove debug code
                # breakpoint()
                print("-> could not plan to instance", i)
                if i not in self._object_attempts:
                    self._object_attempts[i] = 1
                else:
                    self._object_attempts[i] += 1
            if res is not None and res.success:
                break

        if res is not None and res.success:
            # Now move to this location
            print("Full plan to object:")
            for i, pt in enumerate(res.trajectory):
                print("-", i, pt.state)
            self.robot.nav.execute_trajectory(
                [pt.state for pt in res.trajectory],
                pos_err_threshold=self.pos_err_threshold,
                rot_err_threshold=self.rot_err_threshold,
            )
            time.sleep(1.0)
            self.robot.nav.navigate_to([0, 0, np.pi / 2], relative=True)
            self.robot.move_to_manip_posture()
            return True

        return False

    def print_found_classes(self, goal: Optional[str] = None):
        """Helper. print out what we have found according to detic."""
        instances = self.voxel_map.get_instances()
        if goal is not None:
            print(f"Looking for {goal}.")
        print("So far, we have found these classes:")
        for i, instance in enumerate(instances):
            oid = int(instance.category_id.item())
            name = self.semantic_sensor.get_class_name_for_id(oid)
            print(i, name, instance.score)

    def start(self, goal: Optional[str] = None, visualize_map_at_start: bool = False):
        # Tuck the arm away
        print("Sending arm to  home...")
        self.robot.switch_to_manipulation_mode()

        self.robot.move_to_nav_posture()
        self.robot.head.look_close(blocking=False)
        print("... done.")

        # Move the robot into navigation mode
        self.robot.switch_to_navigation_mode()
        self.update(visualize_map=visualize_map_at_start)  # Append latest observations
        self.print_found_classes(goal)
        return self.get_found_instances_by_class(goal)

    def encode_text(self, text: str):
        """Helper function for getting text embeddings"""
        emb = self.encoder.encode_text(text)
        if self.normalize_embeddings:
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb

    def get_found_instances_by_class(
        self, goal: str, threshold: int = 0, debug: bool = False
    ) -> Optional[List[Tuple[int, Instance]]]:
        """Check to see if goal is in our instance memory or not. Return a list of everything with the correct class."""
        matching_instances = []
        instances = self.voxel_map.get_instances()
        for i, instance in enumerate(instances):
            oid = int(instance.category_id.item())
            name = self.semantic_sensor.get_class_name_for_id(oid)
            if name.lower() == goal.lower():
                matching_instances.append((i, instance))
        return self.filter_matches(matching_instances, threshold=threshold)

    def filter_matches(self, matches: List[Tuple[int, Instance]], threshold: int = 1):
        """return only things we have not tried {threshold} times"""
        filtered_matches = []
        for i, instance in matches:
            if i not in self._object_attempts or self._object_attempts[i] < threshold:
                filtered_matches.append((i, instance))
        return filtered_matches

    def go_home(self):
        """Simple helper function to send the robot home safely after a trial."""
        print("Go back to (0, 0, 0) to finish...")
        print("- change posture and switch to navigation mode")
        self.robot.move_to_nav_posture()
        self.robot.head.look_close(blocking=False)
        self.robot.switch_to_navigation_mode()

        print("- try to motion plan there")
        start = self.robot.get_base_pose()
        goal = np.array([0, 0, 0])
        res = self.planner.plan(start, goal)
        # if it fails, skip; else, execute a trajectory to this position
        if res.success:
            print("- executing full plan to home!")
            self.robot.nav.execute_trajectory([pt.state for pt in res.trajectory])
        else:
            print("Can't go home!")

    def choose_best_goal_instance(self, goal: str, debug: bool = False) -> Instance:
        instances = self.voxel_map.get_instances()
        goal_emb = self.encode_text(goal)
        if debug:
            neg1_emb = self.encode_text("the color purple")
            neg2_emb = self.encode_text("a blank white wall")
        best_instance = None
        best_score = -float("Inf")
        for instance in instances:
            if debug:
                print("# views =", len(instance.instance_views))
                print("    cls =", instance.category_id)
            # TODO: remove debug code when not needed for visualization
            # instance._show_point_cloud_open3d()
            img_emb = instance.get_image_embedding(
                aggregation_method="mean", normalize=self.normalize_embeddings
            )
            goal_score = torch.matmul(goal_emb, img_emb).item()
            if debug:
                neg1_score = torch.matmul(neg1_emb, img_emb).item()
                neg2_score = torch.matmul(neg2_emb, img_emb).item()
                print("scores =", goal_score, neg1_score, neg2_score)
            if goal_score > best_score:
                best_instance = instance
                best_score = goal_score
        return best_instance

    def run_exploration(
        self,
        rate: int = 10,
        manual_wait: bool = False,
        explore_iter: int = 3,
        try_to_plan_iter: int = 10,
        dry_run: bool = False,
        random_goals: bool = False,
        visualize: bool = False,
        task_goal: str = None,
        go_home_at_end: bool = False,
        show_goal: bool = False,
    ) -> Optional[Instance]:
        """Go through exploration. We use the voxel_grid map created by our collector to sample free space, and then use our motion planner (RRT for now) to get there. At the end, we plan back to (0,0,0).

        Args:
            visualize(bool): true if we should do intermediate debug visualizations"""
        self.robot.move_to_nav_posture()

        print("Go to (0, 0, 0) to start with...")
        self.robot.nav.navigate_to([0, 0, 0])

        # Explore some number of times
        matches = []
        for i in range(explore_iter):
            print("\n" * 2)
            print("-" * 20, i + 1, "/", explore_iter, "-" * 20)
            self.print_found_classes(task_goal)
            start = self.robot.get_base_pose()
            start_is_valid = self.space.is_valid(start)
            # if start is not valid move backwards a bit
            if not start_is_valid:
                print("Start not valid. back up a bit.")
                self.robot.nav.navigate_to([-0.1, 0, 0], relative=True)
                continue
            print("       Start:", start)
            # sample a goal
            if random_goals:
                goal = next(self.space.sample_random_frontier()).cpu().numpy()
            else:
                res = plan_to_frontier(
                    start,
                    self.planner,
                    self.space,
                    self.voxel_map,
                    try_to_plan_iter=try_to_plan_iter,
                    visualize=visualize,
                )
            if visualize:
                # After doing everything
                self.voxel_map.show(orig=show_goal)

            # if it fails, skip; else, execute a trajectory to this position
            if res.success:
                print("Plan successful!")
                # print("Full plan:")
                # for i, pt in enumerate(res.trajectory):
                #     print("-", i, pt.state)
                if not dry_run:
                    self.robot.nav.execute_trajectory(
                        [pt.state for pt in res.trajectory],
                        pos_err_threshold=self.pos_err_threshold,
                        rot_err_threshold=self.rot_err_threshold,
                    )

            # Append latest observations
            self.update()
            if manual_wait:
                input("... press enter ...")

            if task_goal is not None:
                matches = self.get_found_instances_by_class(task_goal)
                if len(matches) > 0:
                    print("!!! GOAL FOUND! Done exploration. !!!")
                    break

        if go_home_at_end:
            # Finally - plan back to (0,0,0)
            print("Go back to (0, 0, 0) to finish...")
            start = self.robot.get_base_pose()
            goal = np.array([0, 0, 0])
            res = self.planner.plan(start, goal)
            # if it fails, skip; else, execute a trajectory to this position
            if res.success:
                print("Full plan to home:")
                for i, pt in enumerate(res.trajectory):
                    print("-", i, pt.state)
                if not dry_run:
                    self.robot.nav.execute_trajectory(
                        [pt.state for pt in res.trajectory]
                    )
            else:
                print("WARNING: planning to home failed!")

        return matches
