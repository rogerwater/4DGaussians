import abc
import numpy as np
from mpc.utils import (
    ObservationList,
    write_moviepy_gif,
    generate_text_square,
)


class Optimizer(metaclass=abc.ABCMeta):
    def log_best_plans(self, filename, vis_preds, goal, scores, fps=5, include_score=False):
        """Log the best trajectory plans as a gif visualization.
        
        Args:
            filename: Path to save the visualization
            vis_preds: List of predicted observation trajectories
            goal: Goal observation to reach
            scores: Reward scores for each trajectory
            fps: Frames per second for the output gif
            include_score: Whether to include score text in the visualization (default: False)
        """
        # Make sure that the goal has the same time dimension as the predictions
        pred_len = len(vis_preds[0])
        goal_len = len(goal)
        
        if goal_len == 1:
            goal = goal.repeat(pred_len)
        elif goal_len != pred_len:
            # 截断或填充goal以匹配prediction长度
            if goal_len > pred_len:
                # 截断goal
                goal = ObservationList(
                    {k: v[:pred_len] for k, v in goal.data_dict.items()},
                    image_shape=goal.image_shape
                )
            else:
                # 重复最后一帧填充
                goal = goal.repeat(pred_len // goal_len + 1)
                goal = ObservationList(
                    {k: v[:pred_len] for k, v in goal.data_dict.items()},
                    image_shape=goal.image_shape
                )

        # Concatenate predictions with goal along time axis
        vis_preds = [o.append(goal, axis=1) for o in vis_preds]
        vis_preds = ObservationList.from_observations_list(vis_preds, axis=2)

        vis_preds_image = vis_preds.to_image_list()
        
        if include_score:
            # Only add score overlay if explicitly requested
            img_height = vis_preds_image.shape[1]
            img_width_per_plan = vis_preds_image.shape[2] // len(scores)
            
            # Create score text images, one for each plan
            score_image = np.concatenate(
                [
                    generate_text_square(
                        str(np.round(score, decimals=4).item()),
                        size=(img_width_per_plan, img_height)
                    )
                    for score in scores
                ],
                axis=-1,
            )
            # Move channel dimension to last: [3, H, W_total] -> [H, W_total, 3]
            score_image = np.moveaxis(score_image, 0, -1)
            # Tile for all timesteps: [H, W_total, 3] -> [T, H, W_total, 3]
            score_image = np.tile(score_image[None], (len(vis_preds), 1, 1, 1))
            # Concatenate score images below predictions
            vis_preds_image = np.concatenate((vis_preds_image, score_image), axis=1)
        
        # Write the gif
        write_moviepy_gif(list(vis_preds_image), filename, fps=fps)
