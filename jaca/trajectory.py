import rerun as rr
import numpy as np
import time
import uuid


class Trajectory:
    def __init__(self, name):
        self.name = name

    def log_trajectory(
        self,
        trajectory: np.ndarray,
        trajectory_id: str = None,
        colors: np.ndarray = None,
        observation_time: float = None,
    ):
        """
        Log trajectory

        Args:
            trajectory (np.ndarray): [n, 2] or [n, 3]
            observation_time (float, optional): Time since epoch in seconds. If None current time used.
        """
        if observation_time is None:
            observation_time = time.time()
        rr.set_time_seconds("real_clock", observation_time)
        if trajectory_id is None:
            trajectory_id = str(uuid.uuid4())
        if trajectory.shape[1] == 2:
            trajectory = np.hstack((trajectory, np.zeros((trajectory.shape[0], 1))))
        colors = colors or self._get_random_rgb()
        rr.log(f"/trajectory/{trajectory_id}", rr.Clear(recursive=True))
        for i in range(trajectory.shape[0] - 1):
            rr.log(
                f"/trajectory/{trajectory_id}/{i}",
                rr.Arrows3D(
                    origins=trajectory[i],
                    vectors=trajectory[i + 1] - trajectory[i],
                    colors=colors,
                ),
            )
