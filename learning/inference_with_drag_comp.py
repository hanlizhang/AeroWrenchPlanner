"""
inference with drag compensation in controller
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import matplotlib.pyplot as plt
import numpy as np

import sys

import ruamel.yaml as yaml
from flax.training import train_state
import optax
import jax
from mlp_jax import MLP
from model_learning import restore_checkpoint
from scipy.spatial.transform import Rotation as R
from rotorpy.utils.occupancy_map import OccupancyMap
from rotorpy.controllers.quadrotor_control import SE3Control
from rotorpy.vehicles.multirotor import Multirotor
from AeroWrenchPlanner.rotorpy.trajectories.minsnap_nn_jit import MinSnap
from rotorpy.vehicles.crazyflie_params import quad_params
from rotorpy.environments import Environment
from rotorpy.world import World

gamma = 1

PI = np.pi


def sample_waypoints(
    num_waypoints,
    world,
    world_buffer=2,
    check_collision=True,
    min_distance=1,
    max_distance=3,
    max_attempts=1000,
    start_waypoint=None,
    end_waypoint=None,
    rng=None,
    seed=None,
):
    """
    Samples random waypoints (x,y,z) in the world. Ensures waypoints do not collide with objects, although there is no guarantee that
    the path you generate with these waypoints will be collision free.
    Inputs:
        num_waypoints: Number of waypoints to sample.
        world: Instance of World class containing the map extents and any obstacles.
        world_buffer: Buffer around the world used for sampling. This is used to ensure that waypoints are at least this distance away
            from the edge of the world.
        check_collision: If True, checks for collisions with obstacles. If False, does not check for collisions. Checking collisions slows down the script.
        min_distance: Minimum distance between waypoints consecutive waypoints.
        max_distance: Maximum distance between consecutive waypoints.
        max_attempts: Maximum number of attempts to sample a waypoint.
        start_waypoint: If specified, the first waypoint will be this point.
        end_waypoint: If specified, the last waypoint will be this point.
    Outputs:
        waypoints: A list of (x,y,z) waypoints. [[waypoint_1], [waypoint_2], ... , [waypoint_n]]
    """

    if min_distance > max_distance:
        raise Exception("min_distance must be less than or equal to max_distance.")

    def check_distance(waypoint, waypoints, min_distance, max_distance):
        """
        Checks if the waypoint is at least min_distance away from all other waypoints.
        Inputs:
            waypoint: The waypoint to check.
            waypoints: A list of waypoints.
            min_distance: The minimum distance the waypoint must be from all other waypoints.
            max_distance: The maximum distance the waypoint can be from all other waypoints.
        Outputs:
            collision: True if the waypoint is at least min_distance away from all other waypoints. False otherwise.
        """
        collision = False
        for w in waypoints:
            if (np.linalg.norm(waypoint - w) < min_distance) or (
                np.linalg.norm(waypoint - w) > max_distance
            ):
                collision = True
        return collision

    def check_obstacles(waypoint, occupancy_map):
        """
        Checks if the waypoint is colliding with any obstacles in the world.
        Inputs:
            waypoint: The waypoint to check.
            occupancy_map: An instance of the occupancy map.
        Outputs:
            collision: True if the waypoint is colliding with any obstacles in the world. False otherwise.
        """
        collision = False
        if occupancy_map.is_occupied_metric(waypoint):
            collision = True
        return collision

    def single_sample(
        world,
        current_waypoints,
        world_buffer,
        occupancy_map,
        min_distance,
        max_distance,
        max_attempts=1000,
        rng=None,
        seed=None,
    ):
        """
        Samples a single waypoint.
        Inputs:
            world: Instance of World class containing the map extents and any obstacles.
            world_buffer: Buffer around the world used for sampling. This is used to ensure that waypoints are at least this distance away
                from the edge of the world.
            occupancy_map: An instance of the occupancy map.
            min_distance: Minimum distance between waypoints consecutive waypoints.
            max_distance: Maximum distance between consecutive waypoints.
            max_attempts: Maximum number of attempts to sample a waypoint.
            rng: Random number generator. If None, uses numpy's random number generator.
            seed: Seed for the random number generator.
        Outputs:
            waypoint: A single (x,y,z) waypoint.
        """

        if seed is not None:
            np.random.seed(seed)

        num_attempts = 0

        world_lower_limits = (
            np.array(world.world["bounds"]["extents"][0::2]) + world_buffer
        )
        world_upper_limits = (
            np.array(world.world["bounds"]["extents"][1::2]) - world_buffer
        )

        if len(current_waypoints) == 0:
            max_distance_lower_limits = world_lower_limits
            max_distance_upper_limits = world_upper_limits
        else:
            max_distance_lower_limits = current_waypoints[-1] - max_distance
            max_distance_upper_limits = current_waypoints[-1] + max_distance

        lower_limits = np.max(
            np.vstack((world_lower_limits, max_distance_lower_limits)), axis=0
        )
        upper_limits = np.min(
            np.vstack((world_upper_limits, max_distance_upper_limits)), axis=0
        )

        waypoint = np.random.uniform(low=lower_limits, high=upper_limits, size=(3,))
        while check_obstacles(waypoint, occupancy_map) or (
            check_distance(waypoint, current_waypoints, min_distance, max_distance)
            if occupancy_map is not None
            else False
        ):
            waypoint = np.random.uniform(low=lower_limits, high=upper_limits, size=(3,))
            num_attempts += 1
            if num_attempts > max_attempts:
                raise Exception(
                    "Could not sample a waypoint after {} attempts. Issue with obstacles: {}, Issue with min/max distance: {}".format(
                        max_attempts,
                        check_obstacles(waypoint, occupancy_map),
                        check_distance(
                            waypoint, current_waypoints, min_distance, max_distance
                        ),
                    )
                )
        return waypoint

    ######################################################################################################################

    waypoints = []

    if check_collision:
        # Create occupancy map from the world. This can potentially be slow, so only do it if the user wants to check for collisions.
        occupancy_map = OccupancyMap(
            world=world, resolution=[0.5, 0.5, 0.5], margin=0.1
        )
    else:
        occupancy_map = None

    if start_waypoint is not None:
        waypoints = [start_waypoint]
    else:
        # Randomly sample a start waypoint.
        waypoints.append(
            single_sample(
                world,
                waypoints,
                world_buffer,
                occupancy_map,
                min_distance,
                max_distance,
                max_attempts,
                rng,
                seed,
            )
        )

    num_waypoints -= 1

    if end_waypoint is not None:
        num_waypoints -= 1

    for _ in range(num_waypoints):
        waypoints.append(
            single_sample(
                world,
                waypoints,
                world_buffer,
                occupancy_map,
                min_distance,
                max_distance,
                max_attempts,
                rng,
                seed,
            )
        )

    if end_waypoint is not None:
        waypoints.append(end_waypoint)

    return np.array(waypoints)


def sample_yaw(seed, waypoints, yaw_min=-np.pi, yaw_max=np.pi):
    """
    Samples random yaw angles for the waypoints.
    """
    np.random.seed(seed)
    yaw_angles = np.random.uniform(low=yaw_min, high=yaw_max, size=len(waypoints))

    return yaw_angles


def compute_cost(sim_result, robust_c=1.0):
    """
    Computes the cost from the output of a simulator instance.
    Inputs:
        sim_result: The output of a simulator instance.
    Outputs:
        cost: The cost of the trajectory.
    """

    # Some useful values from the trajectory.
    actual_pos = sim_result["state"]["x"]  # Position
    des_pos = sim_result["flat"]["x"]  # Desired position

    # Cost components by cumulative sum of squared norms
    position_error = np.linalg.norm(actual_pos - des_pos, axis=1) ** 2

    return np.sum(position_error)


# Function to run simulation and compute cost
def run_simulation_and_compute_cost(
    waypoints,
    yaw_angles,
    vavg,
    use_neural_network,
    regularizer=None,
    vehicle=None,
    controller=None,
):
    traj = MinSnap(
        points=waypoints,
        yaw_angles=yaw_angles,
        v_avg=vavg,
        use_neural_network=use_neural_network,
        regularizer=regularizer,
    )
    nan_encountered = traj.nan_encountered  # Flag indicating if NaN was encountered

    sim_instance = Environment(
        vehicle=vehicle,
        controller=controller,
        trajectory=traj,
        wind_profile=None,
        sim_rate=100,
    )

    # Set initial state
    x0 = {
        "x": waypoints[0],
        "v": np.zeros(
            3,
        ),
        "q": np.array([0, 0, 0, 1]),  # quaternion
        "w": np.zeros(
            3,
        ),
        "wind": np.array([0, 0, 0]),
        "rotor_speeds": np.array([1788.53, 1788.53, 1788.53, 1788.53]),
    }
    sim_instance.vehicle.initial_state = x0

    waypoint_times = traj.t_keyframes
    sim_result = sim_instance.run(
        t_final=traj.t_keyframes[-1], use_mocap=False, terminate=False, plot=False
    )
    trajectory_cost = compute_cost(sim_result)

    return sim_result, trajectory_cost, waypoint_times, nan_encountered


def plot_results_for_drag_compensation(
    sim_result_init, waypoints, initial_cost, filename=None, waypoints_time=None
):
    # Compute yaw angles from quaternions
    def compute_yaw_from_quaternion(quaternions):
        R_matrices = R.from_quat(quaternions).as_matrix()
        b3 = R_matrices[:, :, 2]
        H = np.zeros((len(quaternions), 3, 3))
        for i in range(len(quaternions)):
            H[i, :, :] = np.array(
                [
                    [
                        1 - (b3[i, 0] ** 2) / (1 + b3[i, 2]),
                        -(b3[i, 0] * b3[i, 1]) / (1 + b3[i, 2]),
                        b3[i, 0],
                    ],
                    [
                        -(b3[i, 0] * b3[i, 1]) / (1 + b3[i, 2]),
                        1 - (b3[i, 1] ** 2) / (1 + b3[i, 2]),
                        b3[i, 1],
                    ],
                    [-b3[i, 0], -b3[i, 1], b3[i, 2]],
                ]
            )
        Hyaw = np.transpose(H, axes=(0, 2, 1)) @ R_matrices
        actual_yaw = np.arctan2(Hyaw[:, 1, 0], Hyaw[:, 0, 0])
        return actual_yaw

    actual_yaw_init = compute_yaw_from_quaternion(sim_result_init["state"]["q"])

    # Create the figure
    fig = plt.figure(figsize=(18, 8))

    # 3D Trajectory plot with waypoints
    ax_traj = fig.add_subplot(121, projection="3d")
    ax_traj.plot3D(
        sim_result_init["state"]["x"][:, 0],
        sim_result_init["state"]["x"][:, 1],
        sim_result_init["state"]["x"][:, 2],
        "b--",
    )
    ax_traj.plot3D(
        sim_result_init["flat"]["x"][:, 0],
        sim_result_init["flat"]["x"][:, 1],
        sim_result_init["flat"]["x"][:, 2],
        "r-.",
    )
    ax_traj.scatter(
        waypoints[:, 0], waypoints[:, 1], waypoints[:, 2], c="k", marker="o"
    )
    ax_traj.set_title("3D Trajectory with Drag Compensation", fontsize=18)
    ax_traj.set_xlabel("X", fontsize=16)
    ax_traj.set_ylabel("Y", fontsize=16)
    ax_traj.set_zlabel("Z", fontsize=16)
    ax_traj.legend(["Actual", "Reference", "Waypoints"], fontsize=14)

    # Display initial cost
    cost_text = f"Initial Cost with Drag Compensation: {initial_cost:.2f}"
    ax_traj.text2D(0.03, 0.01, cost_text, transform=ax_traj.transAxes, fontsize=16)

    # Subplots for X, Y, Z, Yaw
    gs = fig.add_gridspec(4, 2)
    ax_x = fig.add_subplot(gs[0, 1])
    ax_y = fig.add_subplot(gs[1, 1])
    ax_z = fig.add_subplot(gs[2, 1])
    ax_yaw = fig.add_subplot(gs[3, 1])

    # Subplot for X
    ax_x.plot(sim_result_init["time"], sim_result_init["state"]["x"][:, 0], "b--")
    ax_x.plot(sim_result_init["time"], sim_result_init["flat"]["x"][:, 0], "r-.")
    ax_x.set_title("X Position Over Time", fontsize=18)
    ax_x.set_xlabel("Time", fontsize=16)
    ax_x.set_ylabel("X Position", fontsize=16)

    # Subplot for Y
    ax_y.plot(sim_result_init["time"], sim_result_init["state"]["x"][:, 1], "b--")
    ax_y.plot(sim_result_init["time"], sim_result_init["flat"]["x"][:, 1], "r-.")
    ax_y.set_title("Y Position Over Time", fontsize=18)
    ax_y.set_xlabel("Time", fontsize=16)
    ax_y.set_ylabel("Y Position", fontsize=16)

    # Subplot for Z
    ax_z.plot(sim_result_init["time"], sim_result_init["state"]["x"][:, 2], "b--")
    ax_z.plot(sim_result_init["time"], sim_result_init["flat"]["x"][:, 2], "r-.")
    ax_z.set_title("Z Position Over Time", fontsize=18)
    ax_z.set_xlabel("Time", fontsize=16)
    ax_z.set_ylabel("Z Position", fontsize=16)

    # Adding keyframes to the subplots
    for ax, dim in zip([ax_x, ax_y, ax_z], [0, 1, 2]):
        ax.scatter(
            waypoints_time, waypoints[:, dim], c="k", marker="o", label="Waypoints"
        )

    # Subplot for Yaw
    ax_yaw.plot(sim_result_init["time"], actual_yaw_init, "b--")
    ax_yaw.plot(sim_result_init["time"], sim_result_init["flat"]["yaw"], "r-.")
    ax_yaw.set_title("Yaw Angle Over Time", fontsize=18)
    ax_yaw.set_xlabel("Time", fontsize=16)
    ax_yaw.set_ylabel("Yaw Angle", fontsize=16)
    ax_yaw.scatter(
        waypoints_time, np.zeros(len(waypoints)), c="k", marker="o", label="Waypoints"
    )

    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename)

    # Close the figure
    plt.close(fig)


def main():
    # Initialize neural network
    rho = 0.1
    input_size = 96  # number of coeff

    with open(r"/workspace/rotorpy/learning/params.yaml") as f:
        yaml_data = yaml.load(f, Loader=yaml.RoundTripLoader)

    num_hidden = yaml_data["num_hidden"]
    learning_rate = yaml_data["learning_rate"]

    # Load the trained model
    model = MLP(num_hidden=num_hidden, num_outputs=1)
    # Printing the model shows its attributes
    print(model)

    rng = jax.random.PRNGKey(427)
    rng, inp_rng, init_rng = jax.random.split(rng, 3)
    inp = jax.random.normal(inp_rng, (1, input_size))  # Batch size 32, input size 2012
    # Initialize the model
    params = model.init(init_rng, inp)

    optimizer = optax.sgd(learning_rate=learning_rate, momentum=0.9)

    model_state = train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=optimizer
    )

    model_save = yaml_data["save_path"] + str(rho)
    print("model_save", model_save)

    trained_model_state = restore_checkpoint(model_state, model_save, 7)

    # Define the quadrotor parameters
    world_size = 10
    num_waypoints = 4
    vavg = 1
    random_yaw = False
    yaw_min = -0.85 * np.pi
    yaw_max = 0.85 * np.pi

    world_buffer = 2
    min_distance = 1
    max_distance = min_distance + 3
    start_waypoint = None  # If you want to start at a specific waypoint, specify it using [xstart, ystart, zstart]
    end_waypoint = None  # If you want to end at a specific waypoint, specify it using [xend, yend, zend]

    # Now create the world, vehicle, and controller objects.
    world = World.empty(
        [
            -world_size / 2,
            world_size / 2,
            -world_size / 2,
            world_size / 2,
            -world_size / 2,
            world_size / 2,
        ]
    )
    vehicle = Multirotor(quad_params)
    controller_with_drag_compensation = SE3Control(quad_params, drag_compensation=True)

    # Loop for 100 trajectories
    for i in range(100):
        # Sample waypoints
        waypoints = sample_waypoints(
            num_waypoints=num_waypoints,
            world=world,
            world_buffer=world_buffer,
            min_distance=min_distance,
            max_distance=max_distance,
            start_waypoint=start_waypoint,
            end_waypoint=end_waypoint,
            rng=None,
            seed=i,
        )

        # Sample yaw angles
        yaw_angles = sample_yaw(
            seed=i, waypoints=waypoints, yaw_min=yaw_min, yaw_max=yaw_max
        )

        yaw_angles_zero = np.zeros(len(waypoints))

        # /workspace/rotorpy/rotorpy/sim_figures/
        figure_path = "/workspace/data_output/sim_figures_drag_compensation"

        #################### drag compensation ####################
        # run simulation and compute cost for the initial trajectory with drag compensation
        (
            sim_result_init_drag_compensation,
            trajectory_cost_init_drag_compensation,
            waypoints_time,
            _,
        ) = run_simulation_and_compute_cost(
            waypoints,
            yaw_angles_zero,
            vavg,
            use_neural_network=False,
            regularizer=None,
            vehicle=vehicle,
            controller=controller_with_drag_compensation,
        )
        print(
            f"Trajectory {i} initial cost with drag compensation: {trajectory_cost_init_drag_compensation}"
        )
        plot_results_for_drag_compensation(
            sim_result_init_drag_compensation,
            waypoints,
            trajectory_cost_init_drag_compensation,
            filename=figure_path + f"/trajectory_{i}_drag_compensation.png",
            waypoints_time=waypoints_time,
        )


if __name__ == "__main__":
    # try:
    #     main()
    # except rospy.ROSInterruptException:
    #     pass
    main()
