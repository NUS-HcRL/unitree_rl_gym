from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class Pm01FallCfg(LeggedRobotCfg):
    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.9]
        default_joint_angles = {
            "j00_hip_pitch_l": -0.06,
            "j01_hip_roll_l": 0.0,
            "j02_hip_yaw_l": 0.0,
            "j03_knee_pitch_l": 0.12,
            "j04_ankle_pitch_l": -0.06,
            "j05_ankle_roll_l": 0.0,
            "j06_hip_pitch_r": -0.06,
            "j07_hip_roll_r": 0.0,
            "j08_hip_yaw_r": 0.0,
            "j09_knee_pitch_r": 0.12,
            "j10_ankle_pitch_r": -0.06,
            "j11_ankle_roll_r": 0.0,
            "j12_waist_yaw": 0.0,
            "j13_shoulder_pitch_l": 0.0,
            "j14_shoulder_roll_l": 0.15,
            "j15_shoulder_yaw_l": 0.0,
            "j16_elbow_pitch_l": -0.25,
            "j17_elbow_yaw_l": 0.0,
            "j18_shoulder_pitch_r": 0.0,
            "j19_shoulder_roll_r": -0.15,
            "j20_shoulder_yaw_r": 0.0,
            "j21_elbow_pitch_r": -0.25,
            "j22_elbow_yaw_r": 0.0,
            "j23_head_yaw": 0.0,
        }
    
    class env(LeggedRobotCfg.env):
        # 3 (base_lin_vel) + 3 (base_ang_vel) + 3 (projected_gravity) + 3 (commands) + 24 (dof_pos) + 24 (dof_vel) + 22 (actions) = 82
        # Note: actions exclude j12_waist_yaw and j23_head_yaw (locked joints)
        num_observations = 82
        num_privileged_obs = 82  # Same as observations for now
        num_actions = 22  # 22 controllable joints (excluding j12_waist_yaw and j23_head_yaw)
        env_spacing = 1.5
        episode_length_s = 25
        send_timeouts = True

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'plane'
        static_friction = 0.6
        dynamic_friction = 0.6
        num_rows = 20
        max_init_terrain_level = 10
        terrain_proportions = [0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1]

    class control(LeggedRobotCfg.control):
        control_type = 'P'
        stiffness = {
            "hip_pitch": 110,
            "hip_roll": 70,
            "hip_yaw": 70,
            "knee_pitch": 110,
            "ankle_pitch": 30,
            "ankle_roll": 30,
            "waist_yaw": 80,
            "shoulder_pitch": 30,
            "shoulder_roll": 30,
            "shoulder_yaw": 30,
            "elbow_pitch": 30,
            "elbow_yaw": 30,
            "head_yaw": 30,  # Add head_yaw stiffness
        }
        damping = {
            "hip_pitch": 5.0,
            "hip_roll": 3.0,
            "hip_yaw": 3.0,
            "knee_pitch": 5.0,
            "ankle_pitch": 0.3,
            "ankle_roll": 0.3,
            "waist_yaw": 5.0,
            "shoulder_pitch": 0.3,
            "shoulder_roll": 0.3,
            "shoulder_yaw": 0.3,
            "elbow_pitch": 0.3,
            "elbow_yaw": 0.3,
            "head_yaw": 0.3,  # Add head_yaw damping
        }
        action_scale = 0.25
        decimation = 10

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/pm01/urdf/serial_pm_v2_primitive.urdf'
        name = "pm01"
        foot_name = "link_ankle_roll"
        penalize_contacts_on = [
            "link_head_yaw",
            "link_elbow_end_l",
            "link_elbow_end_r"
        ]
        terminate_after_contacts_on = [
            "link_head_yaw",
            "link_elbow_end_l",
            "link_elbow_end_r"
        ]
        self_collisions = 0
        collapse_fixed_joints = False  # Set to False to keep elbow_end and elbow_yaw separate
        flip_visual_attachments = False
        replace_cylinder_with_capsule = False

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.2, 1.3]
        randomize_base_mass = False
        added_mass_range = [-4.0, 4.0]
        push_robots = False
        push_interval_s = 8
        max_push_vel_xy = 0.4

    class commands(LeggedRobotCfg.commands):
        curriculum = False
        max_curriculum = 1.0
        resampling_time = 8.0
        heading_command = False
        class ranges:
            lin_vel_x = [0.0, 0.0]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]
            heading = [0.0, 0.0]

    class normalization(LeggedRobotCfg.normalization):
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 1.0
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
        clip_observations = 100.
        clip_actions = 100.

    class rewards(LeggedRobotCfg.rewards):
        only_positive_rewards = False
        base_height_target = 0.8132
        max_contact_force = 500.0
        tracking_sigma = 5
        soft_dof_pos_limit = 1.0
        soft_dof_vel_limit = 1.0
        soft_torque_limit = 1.0
        # Termination contact threshold for fall protection
        termination_contact_threshold = 40000.0
        catastrophic_penalty_threshold = 1000.0
        
        class scales:
            # Fall protection rewards
            root_lin_vel = -0.0
            critical_contacts = -0.0
            all_contacts = -0.0
            actuation_impulse = -0.0
            dof_pos_limits = -0.0
            torque_saturation = -0.0
            torques = -0.0
            action_rate = -0.0
            dof_acc = -0.0
            contact_sequence = 1.0e+0
            catastrophic_contact = -1.0e+0
            
            # Disabled rewards for fall task
            termination = 0.0
            tracking_lin_vel = 0.0
            tracking_ang_vel = 0.0
            lin_vel_z = 0.0
            ang_vel_xy = 0.0
            orientation = 0.0
            base_height = 0.0
            feet_air_time = 0.0
            collision = 0.0
            feet_stumble = 0.0
            stand_still = 0.0

    class sim(LeggedRobotCfg.sim):
        dt = 0.001

class Pm01FallCfgPPO(LeggedRobotCfgPPO):
    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu'
    
    class algorithm(LeggedRobotCfgPPO.algorithm):
        pass
    
    class runner(LeggedRobotCfgPPO.runner):
        max_iterations = 15000
        run_name = ''
        experiment_name = 'pm01_fall'

