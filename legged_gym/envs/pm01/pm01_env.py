from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym import LEGGED_GYM_ROOT_DIR
from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
import torch
import numpy as np
import os

class Pm01Robot(LeggedRobot):
    
    def _create_envs(self):
        """Override to save body_names"""
        # Call parent to create environments
        super()._create_envs()
        # Save body_names for fall protection
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)
        
        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity
        
        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        
        # Initialize fall protection indices
        self._init_fall_protection_indices()
    
    def _init_buffers(self):
        # Call super first - it will initialize most buffers
        # But it will fail when trying to set p_gains[i] for i >= num_actions
        # We'll catch this and fix it
        try:
            super()._init_buffers()
        except IndexError as e:
            # Base class failed because p_gains/d_gains are too small
            # Resize them to num_dof and re-set PD gains
            if hasattr(self, 'num_dof') and hasattr(self, 'dof_names'):
                # Resize p_gains and d_gains to num_dof
                self.p_gains = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
                self.d_gains = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
                
                # Re-set PD gains from config for all dofs
                for i, name in enumerate(self.dof_names):
                    found = False
                    for dof_name in self.cfg.control.stiffness.keys():
                        if dof_name in name:
                            self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                            self.d_gains[i] = self.cfg.control.damping[dof_name]
                            found = True
                            break
                    if not found:
                        self.p_gains[i] = 0.0
                        self.d_gains[i] = 0.0
                        if self.cfg.control.control_type in ["P", "V"]:
                            print(f"PD gain of joint {name} were not defined, setting them to zero")
                
                # Ensure default_dof_pos is properly set
                if not hasattr(self, 'default_dof_pos') or len(self.default_dof_pos.shape) == 1:
                    self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
                    for i, name in enumerate(self.dof_names):
                        if name in self.cfg.init_state.default_joint_angles:
                            self.default_dof_pos[i] = self.cfg.init_state.default_joint_angles[name]
                    self.default_dof_pos = self.default_dof_pos.unsqueeze(0)
            else:
                raise e
        
        # Ensure p_gains and d_gains are the correct size (num_dof)
        if hasattr(self, 'num_dof') and hasattr(self, 'dof_names'):
            if self.p_gains.shape[0] < self.num_dof:
                # Resize to num_dof
                self.p_gains = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
                self.d_gains = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
                
                # Re-set PD gains from config
                for i, name in enumerate(self.dof_names):
                    found = False
                    for dof_name in self.cfg.control.stiffness.keys():
                        if dof_name in name:
                            self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                            self.d_gains[i] = self.cfg.control.damping[dof_name]
                            found = True
                            break
                    if not found:
                        self.p_gains[i] = 0.0
                        self.d_gains[i] = 0.0
        
        # Initialize contact state for fall protection
        self.contact_state = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.critical_contact_hit = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.catastrophic_contact_hit = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        # Find indices for locked joints (waist_yaw and head_yaw) - these are excluded from actions
        if hasattr(self, 'dof_names') and self.dof_names is not None:
            self.locked_joint_indices = []
            self.action_to_dof_mapping = []  # Maps action index to dof index
            for i, name in enumerate(self.dof_names):
                if 'waist_yaw' in name or 'head_yaw' in name:
                    self.locked_joint_indices.append(i)
                    # Set very high PD gains for locked joints to keep them fixed at default position
                    self.p_gains[i] = 10000.0  # Very high stiffness
                    self.d_gains[i] = 1000.0   # Very high damping
                else:
                    self.action_to_dof_mapping.append(i)
            
            self.locked_joint_indices = torch.tensor(self.locked_joint_indices, dtype=torch.long, device=self.device)
            self.action_to_dof_mapping = torch.tensor(self.action_to_dof_mapping, dtype=torch.long, device=self.device)
            
            # Resize torques to num_dof (base class initializes it as num_actions)
            if self.torques.shape[1] < self.num_dof:
                self.torques = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        
        # Initialize tracking variables for rewards
        self.critical_cf_norm_max_per_iteration = None
        self.critical_cf_norm_min_per_iteration = None
        self.last_iteration_step_critical = -1
        self.all_contacts_cf_norm_max_per_iteration = 0.0
        self.last_iteration_step_all_contacts = -1
        self.contact_seq_transition_0_to_1_envs = []
        self.contact_seq_transition_1_to_2_envs = []
        self.contact_seq_transition_0_to_1_info = {}
        self.contact_seq_transition_1_to_2_info = {}
        self.last_iteration_step_contact_seq = -1
        
        # Joint indices cache
        self._knee_dof_indices = None
        self._hip_pitch_dof_indices = None
        
        # Initialize curriculum learning parameters
        if hasattr(self.cfg.domain_rand, 'curriculum') and hasattr(self.cfg.domain_rand.curriculum, 'max_curriculum_step'):
            self._max_curriculum_step = self.cfg.domain_rand.curriculum.max_curriculum_step
        else:
            # Default: 15000 iterations * 4096 envs * 24 steps = 1,474,560,000 steps
            max_iterations = 15000
            num_steps_per_env = 24
            self._max_curriculum_step = max_iterations * self.num_envs * num_steps_per_env
        
        # Initialize dof_name_to_idx for curriculum learning (lazy initialization)
        self._dof_name_to_idx = None
    
    def compute_training_progress(self):
        """Compute training progress, return value between 0.0 and 1.0"""
        p = self.common_step_counter / self._max_curriculum_step
        return torch.clamp(torch.tensor(p, device=self.device), 0.0, 1.0)
    
    def _init_fall_protection_indices(self):
        """Initialize body and dof indices for fall protection"""
        if not hasattr(self, 'body_names'):
            # Fallback: initialize empty tensors
            self.knee_indices = torch.tensor([], dtype=torch.long, device=self.device)
            self.hip_pitch_indices = torch.tensor([], dtype=torch.long, device=self.device)
            self.shoulder_yaw_indices = torch.tensor([], dtype=torch.long, device=self.device)
            self.elbow_pitch_indices = torch.tensor([], dtype=torch.long, device=self.device)
            return
        
        # Find knee indices
        self.knee_indices = torch.tensor([
            i for i, name in enumerate(self.body_names) 
            if 'knee_pitch' in name.lower()
        ], dtype=torch.long, device=self.device)
        
        # Find hip_pitch indices
        self.hip_pitch_indices = torch.tensor([
            i for i, name in enumerate(self.body_names) 
            if 'hip_pitch' in name.lower() and 'link' in name.lower()
        ], dtype=torch.long, device=self.device)
        
        # Find shoulder_yaw indices
        self.shoulder_yaw_indices = torch.tensor([
            i for i, name in enumerate(self.body_names) 
            if 'shoulder_yaw' in name.lower()
        ], dtype=torch.long, device=self.device)
        
        # Find elbow_pitch indices
        self.elbow_pitch_indices = torch.tensor([
            i for i, name in enumerate(self.body_names) 
            if 'elbow_pitch' in name.lower() or 'elbow_yaw' in name.lower()
        ], dtype=torch.long, device=self.device)
    
    @property
    def knee_dof_indices(self):
        """Get knee dof indices"""
        if self._knee_dof_indices is None:
            if hasattr(self, 'dof_names') and self.dof_names is not None:
                self._knee_dof_indices = torch.tensor([
                    i for i, name in enumerate(self.dof_names) 
                    if ("knee_pitch" in name) and ("j03" in name or "j09" in name)
                ], dtype=torch.long, device=self.device)
            else:
                self._knee_dof_indices = torch.tensor([], dtype=torch.long, device=self.device)
        return self._knee_dof_indices
    
    @property
    def hip_pitch_dof_indices(self):
        """Get hip pitch dof indices"""
        if self._hip_pitch_dof_indices is None:
            if hasattr(self, 'dof_names') and self.dof_names is not None:
                self._hip_pitch_dof_indices = torch.tensor([
                    i for i, name in enumerate(self.dof_names) 
                    if ("hip_pitch" in name) and ("j00" in name or "j06" in name)
                ], dtype=torch.long, device=self.device)
            else:
                self._hip_pitch_dof_indices = torch.tensor([], dtype=torch.long, device=self.device)
        return self._hip_pitch_dof_indices
    
    def reset_idx(self, env_ids):
        """Reset environments and contact state"""
        # Apply curriculum learning to DOF positions before calling super().reset_idx()
        if (hasattr(self.cfg.domain_rand, 'curriculum') and 
            hasattr(self.cfg.domain_rand.curriculum, 'enable_curriculum') and
            self.cfg.domain_rand.curriculum.enable_curriculum):
            self._reset_dofs_with_curriculum(env_ids)
        else:
            # Use default reset
            self._reset_dofs(env_ids)
        
        # Call parent reset_idx for other resets
        # We need to manually call the parts we need since we already reset DOFs
        if len(env_ids) == 0:
            return
        
        self._reset_root_states(env_ids)
        self._resample_commands(env_ids)
        
        # reset buffers
        self.actions[env_ids] = 0.
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        
        # Reset contact state
        self.contact_state[env_ids] = 0
        self.critical_contact_hit[env_ids] = False
        self.catastrophic_contact_hit[env_ids] = False
        
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        
        # Add termination statistics
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf
            # Calculate termination statistics for reset environments
            time_out_count = torch.sum(self.time_out_buf[env_ids]).item()
            termination_count = len(env_ids) - time_out_count
            self.extras["episode"]["termination_rate"] = termination_count / len(env_ids) if len(env_ids) > 0 else 0.0
            self.extras["episode"]["timeout_rate"] = time_out_count / len(env_ids) if len(env_ids) > 0 else 0.0
            # Store termination info for each reset environment
            self.extras["episode"]["termination_info"] = {
                "timeout": time_out_count,
                "termination": termination_count,
                "total": len(env_ids)
            }
    
    def _reset_dofs_with_curriculum(self, env_ids):
        """Reset DOF positions with curriculum learning (joint position randomization)"""
        if len(env_ids) == 0:
            return
        
        # Ensure env_ids are within valid range
        env_ids = env_ids[env_ids < self.num_envs]
        if len(env_ids) == 0:
            return
        
        # Ensure dof_name_to_idx is initialized
        if self._dof_name_to_idx is None:
            self._dof_name_to_idx = {}
            for i, name in enumerate(self.dof_names):
                self._dof_name_to_idx[name] = i
        
        p = self.compute_training_progress().item()  # Get scalar value
        
        # 1) Modify DoF initial positions: 0.8 ~ 1.2 * q0
        if self.default_dof_pos.dim() == 2 and self.default_dof_pos.shape[0] == 1:
            q0 = self.default_dof_pos[0].unsqueeze(0).expand(len(env_ids), -1)  # (n, dofs)
        elif self.default_dof_pos.dim() == 1:
            q0 = self.default_dof_pos.unsqueeze(0).expand(len(env_ids), -1)  # (n, dofs)
        else:
            q0 = self.default_dof_pos[env_ids]  # (n, dofs)
        scale = torch.rand(len(env_ids), self.num_dof, device=self.device) * 0.4 + 0.8  # U(0.8, 1.2)
        q_init = q0 * scale
        
        # 2) Additional randomization for waist/shoulder/elbow based on training progress p
        # Find joint indices
        idx_waist_pitch = None
        idx_waist_roll = None
        idx_waist_yaw = None
        idx_sh_pitch_l = None
        idx_sh_roll_l = None
        idx_sh_pitch_r = None
        idx_sh_roll_r = None
        idx_elbow_l = None
        idx_elbow_r = None
        
        # Find joint indices based on joint names (using dictionary like original code)
        for name, idx in self._dof_name_to_idx.items():
            name_lower = name.lower()
            if "waist_pitch" in name_lower:
                idx_waist_pitch = idx
            elif "waist_roll" in name_lower:
                idx_waist_roll = idx
            elif "waist_yaw" in name_lower or "j12" in name:
                idx_waist_yaw = idx
            elif "shoulder_pitch_l" in name_lower or "j13" in name:
                idx_sh_pitch_l = idx
            elif "shoulder_roll_l" in name_lower or "j14" in name:
                idx_sh_roll_l = idx
            elif "shoulder_pitch_r" in name_lower or "j18" in name:
                idx_sh_pitch_r = idx
            elif "shoulder_roll_r" in name_lower or "j19" in name:
                idx_sh_roll_r = idx
            elif "elbow_pitch_l" in name_lower or "j16" in name:
                # elbow_pitch_l
                if idx_elbow_l is None:
                    idx_elbow_l = idx
            elif "elbow_yaw_l" in name_lower or "j17" in name:
                # elbow_yaw_l，如果还没有设置则使用这个
                if idx_elbow_l is None:
                    idx_elbow_l = idx
            elif "elbow_pitch_r" in name_lower or "j21" in name:
                # elbow_pitch_r
                if idx_elbow_r is None:
                    idx_elbow_r = idx
            elif "elbow_yaw_r" in name_lower or "j22" in name:
                # elbow_yaw_r，如果还没有设置则使用这个
                if idx_elbow_r is None:
                    idx_elbow_r = idx
        
        # Apply curriculum-based randomization for waist/shoulder/elbow (according to table 2)
        # Ensure indices are within valid range
        # waist_pitch: U(-0.5, 0.5)
        if idx_waist_pitch is not None and 0 <= idx_waist_pitch < self.num_dof:
            q_init[:, idx_waist_pitch] = torch.empty(len(env_ids), device=self.device).uniform_(-0.5, 0.5)
        
        # waist_roll: U(-0.5, 0.5)
        if idx_waist_roll is not None and 0 <= idx_waist_roll < self.num_dof:
            q_init[:, idx_waist_roll] = torch.empty(len(env_ids), device=self.device).uniform_(-0.5, 0.5)
        
        # waist_yaw: U(-(p+0.2), p+0.2)
        if idx_waist_yaw is not None and 0 <= idx_waist_yaw < self.num_dof:
            q_init[:, idx_waist_yaw] = torch.empty(len(env_ids), device=self.device).uniform_(-(p + 0.2), p + 0.2)
        
        # shoulder_pitch_l: U(-(p+0.4), p+0.4)
        if idx_sh_pitch_l is not None and 0 <= idx_sh_pitch_l < self.num_dof:
            q_init[:, idx_sh_pitch_l] = torch.empty(len(env_ids), device=self.device).uniform_(-(p + 0.4), p + 0.4)
        
        # shoulder_roll_l: U(0.15, 1.3*p + 0.15)
        if idx_sh_roll_l is not None and 0 <= idx_sh_roll_l < self.num_dof:
            q_init[:, idx_sh_roll_l] = torch.empty(len(env_ids), device=self.device).uniform_(0.15, 1.3 * p + 0.15)
        
        # shoulder_pitch_r: U(-(p+0.4), p+0.4)
        if idx_sh_pitch_r is not None and 0 <= idx_sh_pitch_r < self.num_dof:
            q_init[:, idx_sh_pitch_r] = torch.empty(len(env_ids), device=self.device).uniform_(-(p + 0.4), p + 0.4)
        
        # shoulder_roll_r: U(-(1.3*p + 0.15), -0.15)
        if idx_sh_roll_r is not None and 0 <= idx_sh_roll_r < self.num_dof:
            q_init[:, idx_sh_roll_r] = torch.empty(len(env_ids), device=self.device).uniform_(-(1.3 * p + 0.15), -0.15)
        
        # elbow_l: U(-0.1-0.3*p, 0.1+0.6*p)
        if idx_elbow_l is not None and 0 <= idx_elbow_l < self.num_dof:
            q_init[:, idx_elbow_l] = torch.empty(len(env_ids), device=self.device).uniform_(-0.1 - 0.3 * p, 0.1 + 0.6 * p)
        
        # elbow_r: U(-0.1-0.3*p, 0.1+0.6*p)
        if idx_elbow_r is not None and 0 <= idx_elbow_r < self.num_dof:
            q_init[:, idx_elbow_r] = torch.empty(len(env_ids), device=self.device).uniform_(-0.1 - 0.3 * p, 0.1 + 0.6 * p)
        
        # Update DOF positions
        self.dof_pos[env_ids] = q_init
        
        # Update DOF state in simulation
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32)
        )
        
        # 3) Modify initial root linear velocity (curriculum learning)
        # vx, vy ~ U(-0.1-0.25*p, 0.1+0.25*p)
        if not self.cfg.asset.fix_base_link:
            # Calculate initial velocity
            vx = torch.empty(len(env_ids), device=self.device).uniform_(-0.1 - 0.25 * p, 0.1 + 0.25 * p)
            vy = torch.empty(len(env_ids), device=self.device).uniform_(-0.1 - 0.25 * p, 0.1 + 0.25 * p)
            
            # Set initial velocity
            self.root_states[env_ids, 7] = vx  # lin_vel_x
            self.root_states[env_ids, 8] = vy  # lin_vel_y
            
            # Update root state in simulation
            self.gym.set_actor_root_state_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(self.root_states),
                gymtorch.unwrap_tensor(env_ids_int32),
                len(env_ids_int32)
            )
    
    def check_termination(self):
        """Check if environments need to be reset"""
        # Check contact force termination using configured threshold
        if hasattr(self, 'termination_contact_indices') and self.termination_contact_indices.numel() > 0:
            termination_threshold = getattr(self.cfg.rewards, 'termination_contact_threshold', 40000.0)
            self.reset_buf = torch.any(
                torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > termination_threshold,
                dim=1
            )
        else:
            self.reset_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        # Orientation termination removed - only contact force and timeout will terminate
        
        # Check timeout
        self.time_out_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= self.time_out_buf
    
    def _compute_torques(self, actions):
        """Override to map 22-dim actions to 24-dim torques (excluding waist_yaw and head_yaw)"""
        # actions shape: (num_envs, 22)
        # We need to map to (num_envs, 24) for torques
        
        # Create full action tensor with zeros for locked joints
        full_actions = torch.zeros(self.num_envs, self.num_dof, device=self.device, dtype=actions.dtype)
        if hasattr(self, 'action_to_dof_mapping') and len(self.action_to_dof_mapping) > 0:
            full_actions[:, self.action_to_dof_mapping] = actions
        
        # Compute torques using parent method with full actions
        torques = super()._compute_torques(full_actions)
        
        # Lock waist_yaw and head_yaw joints by setting their torques to zero
        if hasattr(self, 'locked_joint_indices') and len(self.locked_joint_indices) > 0:
            torques[:, self.locked_joint_indices] = 0.0
        
        return torques
    
    def compute_observations(self):
        """Compute observations"""
        # Note: actions are 22-dim, but dof_pos and dof_vel are 24-dim (all joints)
        self.obs_buf = torch.cat((
            self.base_lin_vel * self.obs_scales.lin_vel,
            self.base_ang_vel * self.obs_scales.ang_vel,
            self.projected_gravity,
            self.commands[:, :3] * self.commands_scale,
            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,  # 24 dims
            self.dof_vel * self.obs_scales.dof_vel,  # 24 dims
            self.actions  # 22 dims (excluding waist_yaw and head_yaw)
        ), dim=-1)
        
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = self.obs_buf.clone()
        
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec
    
    # ============= Fall Protection Reward Functions =============
    
    def _reward_root_lin_vel(self):
        """Penalize root linear velocity"""
        return torch.sum(self.base_lin_vel ** 2, dim=1)
    
    def _reward_contact_sequence(self):
        """Contact sequence reward for fall protection"""
        eps = 1.0
        
        def _hit(indices):
            if indices is None or indices.numel() == 0:
                return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            cf_norm = torch.norm(self.contact_forces[:, indices, :], dim=-1)
            return (cf_norm > eps).any(dim=1)
        
        knee_hit = _hit(self.knee_indices)
        hip_pitch_hit = _hit(self.hip_pitch_indices)
        first_contact_hit = knee_hit | hip_pitch_hit
        shoulder_hit = _hit(self.shoulder_yaw_indices)
        elbow_hit = _hit(self.elbow_pitch_indices)
        
        # Check knee/hip flexion (relative to default angles)
        # Use training progress p computed by compute_training_progress()
        p = self.compute_training_progress().item()  # Get scalar value
        
        knee_flex_threshold = 1.3 * min(1.0, p / 0.2)  # Additional flexion beyond default
        knee_flex_ok = True
        if len(self.knee_dof_indices) > 0:
            knee_angles = self.dof_pos[:, self.knee_dof_indices]
            # Get default knee angles for the corresponding DOFs
            default_knee_angles = self.default_dof_pos[0, self.knee_dof_indices]  # (num_knee_dofs,)
            # Check if flexion relative to default exceeds threshold
            knee_flex_ok = ((knee_angles - default_knee_angles.unsqueeze(0)) > knee_flex_threshold).any(dim=1)
        
        hip_flex_threshold = -1.0 * min(1.0, p / 0.2)  # Additional flexion beyond default (negative for hip)
        hip_flex_ok = True
        if len(self.hip_pitch_dof_indices) > 0:
            hip_angles = self.dof_pos[:, self.hip_pitch_dof_indices]
            # Get default hip angles for the corresponding DOFs
            default_hip_angles = self.default_dof_pos[0, self.hip_pitch_dof_indices]  # (num_hip_dofs,)
            # Check if flexion relative to default exceeds threshold (hip flexion is negative)
            hip_flex_ok = ((hip_angles - default_hip_angles.unsqueeze(0)) < hip_flex_threshold).any(dim=1)
        
        flex_ok = (knee_hit & knee_flex_ok) | (hip_pitch_hit & hip_flex_ok)
        
        state = self.contact_state
        transition_0_to_1 = (state == 0) & first_contact_hit & flex_ok & ~shoulder_hit & ~elbow_hit
        transition_1_to_2 = (state == 1) & (shoulder_hit | elbow_hit)
        
        new_state = state.clone()
        new_state[transition_0_to_1] = 1
        new_state[transition_1_to_2] = 2
        self.contact_state = new_state
        
        R_knee_or_hip_first = 300.0
        R_shoulder_after_knee = 150.0
        reward = torch.zeros(self.num_envs, device=self.device)
        reward += R_knee_or_hip_first * transition_0_to_1.float()
        reward += R_shoulder_after_knee * transition_1_to_2.float()
        
        return reward / self.dt
    
    def _reward_critical_contacts(self):
        """Critical contact penalty"""
        if not hasattr(self, 'penalised_contact_indices') or self.penalised_contact_indices.numel() == 0:
            self.critical_contact_hit = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            return torch.zeros(self.num_envs, device=self.device)
        
        cf = self.contact_forces[:, self.penalised_contact_indices, :]
        cf_norm = torch.norm(cf, dim=-1)
        eps = 100.0
        critical_hit = (cf_norm > eps).any(dim=1)
        self.critical_contact_hit = critical_hit
        
        R_cat = 3000.0
        return R_cat * critical_hit.float()
    
    def _reward_catastrophic_contact(self):
        """Catastrophic contact penalty"""
        if not hasattr(self, 'termination_contact_indices') or self.termination_contact_indices.numel() == 0:
            self.catastrophic_contact_hit[:] = False
            return torch.zeros(self.num_envs, device=self.device)
        
        cf = self.contact_forces[:, self.termination_contact_indices, :]
        cf_norm = torch.norm(cf, dim=-1)
        penalty_threshold = getattr(self.cfg.rewards, 'catastrophic_penalty_threshold', 1000.0)
        catastrophic = (cf_norm > penalty_threshold).any(dim=1)
        self.catastrophic_contact_hit = catastrophic
        
        R_cat = 400.0
        return (R_cat * catastrophic.float()) / self.dt
    
    def _reward_all_contacts(self):
        """Penalize all contact forces"""
        cf = self.contact_forces
        cf_norm = torch.norm(cf, dim=-1)
        
        # Create default weights
        w = torch.ones(self.num_bodies, device=self.device, dtype=cf.dtype)
        if hasattr(self, 'body_names'):
            for i, body_name in enumerate(self.body_names):
                if 'ankle_roll' in body_name.lower():
                    w[i] = 0.1
                elif 'knee_pitch' in body_name.lower():
                    w[i] = 0.3
                elif 'shoulder_yaw' in body_name.lower():
                    w[i] = 0.7
        
        cf_norm = torch.clamp(cf_norm, max=50000.0)
        weighted_norm = cf_norm * w.unsqueeze(0)
        return (weighted_norm ** 2).sum(dim=1)
    
    def _reward_actuation_impulse(self):
        """Penalize actuation impulse"""
        q = self.dof_pos
        qd = self.dof_vel
        q_min = self.dof_pos_limits[:, 0]
        q_max = self.dof_pos_limits[:, 1]
        
        q_range = q_max - q_min
        q_l = q_min + 0.05 * q_range
        q_u = q_max - 0.05 * q_range
        
        lower_mask = q <= q_l
        upper_mask = q >= q_u
        
        impulse_lower = torch.where(
            lower_mask,
            torch.clamp(qd, max=0.0) ** 2,
            torch.zeros_like(qd),
        )
        impulse_upper = torch.where(
            upper_mask,
            torch.clamp(qd, min=0.0) ** 2,
            torch.zeros_like(qd),
        )
        
        return (impulse_lower + impulse_upper).sum(dim=1)
    
    def _reward_dof_pos_limits(self):
        """Penalize out-of-limit dof positions"""
        q = self.dof_pos
        q_min = self.dof_pos_limits[:, 0]
        q_max = self.dof_pos_limits[:, 1]
        
        below = torch.clamp(q_min - q, min=0.0)
        above = torch.clamp(q - q_max, min=0.0)
        
        return (below + above).sum(dim=1)
    
    def _reward_torque_saturation(self):
        """Penalize torque saturation"""
        tau = self.torques
        tau_u = self.torque_limits
        ratio = torch.abs(tau) / (tau_u.unsqueeze(0) + 1e-8)
        over = torch.clamp(ratio - 0.95, min=0.0)
        return over.sum(dim=1)
    
    def _reward_torques(self):
        """Penalize large torques"""
        return torch.sum(self.torques ** 2, dim=1)
    
    def _reward_action_rate(self):
        """Penalize action rate"""
        diff = self.actions - self.last_actions
        return torch.norm(diff, dim=1)
    
    def _reward_dof_acc(self):
        """Penalize dof acceleration"""
        diff = self.dof_vel - self.last_dof_vel
        return torch.sum(diff ** 2, dim=1)

