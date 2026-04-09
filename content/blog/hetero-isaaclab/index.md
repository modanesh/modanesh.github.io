---
title: "Heterogeneous Environments in Isaac Lab"
description: A technical guide to training universal robot control policies across multiple morphologies
date: 2026-04-12
author: ["Mohamad H. Danesh"]
showToc: true
disableAnchoredHeadings: false
---

<p style="text-align:center;">
  <img src="project_assets/hetero_isaaclab.gif?raw=true" style="height: 250px; text-align:center;">
</p>


# 🤖 Training Morphology-Agnostic Locomotion Policies with Heterogeneous Robotic Environments in Isaac Lab


## 🌍 Introduction: The Dream of Universal Robot Control {#introduction}

Imagine a world where a single neural network policy can control any legged robot, from Boston Dynamics' Spot to ANYmal to Unitree's quadruped family, without requiring separate training for each platform. This isn't just theoretical elegance; it's a practical necessity. As robotics deployments scale, maintaining separate policies for each robot variant becomes a maintenance nightmare. When you need to update locomotion behavior, you'd have to retrain dozens of specialized policies instead of updating one universal controller.

The vision of **morphology-agnostic policies**, neural networks that generalize across different robot body structures, has captivated the robotics research community. Such policies would learn fundamental locomotion principles that transcend specific hardware, similar to how biological motor control principles apply across species despite vastly different body plans.

However, there's a significant barrier: **most physics simulators and RL frameworks assume homogeneity**. They're architected around the idea that every environment contains an identical robot. Isaac Lab, NVIDIA's cutting-edge robotics simulation framework, is no exception.

---

## 🧱 The Problem: Why Homogeneity Is Baked In {#the-problem}

Isaac Lab's architecture makes several implicit assumptions:

### Core Assumptions

1. **Identical Observation/Action Spaces**: All robots have the same number of joints, producing observations of identical dimensionality.
2. **Unified Scene Management**: The `InteractiveScene` creates N copies of a single robot template.
3. **Homogeneous Reward Computation**: The `RewardManager` expects all terms to return `(num_envs,)` shaped tensors.

### Why These Assumptions Exist

These design choices aren't arbitrary, they enable:
- **Efficient vectorized operations** across all environments
- **Simple memory management** with fixed-size buffers
- **Fast GPU kernels** optimized for uniform data
- **Clean API contracts** between components

But they create fundamental obstacles when training on multiple morphologies (e.g., Spot vs. Unitree Go2) simultaneously.

---

## 🚀 Why This Matters: The Case for Heterogeneous Training {#why-this-matters}

Training on heterogeneous robots isn't academic curiosity, it provides concrete advantages:

### 1. Morphology-Agnostic Feature Learning

When a policy is forced to control different robots with the same weights, it must learn features that capture fundamental locomotion principles rather than robot-specific quirks. The policy can't rely on memorizing specific joint configurations; it must understand abstract concepts like balance, gait stability, and terrain adaptation.


### 2. Efficient Multi-Platform Deployment
Heterogeneous training eliminates the need to maintain separate codebases, training pipelines, and model versions. Training 8 robot types heterogeneously uses the same compute as training 1 robot type, versus 8x the compute for sequential training.

### 3. Better Exploration

Different morphologies explore different regions of the state-action space naturally:
- Lighter robots discover high-speed gaits.
- Heavier robots excel at stable, energy-efficient locomotion.
- Different leg geometries expose different terrain interaction strategies.

Training on all simultaneously enriches the policy's experience space.

### 4. Foundation Model Potential

Heterogeneous training is a stepping stone toward foundation models for robotics, large pretrained policies that can be fine-tuned for specific robots or tasks, similar to how GPT can be adapted to various language tasks.

---

## 🏗️ Architecture Overview {#architecture-overview}

Our solution consists of key components to bridge the heterogeneity gap:

<p style="text-align:center;">
  <img src="project_assets/pipeline.png?raw=true" style="height: 250px; text-align:center;">
</p>

---

## 🛠️ Implementation Deep Dive {#implementation}

### Part 1: Configuration System

We extend Isaac Lab's base classes minimally to track which robots exist in which environments:

```python
@configclass
class HeterogeneousRobotCfg(ArticulationCfg):
    """Configuration for a robot asset in a heterogeneous scene."""
    env_ids: Optional[List[int]] = None

@configclass
class HeterogeneousSensorCfg(ContactSensorCfg):
    """Configuration for a sensor asset in a heterogeneous scene."""
    env_ids: Optional[List[int]] = None
```

**Design principle**: Composition over modification. We don't fork Isaac Lab's classes, we extend them with just the metadata we need.

The scene configuration defines all possible quadrupeds:

```python
@configclass
class HeterogeneousQuadrupedSceneCfg(InteractiveSceneCfg):
    anymal_d: HeterogeneousRobotCfg = ANYMAL_D_CFG.replace(...)
    anymal_c: HeterogeneousRobotCfg = ANYMAL_C_CFG.replace(...)
    anymal_b: HeterogeneousRobotCfg = ANYMAL_B_CFG.replace(...)
    unitree_a1: HeterogeneousRobotCfg = UNITREE_A1_CFG.replace(...)
    spot: HeterogeneousRobotCfg = SPOT_CFG.replace(...)
    # ... 8 total robots defined ...
```

**Key insight**: Define all robots in the config, but only instantiate selected ones at runtime.

---

### Part 2: Dynamic Distribution Logic

The critical challenge is **timing**. We need to configure the scene distribution **before** the parent `DirectRLEnv` initialization creates the physics scene.

```python
class HeterogeneousQuadrupedVelocityEnv(DirectRLEnv):
    def __init__(self, cfg, render_mode=None, **kwargs):
        # Define all possible robots
        self.all_quadrupeds = ["anymal_d", "anymal_c", "anymal_b", 
                               "unitree_a1", "unitree_go1", "unitree_go2", 
                               "unitree_b2", "spot"]
        
        # Get robot selection from kwargs (or default to all)
        self.quadrupeds_list = kwargs.get("quadrupeds", self.all_quadrupeds)
        
        # Setup BEFORE super().__init__()
        self._setup_robot_distribution(cfg, self.quadrupeds_list)
        self._filter_rewards(cfg, self.quadrupeds_list)
        
        # Call parent init which creates scene
        super().__init__(cfg, render_mode, **kwargs)
        
        # Store references in dictionaries for clean iteration
        self.robots: Dict[str, Articulation] = {}
        self.robot_env_ids: Dict[str, torch.Tensor] = {}
        for robot_name in self.quadrupeds_list:
            self.robots[robot_name] = self.scene[robot_name]
            self.robot_env_ids[robot_name] = torch.tensor(
                getattr(self.cfg.scene, robot_name).env_ids, device=self.device
            )
```

The `_filter_rewards` method is crucial: it dynamically iterates through `cfg.rewards` and deletes attributes (reward terms) associated with robots that aren't active in the current run, preventing runtime crashes.

**Why dictionaries?** They scale naturally. Adding a new robot requires no code changes, just add it to the config.

The distribution logic evenly assigns environments to robots:

```python
def _setup_robot_distribution(self, cfg, quadrupeds_to_use):
    num_envs = cfg.scene.num_envs
    num_robots = len(quadrupeds_to_use)
    envs_per_robot = num_envs // num_robots
    start_idx = 0

    for i, robot_name in enumerate(quadrupeds_to_use):
        # Handle non-divisible num_envs
        num_robot_envs = envs_per_robot + (1 if i < num_envs % num_robots else 0)
        env_ids = list(range(start_idx, start_idx + num_robot_envs))
        
        # Configure robot and sensor
        getattr(cfg.scene, robot_name).env_ids = env_ids
        getattr(cfg.scene, robot_name).prim_path = [
            f"/World/envs/env_{j}/{robot_name}" for j in env_ids
        ]
        getattr(cfg.scene, f"{robot_name}_contacts").env_ids = env_ids
        
        start_idx += num_robot_envs

    # Remove unused robots from config
    for robot_name in self.all_quadrupeds:
        if robot_name not in quadrupeds_to_use:
            delattr(cfg.scene, robot_name)
            delattr(cfg.scene, f"{robot_name}_contacts")
```

**Example distribution** with `--num_envs=100` and 3 robots:
- ANYmal-D: environments 0-33 (34 envs)
- Unitree A1: environments 34-66 (33 envs)  
- Spot: environments 67-99 (33 envs)

---

### Part 3: The Joint Order Unification Problem 🔄

A specific challenge often overlooked is that robots define joints differently. 
*   **ANYmal**: LF, LH, RF, RH (Leg Order) / HAA, HFE, KFE (Joint Types)
*   **Spot**: FL, FR, HL, HR (Leg Order) / hx, hy, kn (Joint Types)
*   **Unitree**: FL, FR, RL, RR (Leg Order) / hip, thigh, calf (Joint Types)

If you feed the neural network's output vector directly to the robots, "Action 0" might move different joints on each robot.

**The Solution:** We enforce an **"ANYmal Joint-Major"** format for the Policy (LF, LH, RF, RH) and map everything else to it.

```python
# In __init__
self.action_indices = {}
self.obs_indices = {}

# Target Policy Sequence (Virtual)
virtual_joint_names = ["LF_HAA", "LH_HAA", "RF_HAA", "RH_HAA", "LF_HFE", ...]


for robot_name, robot in self.robots.items():
    physical_joint_names = robot.data.joint_names
    
    # Define semantic maps per robot type
    if "spot" in robot_name:
        leg_map = {"LF": "fl", "LH": "hl", "RF": "fr", "RH": "hr"}
        joint_map = {"HAA": "hx", "HFE": "hy", "KFE": "kn"}
    elif "unitree" in robot_name:
        leg_map = {"LF": "FL", "LH": "RL", "RF": "FR", "RH": "RR"}
        joint_map = {"HAA": "hip", "HFE": "thigh", "KFE": "calf"}
    # ... anymal maps ...

    # Generate Index Tensors
    to_robot_indices = [] # Policy -> Robot
    to_policy_indices = [] # Robot -> Policy

    # Logic to match virtual names to physical indices...
    
    self.action_indices[robot_name] = torch.tensor(to_robot_indices, device=self.device, dtype=torch.long)
    self.obs_indices[robot_name] = torch.tensor(to_policy_indices, device=self.device, dtype=torch.long)
```

During the physics step, we use `torch.index_select` (or advanced indexing) to reorder the policy actions to match the physical robot's expectations.

---

### Part 4: Index Mapping System 🗺️

The most subtle challenge: Isaac Lab components use two different index spaces:

| Index Type | Range | Used By |
| --- | --- | --- |
| **Global Environment IDs** | 0 to `num_envs-1` | Observations, rewards, terminations, buffers |
| **Robot-Local Indices** | 0 to N-1 (N = instances of that robot) | Articulation views, actuators, robot state |

**The problem**: If Unitree occupies environments 34-66:

* Global environment 50 contains Unitree.
* But Unitree's local index is 16 (50 - 34 = 16).

**The solution**: Use `torch.searchsorted()` for efficient global→local conversion:

```python
def _reset_idx(self, env_ids: torch.Tensor):
    for robot_name, robot in self.robots.items():
        robot_env_ids_all = self.robot_env_ids[robot_name]
        
        # Find which resets apply to this robot
        mask = torch.isin(env_ids, robot_env_ids_all)
        robot_global_ids = env_ids[mask]
        
        if len(robot_global_ids) > 0:
            # Convert: global [50,51,52] → local [16,17,18]
            local_indices = torch.searchsorted(robot_env_ids_all, robot_global_ids)
            
            # Use local indices for robot operations
            robot.write_root_state_to_sim(..., env_ids=local_indices)
            for actuator in robot.actuators.values():
                actuator.reset(local_indices)
```

**Critical**: Always use local indices for robot/actuator methods. Using global IDs causes index-out-of-bounds errors.

---

### Part 5: Padded Reward Functions 🧮

The `RewardManager` expects all terms to return `(num_envs,)` tensors. Robot-specific computations naturally produce `(N,)` tensors where N < num_envs.

**The pattern**: Create a full-sized tensor, compute for applicable envs, and fill via indexing:

```python
def get_robot_env_ids(env, cfg: SceneEntityCfg):
    """Helper to extract robot env_ids from asset/sensor config."""
    robot_name = cfg.name.replace("_contacts", "")
    return env.robot_env_ids[robot_name]

def track_lin_vel_xy_exp(env, std, command_name, asset_cfg):
    """Reward for tracking linear velocity commands."""
    asset = env.scene[asset_cfg.name]
    robot_env_ids = get_robot_env_ids(env, asset_cfg)
    
    # Step 1: Create full-sized tensor with zeros
    reward = torch.zeros(env.num_envs, device=env.device)
    
    # Step 2: Compute reward for this robot's environments only
    commands = env._commands[robot_env_ids]
    lin_vel_error = torch.sum(
        torch.square(commands[:, :2] - asset.data.root_lin_vel_b[:, :2]),
        dim=1
    )
    
    # Step 3: Fill only applicable rows
    reward[robot_env_ids] = torch.exp(-lin_vel_error / std**2)
    
    return reward  # Shape: (num_envs,) ✓
```

**Result**: ANYmal rewards strictly affect its subset of environments, Unitree rewards affect its subset, etc. The reward manager sums them without dimension errors.

---

### Part 6: Scalable Implementation Patterns 📈

**Use dictionaries over named attributes:**

```python
# ❌ Doesn't scale
self.anymal = self.scene["anymal_d"]
self.unitree = self.scene["unitree_a1"]

# ✓ Scales to N robots
self.robots = {name: self.scene[name] for name in self.quadrupeds_list}
```

**Configuration-driven behavior:**

```python
# ❌ Hardcoded
if robot_name == "anymal_d": action_scale = 0.5
elif robot_name == "unitree_a1": action_scale = 0.25

# ✓ Config-driven
action_scale = getattr(self.cfg, f"action_scale_{robot_name}")
```

**Generic iteration:**

```python
def _apply_action(self):
    for robot_name, robot in self.robots.items():
        env_ids = self.robot_env_ids[robot_name]
        robot.set_joint_position_target(self.processed_actions[env_ids])
```

Adding a new robot requires:

1. Adding config in the scene definition.
2. Adding an action scale parameter.
3. Adding reward terms.
4. **No code changes** to environment logic.

---

### Part 7: Handling Extreme Morphological Quirks (Spot & Unitree B2) 🐕

When scaling up to 8 diverse robots (especially incorporating Spot and Unitree B2), applying uniform randomization rules breaks down quickly. We built two mechanisms to handle these extremes seamlessly:

1. **Flexible Reset Randomizations**: While ANYmal and most Unitree robots handle multiplicative joint position scaling well (e.g., `0.8x` to `1.2x`), Spot's unique kinematics explode under heavy scaling. We implemented a config-driven `reset_joint_pos_mode` parameter to dynamically toggle between `scale` (multiplicative) and `offset` (additive noise, e.g., `(-0.05, 0.05)`) per robot at reset time.
2. **Dynamic Body Parsing**: Applying morphology randomization (like modifying base mass or CoM) requires isolating the correct rigid bodies. Because naming conventions differ wildly—ANYmal uses `.*_THIGH`, Unitree uses `[FR][LR]_thigh`, and Spot uses `[fh][lr]_uleg`—we use a dedicated `_get_robot_body_groups()` mapper. This allows us to parse links via regex dynamically and apply structural domain randomization accurately without accidentally randomizing a camera mount or head link!

---

## 🎲 Domain Randomization for Robust Policies {#domain-randomization}

A critical component for sim-to-real transfer is **comprehensive domain randomization**. We handle this manually to ensure correct index mapping between the heterogeneous robots.

### Morphology Randomization

We explicitly randomize Mass, Center of Mass (CoM), and Friction at startup using 64 distributed buckets.

*Technical Note:* Isaac Lab's DirectRL implementation interacts with PhysX views. Operations like `set_masses` or `set_material_properties` often require indices to be on the **CPU**, even if the simulation runs on GPU.

```python
def _apply_startup_randomization(self):
    for robot_name, robot in self.robots.items():
        # Indices must be CPU int32 for PhysX calls
        local_env_ids = torch.arange(num_robot_envs, device="cpu", dtype=torch.int32)
    
        # Friction Randomization (Bucketed)
        # Create material properties tensor on CPU...
        robot.root_physx_view.set_material_properties(materials, indices=local_env_ids)
                
        # Mass Randomization
        current_masses = robot.root_physx_view.get_masses().clone()
        # Apply offsets...
        robot.root_physx_view.set_masses(current_masses, indices=local_env_ids)
```

### Interval Randomization

We also implement periodic "pushes" (velocity randomization) in `_pre_physics_step` to force the policy to recover from sudden, unpredicted external disturbances.

---

## ⚠️ Common Pitfalls and Solutions {#pitfalls}

### Pitfall 1: Dimension Mismatch in Observations

**Symptom**: Concatenating robot data (N instances) with global commands (`num_envs` instances) throws a shape error.
**Solution**: Always index global tensors with `robot_env_ids` before concatenation.

```python
# ✓ Right: index commands first
obs = torch.cat([
    robot.data.root_lin_vel_b,           # (8, 3)
    self._commands[robot_env_ids]        # (8, 3) ← match!
], dim=-1)
```

### Pitfall 2: Reward Config Bloat

**Symptom**: `KeyError` or crashes because the code tries to compute rewards for a robot that wasn't spawned during that specific training run.
**Solution**: Use the `_filter_rewards` method to strictly delete config attributes for unused robots before the parent class processes them.

### Pitfall 3: Joint Order Confusion
**Symptom**: The robot flails wildly or crosses legs despite low tracking error.
**Solution**: Verify the `action_indices` mapping. "Front Left" on a Unitree robot corresponds to a specific index in its generic articulation vector; this must perfectly align with the unified Policy's expectation of "Front Left."

---

## 📊 Training Results and Insights {#results}

# TODO

---

## Getting Started {#getting-started}

### Installation

Installation is quite similar to the IsaacLab's default installation, cloning this repo instead of the original one.

```bash
# Clone Isaac Lab
git clone https://github.com/modanesh/Hetero-IsaacLab.git
cd Hetero-IsaacLab

# Install dependencies (and training modules, eg rsl_rl)
./isaaclab.sh --install rsl_rl
```

### Basic Training

To train on a specific subset of robots, you can pass the list with the `--quadrupeds` flag, from the list of available quadrupeds:
```bash
anymal_d,anymal_c,anymal_b,unitree_a1,unitree_go1,unitree_go2,unitree_b2,spot

```

For instance:
```bash
# Train on all 8 robots with 4096 environments
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task=Isaac-Velocity-Flat-HeteroQuadruped-v0 \
    --num_envs=4096 \
    --quadrupeds anymal_d,anymal_c,anymal_b,unitree_a1,unitree_go1,unitree_go2,unitree_b2,spot

# Results will be in logs/rsl_rl/heterogeneous_quad_flat/
```

---

## Conclusion

Building heterogeneous multi-agent environments in Isaac Lab requires careful attention to **configuration design**, **timing**, **index management**, and **joint unification**. But the payoffs are substantial:

✅ **Compute savings** (1 training run vs N sequential runs)
✅ **Superior transfer learning** to new morphologies  
✅ **Simplified deployment** with a single universal policy

The architecture presented here is production-ready and scales naturally. It represents a crucial step toward morphology-agnostic controllers that embody fundamental locomotion principles transcending specific hardware.

---

## Acknowledgments

The implementation builds on Isaac Lab's robust simulation infrastructure and RSL-RL's efficient PPO implementation.

### Code Availability

Complete implementation available at: **https://github.com/modanesh/Hetero-IsaacLab.git**

We welcome contributions, bug reports, and extensions. Areas of particular interest:

* Additional robot morphologies
* Novel reward functions
* Transfer learning experiments
* Real-world deployment results

### Citation

In case you find this project useful for your research, please cite the following:
```bibtex
@software{heteroisaac,
  author = {Danesh, Mohamad H.},
  title     = {Hetero-Isaac: Heterogeneous Quadrupedal Simulation built atop Isaac Lab},
  year      = {2026},
  url =  {https://github.com/modanesh/Hetero-IsaacLab},
  license = {BSD-3-Clause}
}
```

```bibtex
@misc{danesh2026heterogeneous,
  title = {Training Morphology-Agnostic Locomotion Policies with Heterogeneous Robotic Environments in {Isaac Lab}},
  author = {Danesh, Mohamad H.},
  year = {2026},
  howpublished = {Technical Blog Post},
  url = {https://modanesh.github.io/blog/hetero-isaaclab},
}
```

