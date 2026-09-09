# Isaac Lab 初学者指南

> 版本：v2.3.2 | 基于 NVIDIA Isaac Sim 5.1 | Python 3.11

---

## 一、项目简介

Isaac Lab 是一个基于 NVIDIA Isaac Sim 的 **GPU 加速机器人仿真框架**，主要用于：
- **强化学习 (RL)** — 训练机器人控制策略
- **模仿学习 (IL)** — 从演示数据学习行为
- **运动规划** — 机械臂路径规划
- **Sim-to-Real** — 仿真到真实世界的迁移

---

## 二、顶层目录结构

```
IsaacLab/
├── apps/              # Isaac Sim 应用配置文件（.kit）
├── docker/            # Docker 容器化部署
├── docs/              # Sphinx 文档源码
├── logs/              # 运行时日志输出
├── outputs/           # 仿真输出（视频、数据等）
├── scripts/           # 可执行脚本（教程、训练、工具）
├── source/            # 核心 Python 包（6 个扩展包）
├── tools/             # 项目工具（测试、模板、CI）
├── isaaclab.sh        # 统一 CLI 入口（安装、测试、格式化等）
└── pyproject.toml     # 顶层工具配置
```

---

## 三、核心源码 `source/`

这是项目的核心，包含 6 个独立的 Python 包：

### 3.1 `source/isaaclab/` — 核心框架

提供所有基础抽象和管理器：

| 子目录 | 功能 |
|--------|------|
| `app/` | 应用启动器，初始化 Isaac Sim |
| `actuators/` | 执行器模型（理想、PD 控制、神经网络） |
| `assets/` | 物理资产抽象（机器人、刚体、柔体） |
| `controllers/` | 机器人控制器（IK、OSC、阻抗控制） |
| `devices/` | 输入设备（键盘、手柄、SpaceMouse、VR） |
| `envs/` | **环境抽象**（MDP 组件、动作、观测、奖励） |
| `managers/` | **Manager 架构核心**（动作、观测、奖励管理器） |
| `scene/` | 场景管理 |
| `sensors/` | 传感器（相机、接触、IMU、射线投射） |
| `sim/` | 仿真工具（USD 生成器、格式转换器） |
| `terrains/` | 地形生成（粗糙地形、高度场） |
| `utils/` | 通用工具（数学、噪声、缓冲区） |

**核心概念：Asset → Scene → Env → Manager**

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│    Asset    │ →  │    Scene    │ →  │     Env     │ →  │   Manager   │
│  (机器人等) │    │   (场景)    │    │   (环境)    │    │  (MDP管理)  │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

### 3.2 `source/isaaclab_tasks/` — 任务/环境定义

包含所有注册到 Gymnasium 的环境，分两种范式：

#### Manager-based 环境（推荐，模块化可组合）
```
manager_based/
├── classic/           # 经典控制（Cartpole、Ant、Humanoid）
├── locomotion/        # 移动任务
│   └── velocity/      # 速度跟踪（支持 11 种机器人）
│       └── config/    # 每种机器人的配置
│           ├── anymal_c/    # ANYmal C 四足
│           ├── go2/         # Unitree Go2
│           ├── spot/        # Boston Dynamics Spot
│           └── ...
├── manipulation/      # 操作任务
│   ├── lift/          # 抓取抬起
│   ├── reach/         # 末端执行器到达
│   ├── pick_place/    # 抓取放置
│   ├── stack/         # 堆叠
│   └── ...
├── navigation/        # 导航任务
└── locomanipulation/  # 移动+操作组合
```

#### Direct 环境（自包含，单文件）
```
direct/
├── allegro_hand/      # Allegro 灵巧手
├── humanoid_amp/      # 人形 AMP
├── shadow_hand/       # Shadow 灵巧手
├── cartpole/          # Cartpole（含相机变体）
└── ...
```

### 3.3 `source/isaaclab_rl/` — RL 框架封装

连接 Isaac Lab 环境与主流 RL 训练框架：

| 文件 | 对应框架 |
|------|----------|
| `rsl_rl/` | RSL-RL（ETH Zurich） |
| `rl_games/` | RL Games（NVIDIA） |
| `sb3.py` | Stable Baselines3 |
| `skrl.py` | SKRL |
| `ray/` | Ray RLlib |

### 3.4 `source/isaaclab_assets/` — 预配置资产

提供 26+ 种机器人的开箱即用配置：

```
robots/
├── franka.py          # Franka Emika Panda 机械臂
├── unitree.py         # Unitree A1/Go1/Go2/G1/H1
├── anymal.py          # ANYmal B/C/D 四足
├── spot.py            # Boston Dynamics Spot
├── shadow_hand.py     # Shadow 灵巧手
├── humanoid.py        # 人形机器人
└── ...
```

### 3.5 `source/isaaclab_mimic/` — MimicGen 数据生成

自动化演示数据生成：
- `datagen/` — 数据生成管道
- `motion_planners/` — 运动规划（cuRobo）
- `envs/` — Mimic 专用环境配置

### 3.6 `source/isaaclab_contrib/` — 社区贡献

实验性扩展（推进器、触觉传感器等）。

---

## 四、脚本目录 `scripts/`

### 4.1 教程 `scripts/tutorials/`（从零开始学）

```
tutorials/
├── 00_sim/           # 仿真基础
│   ├── create_empty.py        # 创建空场景
│   ├── launch_app.py          # 启动 Isaac Sim
│   ├── spawn_prims.py         # 生成物体
│   └── log_time.py            # 仿真时间
├── 01_assets/        # 资产操作
│   ├── run_articulation.py    # 操作机器人
│   ├── run_rigid_object.py    # 操作刚体
│   └── add_new_robot.py       # 添加新机器人
├── 02_scene/         # 场景管理
│   └── create_scene.py
├── 03_envs/          # 环境创建
│   ├── create_cartpole_base_env.py
│   └── run_cartpole_rl_env.py
├── 04_sensors/       # 传感器
│   ├── run_usd_camera.py      # 相机
│   └── run_ray_caster.py      # 射线投射
└── 05_controllers/   # 控制器
    ├── run_diff_ik.py         # 微分 IK
    └── run_osc.py             # 操作空间控制
```

### 4.2 RL 训练 `scripts/reinforcement_learning/`

```
reinforcement_learning/
├── rsl_rl/           # RSL-RL（推荐）
│   ├── train.py      # 训练
│   └── play.py       # 推理/回放
├── rl_games/         # RL Games
├── sb3/              # Stable Baselines3
└── skrl/             # SKRL
```

### 4.3 工具脚本 `scripts/tools/`

```
tools/
├── convert_urdf.py      # URDF → USD 转换
├── convert_mjcf.py      # MJCF (MuJoCo) → USD 转换
├── convert_mesh.py      # 网格格式转换
├── record_demos.py      # 记录演示数据
├── hdf5_to_mp4.py       # HDF5 → MP4 视频
└── ...
```

### 4.4 环境工具 `scripts/environments/`

```
environments/
├── list_envs.py          # 列出所有可用环境
├── random_agent.py       # 随机动作用于测试
├── zero_agent.py         # 零动作用于测试
└── teleoperation/        # 遥操作
```

---

## 五、应用配置 `apps/`

控制 Isaac Sim 的启动模式：

| 文件 | 用途 |
|------|------|
| `isaaclab.python.kit` | 默认 GUI 模式 |
| `isaaclab.python.headless.kit` | 无头模式（训练用） |
| `rendering_modes/quality.kit` | 高画质渲染 |
| `rendering_modes/performance.kit` | 高性能渲染 |

---

## 六、CLI 工具 `isaaclab.sh`

统一入口，常用命令：

```bash
# 安装 Isaac Lab
./isaaclab.sh --install

# 列出所有环境
python scripts/environments/list_envs.py

# 训练（以 RSL-RL + Ant 为例）
python scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Ant-Direct-v0

# 推理/回放
python scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-Ant-Direct-v0

# 运行教程
python scripts/tutorials/00_sim/create_empty.py

# 格式化代码
./isaaclab.sh --format

# 运行测试
./isaaclab.sh --test
```

---

## 七、MDP 组件（核心概念）

位于 `source/isaaclab/isaaclab/envs/mdp/`，是构建环境的积木：

| 组件 | 文件 | 作用 |
|------|------|------|
| 动作 | `actions/` | 定义动作空间（关节控制、任务空间等） |
| 观测 | `observations.py` | 定义观测内容 |
| 奖励 | `rewards.py` | 定义奖励函数 |
| 终止 | `terminations.py` | 定义终止条件 |
| 事件 | `events.py` | 定义重置和域随机化 |
| 命令 | `commands/` | 定义目标命令（速度、位姿等） |
| 课程 | `curriculums.py` | 定义训练课程 |

---

## 八、传感器系统

| 传感器 | 功能 |
|--------|------|
| `Camera` / `TiledCamera` | RGB、深度、分割图像 |
| `ContactSensor` | 接触力检测 |
| `RayCaster` | 射线投射（LIDAR、高度扫描） |
| `IMU` | 惯性测量 |
| `FrameTransformer` | 坐标变换 |

---

## 九、快速上手流程

### 1. 列出可用环境
```bash
python scripts/environments/list_envs.py
```

### 2. 运行随机/零动作用于测试
```bash
python scripts/environments/random_agent.py --task Isaac-Cartpole-Direct-v0
```

### 3. 训练 RL 策略
```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Cartpole-Direct-v0 \
    --num_envs 64
```

### 4. 回放训练好的策略
```bash
python scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Cartpole-Direct-v0
```

---

## 十、架构总览图

```
┌────────────────────────────────────────────────────────────────┐
│                        Isaac Lab 架构                          │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐    │
│  │ isaaclab_core│  │ isaaclab_tasks│  │  isaaclab_rl     │    │
│  │              │  │              │  │  (RL框架封装)     │    │
│  │ • Assets     │  │ • 环境定义   │  │  • RSL-RL        │    │
│  │ • Sensors    │  │ • MDP组件    │  │  • RL Games      │    │
│  │ • Managers   │  │ • 机器人配置 │  │  • SB3 / SKRL    │    │
│  │ • Controllers│  │              │  │                  │    │
│  │ • Terrains   │  └──────────────┘  └──────────────────┘    │
│  └──────────────┘                                            │
│         │                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐    │
│  │isaaclab_assets│  │isaaclab_mimic│  │ isaaclab_contrib │    │
│  │  (机器人库)   │  │ (数据生成)    │  │  (社区扩展)      │    │
│  │ 26+ 机器人    │  │ MimicGen     │  │  推进器、触觉等   │    │
│  └──────────────┘  └──────────────┘  └──────────────────┘    │
│                                                                │
├────────────────────────────────────────────────────────────────┤
│                    NVIDIA Isaac Sim (底层仿真引擎)             │
└────────────────────────────────────────────────────────────────┘
```

---

## 十一、关键配置文件

| 文件 | 用途 |
|------|------|
| `pyproject.toml` | 顶层工具配置（ruff、pyright、pytest） |
| `environment.yml` | Conda 环境定义 |
| `VERSION` | 版本号：2.3.2 |
| `source/*/config/extension.toml` | 每个扩展包的元数据 |
| `source/*/setup.py` | 每个扩展包的构建配置 |

---

## 十二、学习路径建议

```
第 1 步：运行教程
  └── scripts/tutorials/00_sim/ → 01_assets/ → 02_scene/

第 2 步：理解环境
  └── scripts/tutorials/03_envs/
  └── scripts/environments/list_envs.py

第 3 步：运行现有任务
  └── scripts/environments/random_agent.py
  └── scripts/reinforcement_learning/rsl_rl/train.py

第 4 步：自定义环境
  └── 参考 source/isaaclab_tasks/ 中的任务定义
  └── 参考 docs/source/how-to/ 目录

第 5 步：深入架构
  └── 阅读 source/isaaclab/isaaclab/envs/ 源码
  └── 阅读 docs/source/overview/core-concepts/
```

---

*文档生成时间：2026-09-08*
