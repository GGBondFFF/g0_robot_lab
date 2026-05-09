# g0_robot_lab 项目结构说明

## 1. 当前项目定位

`g0_robot_lab` 是基于 Isaac Lab 新生成的 external project 模板整理出来的项目。

当前主线是 `g0_robot_lab`，不是 `humanoid_lab_v0`。后续不要将 `humanoid_lab_v0` 的目录结构、任务名、checkpoint、调试问题或机器人配置混入 `g0_robot_lab`。

当前阶段的目标不是马上训练，而是先建立一个干净、稳定、可回退的 G0 Isaac Lab 项目骨架。后续再在这个骨架上逐步加入 G0 机器人资产、locomotion 任务配置、reward、observation、termination、event、train.py 和 play.py。

## 2. 当前固定目录结构

```text
g0_robot_lab/
├── scripts/
│   └── rsl_rl/
│       ├── train.py
│       └── play.py
├── source/
│   └── g0_robot_lab/
│       └── g0_robot_lab/
│           ├── assets/
│           │   └── robots/
│           │       └── g0/
│           │           ├── g0.py
│           │           ├── urdf/
│           │           ├── usd/
│           │           └── meshes/
│           ├── tasks/
│           │   └── locomotion/
│           │       ├── __init__.py
│           │       ├── agents/
│           │       │   ├── __init__.py
│           │       │   └── rsl_rl_ppo_cfg.py
│           │       ├── mdp/
│           │       │   ├── __init__.py
│           │       │   ├── rewards.py
│           │       │   ├── observations.py
│           │       │   ├── terminations.py
│           │       │   └── events.py
│           │       └── robots/
│           │           └── g0/
│           │               ├── __init__.py
│           │               └── velocity_env_cfg.py
│           └── __init__.py
└── docs/
    ├── 2026_05_08_work_log.md
    ├── project_structure.md
    └── run_commands.md
```

## 3. 目录职责说明

### 3.1 `scripts/`

`scripts/` 用于存放项目运行入口。

当前计划保留：

```text
scripts/rsl_rl/train.py
scripts/rsl_rl/play.py
```

`train.py` 用于训练 `G0-Velocity-v0`。`play.py` 用于加载训练出的 checkpoint 并在 Isaac Lab 中回放策略。

后续需要特别注意：这两个脚本中必须先启动 `AppLauncher`，再 import `g0_robot_lab` 和 task 注册相关模块。

### 3.2 `source/g0_robot_lab/g0_robot_lab/assets/robots/g0/`

该目录用于存放 G0 机器人的资产配置和模型文件。

目标结构：

```text
assets/robots/g0/
├── g0.py
├── urdf/
├── usd/
└── meshes/
```

各部分职责：

```text
g0.py     ：定义 G0 机器人 ArticulationCfg、关节名、初始姿态、执行器参数等
urdf/     ：存放原始或整理后的 G0 URDF 文件
usd/      ：存放从 URDF 转换得到的 G0 USD 文件
meshes/   ：存放 URDF/USD 依赖的 mesh 文件，例如 .stl、.obj、.dae
```

### 3.3 `source/g0_robot_lab/g0_robot_lab/tasks/locomotion/`

该目录是当前项目的主任务目录。

它取代了 Isaac Lab 模板默认生成的：

```text
source/g0_robot_lab/g0_robot_lab/tasks/manager_based/g0_robot_lab/
```

当前主线统一使用：

```text
source/g0_robot_lab/g0_robot_lab/tasks/locomotion/
```

### 3.4 `tasks/locomotion/agents/`

该目录存放 rsl_rl 的 PPO 配置。

当前文件：

```text
tasks/locomotion/agents/rsl_rl_ppo_cfg.py
```

当前类名仍然沿用模板配置中的：

```text
PPORunnerCfg
```

后续可以根据项目需要改为更明确的名字，例如：

```text
G0VelocityPPORunnerCfg
```

但修改类名时，必须同步更新 `tasks/locomotion/__init__.py` 中的 `rsl_rl_cfg_entry_point`。

### 3.5 `tasks/locomotion/mdp/`

该目录用于存放 locomotion 任务的 MDP 组成部分。

目标文件：

```text
mdp/
├── __init__.py
├── rewards.py
├── observations.py
├── terminations.py
└── events.py
```

各文件职责：

```text
rewards.py       ：奖励函数，例如速度跟踪、姿态稳定、能耗惩罚、足端接触奖励等
observations.py  ：观测函数，例如 base velocity、projected gravity、joint pos/vel、command 等
terminations.py  ：终止条件，例如摔倒、高度过低、姿态过大、非法状态等
events.py        ：reset 和 domain randomization 事件，例如随机初始关节角、随机质量、随机摩擦等
```

当前 `mdp/__init__.py` 会导入 Isaac Lab 内置 MDP 项和本项目自定义项。后续添加新函数后，应确保在 `mdp/__init__.py` 中可以被正确导出。

### 3.6 `tasks/locomotion/robots/g0/`

该目录用于存放 G0 locomotion 任务的环境配置。

当前核心文件：

```text
tasks/locomotion/robots/g0/velocity_env_cfg.py
```

该文件负责定义：

```text
scene
robot asset
actions
observations
rewards
terminations
events
simulation settings
viewer settings
episode length
decimation
```

当前这个文件是从 Isaac Lab 模板中的 `g0_robot_lab_env_cfg.py` 迁移来的，因此目前仍然主要是 Cartpole 模板内容。后续需要逐步替换为真正的 G0 locomotion 配置。

## 4. 已完成的结构迁移

Isaac Lab 模板默认生成的 task 路径原本是：

```text
source/g0_robot_lab/g0_robot_lab/tasks/manager_based/g0_robot_lab/
```

当前已经迁移为：

```text
source/g0_robot_lab/g0_robot_lab/tasks/locomotion/
```

旧的 `manager_based` 目录已经删除。

已完成迁移的主要文件对应关系如下：

```text
旧路径：tasks/manager_based/g0_robot_lab/g0_robot_lab_env_cfg.py
新路径：tasks/locomotion/robots/g0/velocity_env_cfg.py

旧路径：tasks/manager_based/g0_robot_lab/agents/rsl_rl_ppo_cfg.py
新路径：tasks/locomotion/agents/rsl_rl_ppo_cfg.py

旧路径：tasks/manager_based/g0_robot_lab/mdp/rewards.py
新路径：tasks/locomotion/mdp/rewards.py
```

补充创建的新文件：

```text
tasks/locomotion/agents/__init__.py
tasks/locomotion/mdp/observations.py
tasks/locomotion/mdp/terminations.py
tasks/locomotion/mdp/events.py
tasks/locomotion/robots/__init__.py
tasks/locomotion/robots/g0/__init__.py
```

## 5. 当前已注册任务

当前 Gym task id：

```text
G0-Velocity-v0
```

注册位置：

```text
source/g0_robot_lab/g0_robot_lab/tasks/locomotion/__init__.py
```

当前注册入口：

```text
env_cfg_entry_point:
g0_robot_lab.tasks.locomotion.robots.g0.velocity_env_cfg:G0RobotLabEnvCfg

rsl_rl_cfg_entry_point:
g0_robot_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:PPORunnerCfg
```

当前已经通过 AppLauncher 启动后的注册检查，能够在 Gym registry 中看到：

```text
G0-Velocity-v0
```

## 6. AppLauncher 和 import 顺序要求

在 Isaac Sim 5.1 / Isaac Lab 当前环境下，不能在普通 Python 进程中直接先执行：

```python
import g0_robot_lab
```

否则可能出现：

```text
ModuleNotFoundError: No module named 'pxr'
```

原因不是 `g0_robot_lab` 项目结构错误，而是 Isaac Sim 相关的 USD / Omniverse 模块需要在 SimulationApp 启动后加载。

正确顺序是：

```python
from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import g0_robot_lab
import g0_robot_lab.tasks
```

因此后续 `scripts/rsl_rl/train.py` 和 `scripts/rsl_rl/play.py` 必须遵守：

```text
先 AppLauncher / SimulationApp
再 import g0_robot_lab / g0_robot_lab.tasks / task registry
```

不要在脚本最顶部过早写：

```python
import g0_robot_lab
```

## 7. 当前仍然是模板内容的部分

虽然目录结构已经整理完成，但 `velocity_env_cfg.py` 当前仍然主要来自 Cartpole 模板，里面仍然可能包含：

```text
CARTPOLE_CFG
slider_to_cart
cart_to_pole
cartpole reward
cartpole termination
```

这部分后续需要逐步替换为真正的 G0 机器人 locomotion 配置。

当前这一步的重点不是让 G0 直接训练，而是先完成：

```text
1. 项目结构固定
2. task 路径整理
3. task 注册成功
4. Git 历史可回退
5. 文档记录清楚
```

## 8. 当前 Git 状态

当前本地 Git 仓库已经创建，根目录正确，主分支为：

```text
main
```

已完成关键提交：

```text
Initial Isaac Lab template for g0_robot_lab
Migrate template task to locomotion structure
```

当前计划新增文档提交：

```text
Add project documentation for g0_robot_lab
```

## 9. 后续推进计划

下一阶段建议按以下顺序推进：

```text
1. 补齐 assets/robots/g0 目录
2. 添加 g0.py 机器人资产配置
3. 放入或转换 G0 的 URDF / USD / meshes
4. 将 velocity_env_cfg.py 从 Cartpole 模板改成 G0 robot cfg
5. 重写 observations.py
6. 重写 rewards.py
7. 重写 terminations.py
8. 重写 events.py
9. 检查 train.py 和 play.py 是否遵守 AppLauncher-first 导入顺序
10. 跑 G0-Velocity-v0 smoke test
11. 确认 smoke test 后再开始正式 locomotion 训练
```

## 10. 当前阶段结论

当前 `g0_robot_lab` 已经从 Isaac Lab 默认模板结构整理为项目目标结构。

当前有效主线是：

```text
g0_robot_lab → tasks/locomotion → G0-Velocity-v0
```
 后续所有分析和修改都应该基于这个结构，不再使用 humanoid_lab_v0 或旧的 manager_based 目录作为当前主线。