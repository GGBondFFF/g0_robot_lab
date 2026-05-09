# g0_robot_lab Run Commands

本文档记录 `g0_robot_lab` 项目的常用运行指令，包括资产检查、URDF 转 USD、环境 smoke test、训练、play、policy 导出、日志检查、Git 提交和后续部署准备。

当前主线项目：

```bash
/home/lz/g0_robot_lab/g0_robot_lab
```

当前任务 ID：

```bash
G0-Velocity-v0
```

Isaac Lab 路径：

```bash
/home/lz/IsaacLab
```

运行 Isaac Lab 脚本统一使用：

```bash
/home/lz/IsaacLab/isaaclab.sh -p
```

注意：当前项目必须先通过 Isaac Lab 的 `AppLauncher` / `SimulationApp` 启动后，再 import `g0_robot_lab`。不要在普通 Python 中直接执行：

```python
import g0_robot_lab
```

否则可能出现：

```text
ModuleNotFoundError: No module named 'pxr'
```

---

## 1. 进入项目目录

```bash
cd /home/lz/g0_robot_lab/g0_robot_lab
```

确认当前 conda 环境：

```bash
conda activate g0_isaaclab
```

---

## 2. 当前项目主线

当前项目是：

```text
g0_robot_lab
```

不要混入：

```text
humanoid_lab_v0
```

不要回到旧目录：

```text
source/g0_robot_lab/g0_robot_lab/tasks/manager_based/
```

当前任务结构应保持为：

```text
source/g0_robot_lab/g0_robot_lab/tasks/locomotion/
```

当前 G0 机器人资产目录：

```text
source/g0_robot_lab/g0_robot_lab/assets/robots/g0/
```

---

## 3. 检查项目结构

```bash
find source/g0_robot_lab/g0_robot_lab -maxdepth 5 -type d | sort
find docs -maxdepth 2 -type f | sort
```

关键目录应包括：

```text
source/g0_robot_lab/g0_robot_lab/assets/robots/g0/
source/g0_robot_lab/g0_robot_lab/assets/robots/g0/urdf/
source/g0_robot_lab/g0_robot_lab/assets/robots/g0/usd/
source/g0_robot_lab/g0_robot_lab/assets/robots/g0/meshes/
source/g0_robot_lab/g0_robot_lab/tasks/locomotion/
source/g0_robot_lab/g0_robot_lab/tasks/locomotion/agents/
source/g0_robot_lab/g0_robot_lab/tasks/locomotion/mdp/
source/g0_robot_lab/g0_robot_lab/tasks/locomotion/robots/g0/
```

---

## 4. 检查 G0 URDF

进入 G0 资产目录：

```bash
cd /home/lz/g0_robot_lab/g0_robot_lab/source/g0_robot_lab/g0_robot_lab/assets/robots/g0
```

运行 URDF 检查脚本：

```bash
python3 inspect_g0_urdf.py
```

期望末尾输出：

```text
G0 URDF inspection passed.
```

检查完成后回到项目根目录：

```bash
cd /home/lz/g0_robot_lab/g0_robot_lab
```

当前 G0 URDF 路径：

```text
source/g0_robot_lab/g0_robot_lab/assets/robots/g0/urdf/g0.urdf
```

当前 mesh 路径：

```text
source/g0_robot_lab/g0_robot_lab/assets/robots/g0/meshes/
```

---

## 5. URDF 转 USD

输入：

```text
source/g0_robot_lab/g0_robot_lab/assets/robots/g0/urdf/g0.urdf
```

输出：

```text
source/g0_robot_lab/g0_robot_lab/assets/robots/g0/usd/g0.usd
```

运行：

```bash
/home/lz/IsaacLab/isaaclab.sh -p scripts/tools/convert_g0_urdf_to_usd.py --headless
```

检查生成结果：

```bash
ls -lh source/g0_robot_lab/g0_robot_lab/assets/robots/g0/usd/g0.usd
```

成功后应能看到：

```text
source/g0_robot_lab/g0_robot_lab/assets/robots/g0/usd/g0.usd
```

这说明：

```text
g0.urdf -> g0.usd
```

已经转换完成。

注意：当前仓库的 `.gitignore` 忽略了 USD 文件。因此如果从新机器 clone 仓库，通常需要重新运行本节命令生成 `g0.usd`。

---

## 6. 检查 G0_CFG import

注意：必须 AppLauncher-first。

```bash
/home/lz/IsaacLab/isaaclab.sh -p -c "
from isaaclab.app import AppLauncher
app_launcher = AppLauncher({'headless': True})
simulation_app = app_launcher.app

from g0_robot_lab.assets.robots.g0 import G0_CFG
print('G0_CFG import OK')
print(G0_CFG.spawn.usd_path)

simulation_app.close()
"
```

期望输出：

```text
G0_CFG import OK
/home/lz/g0_robot_lab/g0_robot_lab/source/g0_robot_lab/g0_robot_lab/assets/robots/g0/usd/g0.usd
```

---

## 7. 检查 task 注册

```bash
/home/lz/IsaacLab/isaaclab.sh -p -c "
from isaaclab.app import AppLauncher
app_launcher = AppLauncher({'headless': True})
simulation_app = app_launcher.app

import gymnasium as gym
import g0_robot_lab
import g0_robot_lab.tasks

matched = [env_id for env_id in gym.registry.keys() if 'G0' in env_id or 'g0' in env_id]
print('Matched env ids:')
for env_id in matched:
    print(env_id)

simulation_app.close()
"
```

期望看到：

```text
G0-Velocity-v0
```

---

## 8. 检查 Cartpole 残留

当前 `velocity_env_cfg.py` 不应再包含 Cartpole 模板内容。

```bash
grep -R "CARTPOLE_CFG\|cartpole\|slider_to_cart\|cart_to_pole" -n \
source/g0_robot_lab/g0_robot_lab/tasks/locomotion/robots/g0/velocity_env_cfg.py
```

理想情况：没有输出。

如果有输出，说明 `velocity_env_cfg.py` 中还有旧模板残留，需要继续清理。

---

## 9. 环境创建 smoke test

ManagerBased 环境不能裸 `gym.make("G0-Velocity-v0")`。必须传入 `cfg=env_cfg`。

```bash
/home/lz/IsaacLab/isaaclab.sh -p -c "
from isaaclab.app import AppLauncher
app_launcher = AppLauncher({'headless': True})
simulation_app = app_launcher.app

import gymnasium as gym
import g0_robot_lab
import g0_robot_lab.tasks

from g0_robot_lab.tasks.locomotion.robots.g0.velocity_env_cfg import G0RobotLabEnvCfg

env_cfg = G0RobotLabEnvCfg()
env_cfg.scene.num_envs = 1
env_cfg.sim.device = 'cuda:0'

env = gym.make('G0-Velocity-v0', cfg=env_cfg)

print('[INFO] env created')
print('[INFO] observation space:', env.observation_space)
print('[INFO] action space:', env.action_space)

env.close()
simulation_app.close()
"
```

当前最小环境的期望结果：

```text
Action space: 22
Observation space: 44
```

含义：

```text
22 = G0 的 22 个 revolute joints
44 = joint_pos_rel(22) + joint_vel_rel(22)
```

---

## 10. 当前最小环境状态

当前 `G0-Velocity-v0` 是 smoke-test 环境，不是正式 locomotion 环境。

当前 action：

```text
joint_pos: 22
```

当前 observation：

```text
joint_pos_rel: 22
joint_vel_rel: 22
total: 44
```

当前 reward：

```text
alive
terminating
```

当前 termination：

```text
time_out
```

当前没有：

```text
base velocity command
tracking_lin_vel reward
tracking_ang_vel reward
upright reward
feet contact reward
joint limit penalty
action smoothness penalty
fall termination
terrain curriculum
domain randomization
```

---

## 11. 训练 smoke test

最小训练测试：

```bash
cd /home/lz/g0_robot_lab/g0_robot_lab

/home/lz/IsaacLab/isaaclab.sh -p scripts/rsl_rl/train.py \
  --task G0-Velocity-v0 \
  --num_envs 16 \
  --max_iterations 2 \
  --headless
```

这一步不是为了训练出可用策略，而是验证：

```text
1. task 配置能被 Hydra 解析
2. env 能被 train.py 创建
3. RSL-RL runner 能启动
4. actor / critic 网络维度匹配
5. checkpoint 能正常保存
```

训练完成后检查 checkpoint：

```bash
find logs/rsl_rl/g0_velocity -name "model_*.pt" | sort
```

示例：

```text
logs/rsl_rl/g0_velocity/2026-05-09_14-44-10/model_0.pt
logs/rsl_rl/g0_velocity/2026-05-09_14-44-10/model_1.pt
```

---

## 12. 设置 checkpoint 变量

使用最新 checkpoint：

```bash
cd /home/lz/g0_robot_lab/g0_robot_lab

CKPT=$(find logs/rsl_rl/g0_velocity -name "model_*.pt" | sort | tail -1)
echo "$CKPT"
```

也可以手动指定：

```bash
CKPT=logs/rsl_rl/g0_velocity/2026-05-09_14-44-10/model_1.pt
echo "$CKPT"
```

---

## 13. Play headless smoke test

建议先用 headless + video 模式验证 play 链路。

```bash
/home/lz/IsaacLab/isaaclab.sh -p scripts/rsl_rl/play.py \
  --task G0-Velocity-v0 \
  --num_envs 1 \
  --checkpoint "$CKPT" \
  --headless \
  --video \
  --video_length 100
```

为什么要加：

```text
--video --video_length 100
```

因为 headless play 如果不录视频，脚本可能会持续运行。加 video 后，达到 `video_length` 会自动退出。

这一步用于验证：

```text
1. play.py 能解析 G0-Velocity-v0
2. play.py 能创建环境
3. checkpoint 能被加载
4. policy 能推理
5. env.step(actions) 能运行
6. policy.pt / policy.onnx 能导出
7. play video 能生成
```

---

## 14. 检查 policy 导出

play.py 会导出 JIT 和 ONNX policy。

```bash
find logs/rsl_rl/g0_velocity -path "*/exported/*" -type f | sort
```

期望看到：

```text
policy.pt
policy.onnx
```

如果指定某一次运行：

```bash
find logs/rsl_rl/g0_velocity/2026-05-09_14-44-10 -path "*/exported/*" -type f | sort
```

---

## 15. 检查 play 视频

```bash
find logs/rsl_rl/g0_velocity -path "*/videos/play/*" -type f | sort
```

如果指定某一次运行：

```bash
find logs/rsl_rl/g0_velocity/2026-05-09_14-44-10 -path "*/videos/play/*" -type f | sort
```

---

## 16. GUI play

确认 headless play 能通后，再开 GUI 查看机器人显示、姿态和初始高度。

```bash
cd /home/lz/g0_robot_lab/g0_robot_lab

CKPT=$(find logs/rsl_rl/g0_velocity -name "model_*.pt" | sort | tail -1)

/home/lz/IsaacLab/isaaclab.sh -p scripts/rsl_rl/play.py \
  --task G0-Velocity-v0 \
  --num_envs 1 \
  --checkpoint "$CKPT"
```

当前阶段 GUI play 只检查：

```text
1. G0 是否能显示出来
2. mesh 是否缺失
3. 初始高度是否明显不对
4. 是否穿地
5. 是否一启动就爆飞
6. camera 是否能看到机器人
```

当前 reward 仍是占位任务，不要期待机器人能行走。

---

## 17. 当前已验证状态

当前已经验证通过：

```text
1. G0 URDF 清洗通过
2. g0.urdf 转 g0.usd 成功
3. G0_CFG import 成功
4. G0-Velocity-v0 env 创建成功
5. train smoke test 成功
6. play headless smoke test 成功
7. GUI play 能正常显示 G0
```

当前已知问题：

```text
1. 初始高度还不准确
2. 当前 reward 只是占位
3. 当前环境还不是正式 locomotion 环境
4. RSL-RL 配置仍有 deprecated warning
5. obs_groups 仍可后续显式配置
6. 当前 G0_CFG 中 disable_gravity 需要在正式 locomotion 前确认
```

---

## 18. 常见 warning 说明

### 18.1 RSL-RL policy deprecated warning

如果看到：

```text
The `policy` configuration is deprecated for rsl-rl >= 4.0.0
```

当前不阻塞运行。后续可以把 `rsl_rl_ppo_cfg.py` 从旧的：

```text
policy = RslRlPpoActorCriticCfg(...)
```

升级成更明确的：

```text
actor
critic
distribution_cfg
obs_groups
```

---

### 18.2 obs_groups warning

如果看到：

```text
obs_groups is empty
obs_groups does not contain the 'actor' key
obs_groups does not contain the 'critic' key
```

当前不阻塞运行。RSL-RL 会自动把 `policy` observation group 用作 actor 和 critic。

后续正式训练前建议显式配置：

```text
actor: ["policy"]
critic: ["policy"]
```

---

### 18.3 Fabric visual path warning

如果看到类似：

```text
getAttributeCount called on non-existent path ...
```

当前不一定阻塞。需要在 GUI play 中确认 mesh 是否正常显示。

---

### 18.4 初始高度不对

当前 GUI play 已经确认 G0 能显示，但初始高度还不准确。

后续需要调整：

```text
source/g0_robot_lab/g0_robot_lab/assets/robots/g0/g0.py
```

中的：

```python
init_state=ArticulationCfg.InitialStateCfg(
    pos=(0.0, 0.0, ...),
)
```

同时还要结合默认关节角和脚底接触高度来调整。

---

## 19. 后续正式 locomotion 任务顺序

当前 smoke test 完成后，下一阶段应按以下顺序推进：

```text
1. 修正 G0 初始高度
2. 设置合理默认站立关节角
3. 明确 22 个 joint 的顺序和分组
4. 从 joint_pos action 过渡到正式 locomotion action 配置
5. 添加 base velocity command
6. 添加 locomotion observations
7. 添加 fall termination
8. 添加 tracking velocity reward
9. 添加 upright / base height / joint limit / action smoothness reward
10. 添加 feet contact 和 gait 相关 reward
11. 进行小规模训练
12. 进行 GUI play 检查
13. 再扩大 num_envs 和 max_iterations
```

---

## 20. 部署相关输出

当前 play 后会导出：

```text
policy.pt
policy.onnx
```

检查命令：

```bash
find logs/rsl_rl/g0_velocity -path "*/exported/*" -type f | sort
```

---

### 20.1 policy.pt

用途：

```text
PyTorch JIT / TorchScript 形式，适合 Python 或 C++ TorchScript 推理链路。
```

---

### 20.2 policy.onnx

用途：

```text
ONNX 形式，适合后续转 TensorRT、ONNX Runtime 或板端推理框架。
```

---

### 20.3 部署前必须确认

正式部署前必须固定：

```text
1. observation 维度
2. observation 顺序
3. action 维度
4. action 顺序
5. action scale
6. default joint position
7. policy 输入归一化方式
8. policy 输出含义
9. 控制频率
10. joint 名称到实机电机 ID 的映射
```

当前 smoke-test policy 不能用于实机部署，只用于验证训练和 play 链路。

---

## 21. Git 提交建议

查看修改：

```bash
git status
```

建议加入：

```bash
git add \
  scripts/tools/convert_g0_urdf_to_usd.py \
  source/g0_robot_lab/g0_robot_lab/assets/robots/g0 \
  source/g0_robot_lab/g0_robot_lab/tasks/locomotion/robots/g0/velocity_env_cfg.py \
  docs/run_commands.md
```

不要提交：

```text
logs/
```

提交：

```bash
git commit -m "Add G0 USD asset and smoke-test commands"
git push
```

注意：如果 `.gitignore` 忽略了 `*.usd`，那么 `g0.usd` 不会被提交。此时新机器 clone 后必须先运行 URDF 转 USD 命令。

---

## 22. 推荐 .gitignore 检查

确认是否忽略 logs：

```bash
cat .gitignore
```

如果没有忽略 logs，建议加入：

```gitignore
# Isaac Lab / RSL-RL outputs
logs/
outputs/
wandb/
*.mp4
*.avi
*.npy
*.npz
```

然后提交：

```bash
git add .gitignore
git commit -m "Ignore training outputs"
git push
```

---

## 23. 常用命令总览

### 23.1 URDF 转 USD

```bash
/home/lz/IsaacLab/isaaclab.sh -p scripts/tools/convert_g0_urdf_to_usd.py --headless
```

---

### 23.2 环境创建 smoke test

```bash
/home/lz/IsaacLab/isaaclab.sh -p -c "
from isaaclab.app import AppLauncher
app_launcher = AppLauncher({'headless': True})
simulation_app = app_launcher.app

import gymnasium as gym
import g0_robot_lab
import g0_robot_lab.tasks

from g0_robot_lab.tasks.locomotion.robots.g0.velocity_env_cfg import G0RobotLabEnvCfg

env_cfg = G0RobotLabEnvCfg()
env_cfg.scene.num_envs = 1
env_cfg.sim.device = 'cuda:0'

env = gym.make('G0-Velocity-v0', cfg=env_cfg)

print('[INFO] env created')
print('[INFO] observation space:', env.observation_space)
print('[INFO] action space:', env.action_space)

env.close()
simulation_app.close()
"
```

---

### 23.3 训练 smoke test

```bash
/home/lz/IsaacLab/isaaclab.sh -p scripts/rsl_rl/train.py \
  --task G0-Velocity-v0 \
  --num_envs 16 \
  --max_iterations 2 \
  --headless
```

---

### 23.4 正式训练模板

后续正式训练可以从这个命令开始扩展：

```bash
/home/lz/IsaacLab/isaaclab.sh -p scripts/rsl_rl/train.py \
  --task G0-Velocity-v0 \
  --num_envs 4096 \
  --max_iterations 20000 \
  --headless
```

注意：当前 reward 还是占位，暂时不要直接跑长训练。正式训练前应先完成 locomotion reward、termination、command 和 observation 设计。

---

### 23.5 Play headless smoke test

```bash
CKPT=$(find logs/rsl_rl/g0_velocity -name "model_*.pt" | sort | tail -1)

/home/lz/IsaacLab/isaaclab.sh -p scripts/rsl_rl/play.py \
  --task G0-Velocity-v0 \
  --num_envs 1 \
  --checkpoint "$CKPT" \
  --headless \
  --video \
  --video_length 100
```

---

### 23.6 GUI play

```bash
CKPT=$(find logs/rsl_rl/g0_velocity -name "model_*.pt" | sort | tail -1)

/home/lz/IsaacLab/isaaclab.sh -p scripts/rsl_rl/play.py \
  --task G0-Velocity-v0 \
  --num_envs 1 \
  --checkpoint "$CKPT"
```

---

### 23.7 检查 checkpoint

```bash
find logs/rsl_rl/g0_velocity -name "model_*.pt" | sort
```

---

### 23.8 检查导出的 policy

```bash
find logs/rsl_rl/g0_velocity -path "*/exported/*" -type f | sort
```

---

### 23.9 检查 play 视频

```bash
find logs/rsl_rl/g0_velocity -path "*/videos/play/*" -type f | sort
```

---

## 24. 当前阶段结论

当前阶段目标已经完成：

```text
G0 机器人资产已经成功接入 Isaac Lab。
G0-Velocity-v0 已经可以创建环境。
训练 smoke test 已经通过。
play smoke test 已经通过。
GUI 中能够正常显示 G0。
```

下一阶段重点不是继续验证基础链路，而是开始构建真正的 locomotion 任务：

```text
1. 修初始高度
2. 设默认站立姿态
3. 配 command
4. 配 observations
5. 配 rewards
6. 配 terminations
7. 开始正式训练
```
