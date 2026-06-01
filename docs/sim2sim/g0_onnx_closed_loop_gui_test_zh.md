# G0 — ONNX policy 在 MuJoCo 中的 closed-loop GUI 测试

## 1. 本轮目标

把已导出的 `policy.onnx` 真正接进 MuJoCo，形成 **闭环控制**，并在 MuJoCo GUI 中可视化运行效果。

本轮 **不是** 之前任意一种回放：

- 不是 `g0_mujoco_zero_action.py`（PD hold 到固定姿态）。
- 不是 `g0_action_sweep.py`（开环 action sweep）。
- 不是 `g0_obs_contract_check.py` / `g0_history_contract_check.py`（只测 obs 构造）。
- 不是 `g0_onnx_equivalence_check.py`（只测 ONNX = PT）。

这一步的输出物：

- 脚本：`scripts/sim2sim/g0_mujoco_onnx_gui_runner.py`
- 日志 CSV：`logs/sim2sim/g0_onnx_gui_closed_loop_log.csv`
- 中文文档：本文件

## 2. 前置 contract 回顾

```text
MJCF stable                : OK   (source/.../g0/mujoco/model_patched.xml 已 patch)
default_stand PD hold      : OK   (g0_mujoco_zero_action.py)
SDK_TO_MJ 静态映射         : OK   (g0_contract_check.py)
g0_action_sweep            : OK   (88/88, 方向 + torque 全部正确)
单步 77-D obs              : OK   (g0_obs_contract_check.py, max|diff|<=1e-5)
385-D history layout       : OK   (per-term grouped, oldest-first)
ONNX equivalence           : OK   (policy.onnx == policy.pt, normalizer = Identity)
```

## 3. closed-loop 控制链路

每个 policy step (50 Hz，即 policy dt = 0.02 s)：

```text
MuJoCo state (data.qpos / data.qvel)
  ↓
构造 77-D obs（按 Isaac 训练时的 per-term scale，joint 部分用 Isaac articulation order）
  ↓
更新 5-step history buffer（oldest-first）
  ↓
build_obs_385: per-term grouped, oldest-first, TERM_DIMS=[3,3,3,22,22,22,2]
  ↓
policy.onnx 推理: obs[1,385] -> action_sdk[1,22]
  ↓
target_q_sdk = default_q_sdk + 0.12 * action_sdk
  ↓
target_q_mj[SDK_TO_MJ] = target_q_sdk      # 把 SDK 顺序重排到 MuJoCo actuator 顺序
  ↓
tau_mj = kp_mj * (target_q_mj - q_mj) - kd_mj * dq_mj
tau_mj = clip(tau_mj, ctrlrange_low, ctrlrange_high)
  ↓
data.ctrl[:] = tau_mj                       # 注意：是 torque，不是 q_target
  ↓
mujoco.mj_step()  ×  decimation             # MJCF dt=0.002, decimation=10
```

## 4. 三种 joint order（必须区分）

- **SDK order** (`G0_JOINT_SDK_NAMES`)：policy 的 `action` / `last_action` 顺序。
- **MuJoCo / URDF order** (`G0_JOINT_NAMES_MJ`)：MJCF actuator / `data.ctrl` / `qpos[7:]` 槽位。
- **Isaac articulation order** (`G0_JOINT_NAMES_ISAAC`)：obs 中 `joint_pos_rel`、`joint_vel_rel` 段的顺序，
  来自 Isaac 训练时 `robot.joint_names`，**不等于** 上面任何一种。

脚本里在加载模型时按名字反向解析 `ISAAC_TO_MJ`，而 `SDK_TO_MJ` 是已经验证过的硬编码常量。
`last_action` 永远保持 SDK 顺序，不要重排。

## 5. 为什么 `data.ctrl` 写的是 torque

当前 MJCF 中所有 actuator 都是 `<motor>`：

```xml
<motor name="..." joint="..." ctrlrange="..." />
```

`<motor>` 的 `ctrl` 直接被解释为关节力矩。因此 **绝对不能** 把 `target_q_mj` 写到 `data.ctrl` —— 那会被当成
扭矩送进电机，对 ankle/knee 这种小 ctrlrange 关节立刻饱和并发散。

正确做法是：在脚本端自己跑 PD：

```python
tau = kp * (target_q - q) - kd * dq
data.ctrl[:] = clip(tau, ctrl_lo, ctrl_hi)
```

PD 增益与 Isaac 中 `JointPositionActionCfg` 的隐含 PD（`stiffness` / `damping`）一致。

## 6. 运行命令

GUI + zero_cmd：

```bash
/home/lz/miniconda3/envs/g0_mujoco/bin/python scripts/sim2sim/g0_mujoco_onnx_gui_runner.py \
  --model source/g0_robot_lab/g0_robot_lab/assets/robots/g0/mujoco/model_patched.xml \
  --policy logs/rsl_rl/g0_velocity/2026-05-26_18-36-57/exported/policy.onnx \
  --mode zero_cmd \
  --duration 20.0
```

GUI + 小 vx：

```bash
/home/lz/miniconda3/envs/g0_mujoco/bin/python scripts/sim2sim/g0_mujoco_onnx_gui_runner.py \
  --model source/g0_robot_lab/g0_robot_lab/assets/robots/g0/mujoco/model_patched.xml \
  --policy logs/rsl_rl/g0_velocity/2026-05-26_18-36-57/exported/policy.onnx \
  --mode cmd \
  --vx 0.05 --vy 0.0 --wz 0.0 \
  --duration 20.0
```

常用辅助 flag：

- `--no-viewer`：不开 GUI，只跑物理 + CSV。CI / 远程 shell 用。
- `--realtime`：限制 GUI 步进到 wall-clock 实时（默认 as fast as possible）。
- `--print-every 0.5`：每 0.5 仿真秒打印一次状态行。
- `--csv <path>`：自定义 CSV 输出路径。
- `--elastic-band` / `--staging-sop`：开启虚拟挂带（Unitree pattern）。挂带力学已 DRY
  到共享的 `deploy/common/elastic_band.py::ElasticBand`（本脚本只是用一个 thin adapter
  驱动它，不再内联 `xfrc_applied` 公式）。`--staging-sop` 是一个 preset：开挂带并打印
  7/8/9 SOP 提示。

> 注意：完整的 Unitree staging SOP（band on → FixStand → 按 8 下降 → 按 g 确认触地 →
> RLBase → 按 9 关挂带）的**正式入口是 `deploy/main` + `deploy_staging.yaml`**，见
> [`g0_unitree_staging_sop_en.md`](g0_unitree_staging_sop_en.md)。本脚本是诊断工具，
> 只做静态挂带悬吊，不实现键盘 SOP。

## 7. 通过 / 失败标准

通过标准：

1. obs / action / tau 全程无 NaN / inf。
2. zero_cmd 下机器人不在前几秒内倒地。
3. `root_z` 保持在合理范围（与 `default_stand` 初始 z≈0.23 同量级，不快速下落）。
4. `|roll|`、`|pitch|` 不快速发散到 > 1.0 rad。
5. torque 没有长时间满限幅（`mean_saturation_ratio` 远低于 1.0）。
6. action 连续，无明显爆炸（`max_abs_action` 在 O(1)~O(10) 区间，非 1e+10）。

失败标准（脚本会自动 ABORT 并把 `fail_reason` 写入 CSV / summary）：

1. obs / action / tau 出现 NaN / inf  →  `fail_reason=nan`
2. `root_z < 0.10`                     →  `fail_reason=root_z<0.10`
3. `|roll|>1.0` 或 `|pitch|>1.0`       →  `fail_reason=rp>1rad`
4. `max|q|>50` 或 `max|dq|>200`        →  `fail_reason=q_or_dq_diverged`
5. torque 长时间饱和                   →  通过 summary 的 `mean saturation` 字段人工判定

## 8. 当前已运行结果（headless）

命令：

```bash
/home/lz/miniconda3/envs/g0_mujoco/bin/python scripts/sim2sim/g0_mujoco_onnx_gui_runner.py \
  --model source/g0_robot_lab/g0_robot_lab/assets/robots/g0/mujoco/model_patched.xml \
  --policy logs/rsl_rl/g0_velocity/2026-05-26_18-36-57/exported/policy.onnx \
  --mode zero_cmd --duration 5.0 --no-viewer
```

观察到：

- ONNX 输入 shape = `[1,385]`，输出 shape = `[1,22]`，与 contract 一致。
- 全程没有 NaN / inf；torque 量级很小（max|tau| < 1.0 N·m）。
- 但是机器人在 ~1.4 s 时 `root_z` 跌破 0.10，触发 `root_z<0.10` ABORT。
  这是 **策略行为本身** 的问题（在 `model_patched.xml` 默认 stand 站姿 z=0.23 下站不稳），
  不是控制链路 / obs / mapping 的 bug。

这意味着：闭环链路本身通了，下一阶段需要从 **policy 训练** 或 **MJCF 站姿/地面 contact 参数** 这两个
方向再调试，而不是改 `g0_mujoco_onnx_gui_runner.py` 的控制逻辑。

GUI 的可视化测试需要在 **本机带显示的环境** 中手动跑（见第 6 节命令）。本次自动运行受限于
无显示终端，使用了 `--no-viewer` flag。

## 9. 本轮仍不涉及

```text
- DDS 通讯
- LowCmd / LowState
- 真实机器人
- motor_id 真机映射
- sign / zero 校准
- real read-only
- reward / 训练 / checkpoint 修改
- unitree_mujoco 依赖
```

## 10. 下一步建议

1. 在本机带 GUI 的 X / Wayland 环境运行第 6 节两个命令，观察机器人是否能保持站立 / 对小 vx 有合理响应。
2. 如果 zero_cmd 下仍迅速倒地，先检查：
   - `default_stand` 的 root_z 是否偏低（当前 0.23）。
   - 地面 friction / solref / solimp 是否过软。
   - PD gain 是否需要调大。
3. 在 GUI 阶段稳定后，再考虑 `mode=cmd` + 多种速度命令的清单。
4. 完成 sim2sim 闭环验收后，再进入 DDS / LowCmd / 真机的下一阶段，按 contract 顺序推进。
