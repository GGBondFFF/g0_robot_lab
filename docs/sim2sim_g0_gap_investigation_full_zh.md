# G0 sim2sim(IsaacLab → MuJoCo)Gap 完整排查记录

> 目标:G0 速度策略在 IsaacLab 训练里能站/走,放进 MuJoCo(sim2sim)却 **~1.4s 必倒**。本文记录从"模型加载不了"一直到"站立+直线行走 gap 被 cover"的完整过程:每一步的现象、数据对比、以及依据哪个数据做了什么修改。
>
> 调查时间:2026-06-03 ~ 2026-06-04。运行环境:MuJoCo 侧用 conda env **`g0_mujoco`**(`/home/lz/miniconda3/envs/g0_mujoco/bin/python`),IsaacLab 侧用 `g0_isaaclab`。

---

## 0. 结论先行(TL;DR)

- **根因**:不是接线/参数 bug,而是 **sim2sim 鲁棒性 gap**。链条:
  > MuJoCo 与 Isaac 之间残余的求解器/接触动力学差 → 产生很小的 **base_ang_vel(基座角速度)偏差** → 这个策略对它**过度敏感**(把 ~0.1 量级的 obs 残差放大成 ~1 量级的动作误差,约 10×)→ 误差经 **last_action 观测项正反馈**自我强化 → ~1.4s 失衡倒地。
- **承重通道 = base_ang_vel;放大器 = last_action**(消融实验证明:把这两项任一"喂回 Isaac 真值",机器人就站住了)。
- **修复**:
  1. 把 MuJoCo 的 PD 从"每个 policy step 算一次力矩并冻结整个 decimation 窗口(zero-order-hold)"改成 **per-substep(每个仿真子步重算力矩)** —— 这一项让"零动作纯 PD 站立"从"立刻倒"变成"稳站 10s+"。
  2. **域随机化(DR)**:把训练侧 base_ang_vel 观测噪声从 **±0.2 rad/s 拉到 ±1.0**(并加 ±1.0 角速度 push、±20% 执行器增益随机),分阶段训练(先 bootstrap 再硬化微调)。
- **验证(MuJoCo 实测,倒地时间)**:原始 **1.4s** → ±0.8 DR **4.0s** → **±1.0 DR 稳站**。±1.0 策略:零指令稳站 30s,vx=0.3 前进行走 15s 不倒,GUI 实测前进 21s 不倒,力矩不饱和。
- **遗留**:转向(yaw)弱(慢漂),因为 ang_vel 命令课程一直卡在地板值 0.1(按约束未改晋级判据)。不影响站立和直线行走。

---

## 1. 背景、工具与契约

### 1.1 策略与观测契约
- 策略:`logs/rsl_rl/g0_velocity/2026-05-26_18-36-57/exported/policy.onnx`,输入 `obs [1,385]`,输出 `actions [1,22]`,纯 MLP(385→512→256→128→22,ELU,**无归一化器**,`empirical_normalization=False`)。
- 单步观测 77 维,7 个 term(scale 见括号):
  `base_ang_vel(×0.2)` 3 + `projected_gravity(×1)` 3 + `velocity_commands(×1)` 3 + `joint_pos_rel(×1)` 22 + `joint_vel_rel(×0.05)` 22 + `last_action(×1)` 22 + `gait_phase(×1)` 2。
- 历史:`history_length=5`,**per-term、oldest-first**(layout C),77×5 = 385。
- 三套关节顺序:**SDK**(策略 obs/action)、**MJ**(MuJoCo actuator/motor_id)、**Isaac**(IsaacLab articulation,joint_pos/vel term 用)。
- 控制:`step_dt = decimation(4) × sim.dt(0.005) = 0.02s`(50Hz);`gait period = 0.8`;`ACTION_SCALE = 0.12`;`target_q = default + 0.12·action`。
- 动力学频率(训练):`sim.dt = 0.005`(200Hz)。

### 1.2 新建的 runner
`scripts/sim2sim.py`(humanoid-gym 风格:`Sim2simCfg / get_obs / pd_control / run_mujoco`),接 385 维 ONNX 契约。关键参数:`--no-viewer`(headless)、`--realtime`(GUI 1×)、`--no-abort`(摔了也继续)、`--pd-zero-order-hold`(遗留开关,默认 per-substep)、`--duration<=0`(无限运行)。

### 1.3 关键 dump
`logs/sim2sim/g0_isaac_torque_dump.npz`:Isaac 一段成功站立轨迹(150 步/3s),含完整初始状态、`traj_action`(SDK 顺序)、`traj_joint_pos`(Isaac 轨迹)、`traj_obs_385`、`traj_applied_torque/computed_torque`。Isaac 全程 root_height ≈ 0.2345(站住)。

---

## 2. 第一阶段:让重新生成的 MJCF 能加载

重生成的 `assets/robots/g0/mjcf/g0_mujoco.xml` 一开始 **MuJoCo 根本加载不了**,逐个修:

| # | 现象(报错) | 原因 | 修改 |
|---|---|---|---|
| 1 | `XML_ERROR_PARSING_ATTRIBUTE line 61` | 每个驱动关节有**重复的 `range` 属性**(`range="-1.2 1.2" … actuatorfrclimited="true" range="-2 2"`),XML 非法 | 第二个 `range` → `actuatorfrcrange`(它本就是配合 `actuatorfrclimited` 的力矩限幅) |
| 2 | `unrecognized attribute: 'sensornoise'` | `<flag sensornoise="enable">` 当前 MuJoCo 不识别 | 删除该属性,保留 `frictionloss="enable"` |
| 3 | `material 'matplane' not found in geom 0` | ground 引用了未定义材质 | `<asset>` 里补 checker 纹理 + `matplane` 材质 |

修完加载成功:`nu=22, nq=29, nv=28`,actuator(motor_id)0–21 的关节顺序与 MJ 顺序一致。

---

## 3. 第二阶段:物理参数"单变量隔离"(全部 headless,确定性可复现)

> 方法:逐项把 MuJoCo 参数对齐训练/正确值,**每一项都让模型更正确、站立段更干净,但没有一个能阻止 ~1.4s 倒地。** 这一阶段最后逼出真正的根因(第 3.5 节)。

### 3.1 ctrlrange / 力矩上限 —— 一致,排除
- MJCF:标准舵机 `ctrlrange ±0.5`,直角舵机(elbow/knee/ankle_pitch ×2)`±0.5833`。
- `g0_actuators.py`:`STANDARD_SERVO_RATED_TORQUE=0.5`,`RIGHT_ANGLE=0.5×7/6=0.58333`。
- `g0.py`:Isaac `effort_limit_sim` = 同样的 0.5 / 0.5833。
- **结论**:MuJoCo ctrlrange == Isaac effort_limit == 额定力矩,**完全一致,精确到 10 位**。不是成因。(纠正了我中途"可能是占位值"的猜测。)

### 3.2 自碰撞 —— 真 bug,但非主因
实测默认站姿接触(root_z=0.23):

| 接触 | 穿透 | 性质 |
|---|---|---|
| 双脚 ↔ 地面(3 点) | 2.5mm | 脚-地,轻微 |
| **base_link ↔ torso_link** | 1.6mm | **自碰撞** |

- 双脚"恰好触地"在 root_z≈**0.2325**(所以 0.23 只陷 2.5mm,基本是训练设计的"脚微接触")。
- base↔torso 自碰撞与高度无关(抬到 0.30m 还在)。**Isaac 侧 `g0.py:228` 是 `enabled_self_collisions=False`**(关自碰撞),MuJoCo 默认 `contype=1 conaffinity=15` 开着 → 凭空多了一个训练时不存在的接触力,且 `solref='0.001 2'` 很硬,从第 0 步就踹 base。
- **依据"Isaac 关自碰撞"做的修改**:用碰撞位掩码让机器人只和地面碰、彼此不碰:
  - default geom:`contype="1" conaffinity="15"` → `contype="2" conaffinity="1"`
  - ground:`conaffinity='15'` → `contype="1" conaffinity="2"`
  - 判定 `(contype_A & conaffinity_B)|(contype_B & conaffinity_A)`:机器人间 `(2&1)=0` 不碰,机器人-地面 `(2&2)≠0` 照碰。验证:默认姿态 ncon 从 4→0(只剩脚降到地面时的接触),无机器人-机器人接触。
- **效果(headless,自碰撞 OFF)**:仍 ~1s 前倾倒地、躺平不起;倒地 `|act|` 从 15–32 降到 8–12(没那么暴力)。**不是主因。** （另:用户提醒"GUI 里看似恢复"是按 Backspace 复位造成的,不是策略自愈。)

### 3.3 armature —— 偏高 5–7 倍,修正后小幅改善
- MJCF:`<default class="joint_param"> armature="0.01"`(22 关节统一,占位值)。
- 正确值(`g0_actuators.py` / `patch_g0_mjcf_for_sim2sim.py` / Isaac `g0.py` 三处一致):标准 **0.0015**,直角 **0.0020417**(=0.0015×(7/6)²)。
- **修改**:default 改 0.0015,6 个直角舵机关节加 `armature="0.002041666667"`。验证逐关节正确。
- **效果**:起步 0–0.5s 明显更干净(`|act|` 1.3–2,接近理想站立);首次 root_z<0.10 从 ~1.4s 推到 ~1.6s(+0.4s)。仍倒。

### 3.4 damping / frictionloss —— 对齐训练反而更早倒(诊断信号)
- MJCF:`damping=0.01, frictionloss=0.01`(占位);patch:`2e-4 / 1e-3`;Isaac:被动阻尼/摩擦 **= 0**(只有 PD kd)。`g0_actuators.py` 里 `SERVO_DAMPING=0.1/FRICTION=0.05` 是 dead constant,无人引用。
- **修改**:default 改 `damping=0.0002, frictionloss=0.001`。
- **效果**:首次倒地从 ~1.68s **提前到 ~1.44s(早 0.24s)**。→ 原来的 0.01 是在"人为撑住"机器人;降到训练量级后底层不稳定更早暴露。**说明摔倒主因不是关节阻尼/摩擦**(它们之前在帮倒忙地稳)。按保真度保留了小值。

### 3.5 ★ PD 更新率 + 零动作站立 —— 锁定根因之一
**实验**:不接策略,只用 PD 把关节维持到默认站姿(零动作),看能否站住。

| PD 方案 | 结果(10s) |
|---|---|
| **zero-order-hold**(每 policy step 算一次力矩,冻结 10 个子步=脚本原默认) | **立刻面朝下倒**:root_z 0.040,pitch −1.54,`\|tau\|` 顶满 0.583 |
| **per-substep**(每个 500Hz 子步用当前 q/dq 重算) | **稳如磐石**:root_z 0.231 十秒不动,rpy≈0,`\|tau\|` 仅 **0.13** |

- **机理**:zero-order-hold 把力矩冻结 20ms,阻尼项 `kd·dq` 用的是窗口开头的旧速度,关节动起来后阻尼跟不上 → 振荡发散 → 饱和 → 倒。per-substep 每子步用当前速度反向制动 → 稳定、力矩极小。这正是 **隐式(Isaac)vs 显式-冻结(原脚本)PD** 的本质差异。
- **纠正一个早期错误说法**:zero-order-hold **不**贴合 unitree bridge —— unitree 冻结的是**指令(q_des/kp/kd)**,电机侧每个仿真步重算力矩;真机板载 PD 也高频重算。**per-substep 同时贴合 Isaac 隐式 PD、unitree bridge、真机**。
- **修改**:`sim2sim.py` 默认改为 **per-substep**,`--pd-zero-order-hold` 作为遗留对照开关。

### 3.6 物理步 dt —— 不是 bav 偏差的成因
闭环从 Isaac 初值起跑(下文工具),比较不同 dt:

| dt | decimation | 首次倒地步 | early bav 偏差 @[5,10,20] |
|---|---|---|---|
| 0.002(基线) | 10 | 76 | 0.175, 0.176, 0.146 |
| 0.005(对齐 Isaac) | 4 | 78 | 0.172, 0.172, 0.139 |
| 0.0025 | 8 | 101 | 0.173, 0.172, 0.154 |

- **结论**:三种 dt 下早期 base_ang_vel 偏差几乎一样,倒地时间基本不变。**dt 不是 base_ang_vel 偏差的来源。**(注意:solref 时间常数须 ≥ 2×dt。)

### 3.7 solver / solref —— 接触过硬,放松后站立段最干净但仍倒
- `<default> solref='0.001 2'`:时间常数 0.001s < 2×timestep(0.004),违反 MuJoCo 稳定准则 → 接触抖动。
- **修改**:`solref='0.001 2'` → `'0.02 1'`(geom + equality 两处)。
- **效果**:站立段最干净(root_z 稳在 0.23 直到 ~1.0s,`|act|` 更低),仍 ~1.44s 倒。
- 附:给 vx=0.2 小速度("让 gait 时钟有意义"的尝试)→ **反而更早倒 ~0.96s**(策略一收到前进指令就迈步、立刻失衡)。

### 3.8 阶段小结
关节级物理参数(ctrlrange / 自碰撞 / armature / damping / frictionloss)+ PD 更新率 + dt + solver,**逐项排查后都不是"摔倒"的决定性单因素**;唯一压倒性的发现是 **zero-order-hold PD 本身不稳定**(已修为 per-substep)。但即便 per-substep,闭环接策略仍 ~1.4s 倒 → 把矛头指向"闭环"。

---

## 4. 第三阶段:契约数值对账(排除"接线错")

### 4.1 obs 构造对账(Phase-B,对新 MJCF)
用现成的 Isaac dump,把 Isaac reset 状态写进 MuJoCo,重建 77 维 obs 逐段比:

| 段 | max\|diff\| |
|---|---|
| base_ang_vel | 0.000e+00 |
| projected_gravity | 4.663e-15 |
| velocity_commands | 0 |
| joint_pos_rel | 0 |
| joint_vel_rel | 0 |
| last_action | 0 |
| gait_phase | 0 |
| **整体** | **PASS(≤1e-5,实际浮点级)** |

→ **obs 构造、关节顺序映射全部正确。**

### 4.2 一个常见误解:MuJoCo 不加 IMU sensor 能拿到这些观测吗?
**能。** base_ang_vel 直接读 `data.qvel[3:6]`(free joint 后 3 维 = 体坐标系角速度)或 `mj_objectVelocity(...,flg_local=1)`;projected_gravity 用 root 四元数把世界重力转体坐标系;关节量读 `data.qpos/qvel`。**都不需要 `<sensor>`。** 对账已证明数值正确。别的项目加 IMU site/sensor 是为 **sim2real**(对齐真机 IMU 安装位姿、走 SDK 读取路径),与 sim2sim 倒下无关。

### 4.3 增益核对 —— 与训练完全一致
checkpoint `params/env.yaml` 的 actuator stiffness/damping/armature/effort 与脚本用的**一字不差**(kp:hip_pitch/roll 4、hip_yaw 3、ankle_roll 4.5、knee/ankle_pitch 4、waist 2、shoulder 1.5、elbow 2;kd 0.06–0.26;armature 0.0015/0.0020417;effort 0.5/0.5833)。**无增益 bug。**(纠正了我中途"残差源于增益不准"的错误判断。)

---

## 5. 第四阶段:开环动作 replay —— 证明开环动力学是对的

**思路**:把 Isaac 录制的动作序列原样喂进 MuJoCo(开环,不看 MuJoCo 自己的 obs),从 Isaac 初值起跑,per-substep,看能否跟出 Isaac 轨迹。

### 5.1 一个坑:动作顺序
- 第一次用错(把 dump 的 `traj_action` 当 Isaac 顺序)→ 接错关节 → 假"MuJoCo 倒了"。
- step-0 力矩与 Isaac `computed_torque` 的相关性:**Isaac 顺序 corr = −0.45,SDK 顺序 corr = +0.81** → **action 是 SDK 顺序**(策略原生输出),`sim2sim.py` 本来就按 SDK 处理,是对的。

### 5.2 修正后的 replay 结果(SDK 顺序 + per-substep + Isaac 初值)

| step | t | mj_rz | is_rz | d_rz | d_roll | d_pitch | max_d_q |
|---|---|---|---|---|---|---|---|
| 25 | 0.5 | 0.233 | 0.233 | 0.000 | 0.011 | 0.010 | 0.009 |
| 75 | 1.5 | 0.233 | 0.234 | 0.001 | 0.032 | 0.012 | 0.008 |
| 150 | 3.0 | 0.233 | 0.235 | 0.002 | 0.041 | 0.063 | 0.029 |

- **轨迹最大偏差**:d_rz 0.003、d_roll 0.097、d_pitch 0.063、max_d_q 0.092。**MuJoCo 跟住 Isaac、稳站 3 秒。**
- → **开环 body/接触/执行器动力学基本正确;gap 在闭环反馈。**
- 附:`computed_torque` 即使用正确增益/顺序也重建不出(max 误差 0.6)→ 那一列大概率是 PhysX/TGS 求解出的关节广义力(含约束/耦合),不是朴素 PD 输出,属 dump 语义,不是 bug。

---

## 6. 第五阶段:闭环 obs-diff + 通道消融 —— 找到承重通道

### 6.1 闭环逐步对比 obs(从 Isaac 初值,外生 cmd/gait 强制用 Isaac 值)

| step | root_z | d_pitch | **d_act** | bav | jpos | jvel | **lastact** |
|---|---|---|---|---|---|---|---|
| 0 | 0.230 | 0.000 | **0.000** | 0 | 0 | 0 | 0 |
| 10 | 0.238 | 0.009 | **1.18** | 0.18 | 0.16 | 0.08 | **1.26** |
| 30 | 0.235 | 0.112 | **3.58** | 0.56 | 0.20 | 0.34 | **4.49** |
| 70 | 0.202 | **1.23** | 9.40 | 1.09 | 0.71 | 0.51 | 9.15 |
| 80 | 0.064 | 倒地 | 45.5 | | | | |

- step 0 动作零误差(管线精确);**step 10 物理状态还几乎=Isaac,动作已偏 1.18** → 策略把 ~0.1 的 obs 残差放大成 ~1 的动作(~10×),经 **last_action** 回灌自我强化 → ~70 步(1.4s)真倒。

### 6.2 通道消融(把单个 obs 项强制喂 Isaac 录制值)

| 强制项 | 首次倒地步 | final root_z |
|---|---|---|
| baseline(全 MuJoCo) | 76 | 0.053 倒 |
| **base_ang_vel ← Isaac** | **没倒,站住** | 0.237 |
| **last_action ← Isaac** | **没倒,站住** | 0.230 |
| bav + lastact | 站住 | 0.234 |
| jpos + jvel ← Isaac | **69 仍倒** | 0.031 |
| 全部状态项 ← Isaac | 站住 | 0.233 |

- **承重物理通道 = base_ang_vel;放大器 = last_action 正反馈;关节项不是主因。** 堵住种子(bav)或放大器(lastact)任一即可。

### 6.3 量化 base_ang_vel 偏差(用开环 replay,纯动力学、机器人站住、无反馈污染)

| 统计(逐轴绝对偏差) | 值(rad/s) |
|---|---|
| 中位数 p50 | 0.17 |
| 均值 | 0.25 |
| **p95** | **0.79** |
| max | 1.24 |
| **当前训练噪声带** | **±0.2** |

→ 现有 base_ang_vel 噪声 ±0.2 **只覆盖了中位数**,p95(0.79)/max(1.24)的尾部完全没 cover。

---

## 7. 总根因(收敛结论)

> 所有**接线/参数/增益/顺序/开环动力学都正确**。闭环倒下是 **sim2sim 鲁棒性 gap**:残余求解器/接触动力学差产生很小的 base_ang_vel 偏差(实测 p95 0.79、max 1.24 rad/s),**策略对它过度敏感**(~10× 放大),经 last_action 正反馈滚大 → 必倒。这解释了"开环能站、闭环不能站"。
>
> 策略和 last_action 机制都改不了 → 唯一杠杆是**降低策略对 base_ang_vel 偏差的敏感度 = 域随机化(DR)**。

---

## 8. 第六阶段:DR 方案与分阶段训练

### 8.1 现有 DR 分析(`velocity_env_cfg.py`)
- **obs 噪声**(加在 scale 前的原始量上):base_ang_vel **±0.2**、proj_grav ±0.05、jpos ±0.01、jvel ±1.5。
- **events**:摩擦 (0.3,1.0)、torso 质量 (−0.3,1.0)、push **仅 x/y ±0.5** 每 5s;**无执行器随机化、无角速度 push**。
- 对照 gap:base_ang_vel 噪声 ±0.2 比实测 p95(0.79)小 ~4 倍;承重通道训练不足。

### 8.2 三项 DR 改动(对准诊断)
1. **base_ang_vel 噪声 ±0.2 → ±0.8**(PolicyCfg,第 ~259 行)—— 覆盖 p95。
2. **push 加 roll/pitch/yaw ±0.8**(EventCfg)—— 训练对**持续性**角速度偏置的恢复(iid 噪声教不了)。
3. **新增 `randomize_actuator_gains`**(EventCfg,startup,stiffness/damping ×(0.8,1.2))—— 覆盖隐式↔显式 PD"种子"。
- obs 385 布局/scale 未动 → 仍与 sim2sim.py 部署兼容。

### 8.3 一次性全加 → 训练卡死
从零训练 870 it 时:`Curriculum/lin_vel_cmd_levels` 和 `ang_vel` **一直卡在地板 0.1**;`base_height` 终止 **0.945**(平均 ~7s 就掉高度);mean_reward 平 ~5.1、episode_length ~350,全程零进步。
- **诊断**:重 DR 从零训练 → 连站都站不稳 → 94% 掉高度终止 → 课程晋级门槛永不满足 → 卡在 0.1 → 学不会走。(原码注释也提醒过 push "Enable only after standing/walking stable"。)

### 8.4 分阶段机制 `DR_STAGE`(不改晋级判据)
- 顶部常量 `DR_STAGE`;`G0RobotLabEnvCfg.__post_init__` 里 `if DR_STAGE >= 2:` 施加硬化 DR。
- **`DR_STAGE=1`**(仓库默认):bootstrap —— base_ang_vel 噪声 ±0.2、仅线性 push、执行器增益 no-op(scale 1.0)。
- **`DR_STAGE=2`**:硬化 DR(微调用,不从零训)。
- **晋级判据一行未动。**

### 8.5 Stage-1(从零,15000 it,run `2026-06-03_18-34-33`)—— 成功

| 指标 | it1k | it8k | 末(15k) |
|---|---|---|---|
| **lin_vel 课程** | 0.1 | **1.0** | 1.0(it6542 到顶) |
| ang_vel 课程 | 0.1 | 0.1 | **0.1**(仍卡) |
| mean_reward | 5.2 | 27.8 | 31(峰 35@14873) |
| mean_episode_length | 360 | 952 | 928 |
| base_height 终止 | 0.954 | 0.120 | **0.072** |
| track_lin_vel_xy | 0.32 | 0.64 | 0.68 |

→ 直线行走+站立稳定训好(解卡、不摔、活满全程)。**遗留:ang_vel 课程卡 0.1(转向弱)。**

### 8.6 Stage-2 ±0.8(从 model_14900 续,run `2026-06-04_09-07-09`,到 it19899)
- env.yaml 确认 DR 生效(push 角速度 ±0.8、增益 (0.8,1.2)、噪声 ±0.8)。
- 续训重爬 lin 课程到 1.0(注:课程等级不存进 checkpoint,resume 会归零重爬),reward ~25、ep_len 904、base_height 0.16。
- **验收(per-substep)**:

| 测试 | 原策略 | **±0.8** |
|---|---|---|
| TEST1 默认姿态零指令(sim2sim.py) | 倒 ~1.4s | **倒 ~4.0s** |
| TEST2 Isaac 初值闭环 | 倒 step71(1.4s) | **倒 step200(4.0s)** |

→ **改善 ~3×,但仍倒,未 cover。** 原因:±0.8 只覆盖 p95(max 1.24),闭环里动作一偏 bav 就超 p95 进正反馈;且仅微调 5000 步、课程归零重爬。

### 8.7 Stage-3 ±1.0(从 model_19899 续,run `2026-06-04_10-39-27`,到 it29898)—— 成功
- **修改依据**:实测 bav max≈1.24,±0.8 只到 p95 → 把 base_ang_vel 噪声与角速度 push **±0.8 → ±1.0**(执行器增益不动)。
- 续训重爬 lin 课程到 1.0,reward ~23/峰 28,ep_len 904,base_height 0.16。
- **验收(per-substep,三策略对比)**:

| 策略 | TEST1 默认姿态零指令 | TEST2 Isaac 初值闭环(min_rz) |
|---|---|---|
| ORIG(05-26) | 倒 1.4s | 倒 step71(min 0.044) |
| ±0.8 | 倒 4.0s | 倒 step200(min 0.052) |
| **±1.0** | **稳站 10s** | **稳站 10s(min 0.228)** |

- 追加确认:**零指令稳站 30s**(root_z 0.23、`|act|` 1.4–2.8、`|tau|` 0.24–0.35 不饱和);**vx=0.3 前进行走 15s 不倒**;**GUI realtime 前进 21s 不倒**。是从容稳定控制,非勉强 survive。
- **进展曲线:1.4s(原始)→ 4.0s(±0.8)→ 稳站/行走(±1.0)** —— 直接证明杠杆就是"策略对 base_ang_vel 偏差的容忍度"。

---

## 9. 最终结论

- **sim2sim 的 MuJoCo gap(站立 + 直线行走)已 cover。** 可用策略:`logs/rsl_rl/g0_velocity/2026-06-04_10-39-27/exported/policy.onnx`。
- 两条修复缺一不可:
  1. **per-substep PD**(让纯 PD 静态站立成立,贴合 Isaac/真机);
  2. **DR 把 base_ang_vel 容忍度从 ±0.2 拉到 ±1.0**(分阶段:先 bootstrap 再硬化微调)。
- 仓库默认 `DR_STAGE=1`;复现鲁棒策略 = `DR_STAGE=2` 从 stage-1 checkpoint 续训 ±1.0。已提交分支 `feat/unitree-staging-sop`(commit `12bfe86`)。

## 10. 遗留与后续

- **转向(yaw)弱**:`ang_vel` 命令课程一直卡 0.1(慢漂,不摔,但 wz 指令跟踪差)。修复需动 ang 课程晋级判据(本次约束未改)。
- **sim2real**:届时再处理 IMU sensor 安装位姿对齐、读取路径等。
- **进一步硬化**:如需更大裕度,可继续加宽 DR / 缩小残余 bav 源头(接触 solimp/solref/condim、Newton 求解器)。

## 11. 复现命令速查

```bash
# 环境
PY=/home/lz/miniconda3/envs/g0_mujoco/bin/python      # MuJoCo / sim2sim
PYI=/home/lz/miniconda3/envs/g0_isaaclab/bin/python    # IsaacLab 训练

# sim2sim 验收(headless):站立
$PY scripts/sim2sim.py --policy <exported/policy.onnx> --init-height 0.233 \
    --no-viewer --duration 30 --no-abort
# sim2sim 行走
$PY scripts/sim2sim.py --policy <...> --mode cmd --vx 0.3 --init-height 0.233 \
    --no-viewer --duration 15 --no-abort
# GUI 实时
DISPLAY=:0 MUJOCO_GL=glfw $PY scripts/sim2sim.py --policy <...> --mode cmd --vx 0.3 \
    --init-height 0.233 --realtime --no-abort

# Stage-1 从零训练(velocity_env_cfg.py 保持 DR_STAGE=1)
$PYI scripts/rsl_rl/train.py --task G0-Velocity-v0 --num_envs 4096 \
    --max_iterations 15000 --headless

# Stage-2 硬化微调(先把 DR_STAGE 改成 2)
$PYI scripts/rsl_rl/train.py --task G0-Velocity-v0 --num_envs 4096 \
    --max_iterations 10000 --headless \
    --resume --load_run <stage1_run> --checkpoint model_14900.pt

# 导出 ONNX
$PYI scripts/rsl_rl/play.py --task G0-Velocity-v0 --num_envs 1 \
    --load_run <run> --checkpoint <model_N.pt> --headless
```

---

*本文由 sim2sim gap 排查全过程整理(2026-06-03/04)。数据均来自 headless 确定性运行,可复现。*
