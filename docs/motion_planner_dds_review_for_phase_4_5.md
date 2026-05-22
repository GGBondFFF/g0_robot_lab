# motion-planner DDS 审查报告（供 G0 Phase 4.5 参考）

- **审查对象**：`/home/lz/ws/motion-planner` + `/home/lz/ws/mbus`
- **审查方式**：只读代码审查（未改代码、未运行二进制、未连接硬件）
- **审查日期**：2026-05-21
- **关联 G0 文档**：
  - [dds_hardware_free_rehearsal_plan.md](./dds_hardware_free_rehearsal_plan.md)
  - [dds_motor_id_mapping_audit.md](./dds_motor_id_mapping_audit.md)
  - [dds_virtual_transport_contract.md](./dds_virtual_transport_contract.md)
  - [lowcmd_mapping_safety_audit.md](./lowcmd_mapping_safety_audit.md)

---

## 核心结论（先读）

这套方案是 **真实 CycloneDDS 中间件**（`mbus` 封装），**不是** G0 已完成的 in-process virtual bus / AF_UNIX+JSONL virtual bridge。

双终端分工是：**仿真端订阅高层 `robot_control`**，**发送端发布 `RobotControl::Control`**（机身速度/姿态/夹爪），**不直接发单关节 `MotorControl`**。

因此它更像「远程操作员 → 规划器 → MuJoCo PD」联调，**不等同于** G0 Phase 4「单关节 mapping/sign visual rehearsal」。

---

## 1. 两终端运行流程

| 终端 | 进程 | 作用 |
|------|------|------|
| **Terminal 1** | `motion_planner_sim` | 加载 MJCF、GLFW 可视化、跑 `HexaStateMachine`、**订阅** DDS `robot_control` |
| **Terminal 2** | `fake_control_agent` | **发布** DDS `robot_control`（TOML 或键盘） |

**通信机制**：CycloneDDS（UDP 多播，`Domain Id=0`），经 `mbus::MbusContext`。同一台机器上通常走 `lo`，**不是** socket 自定义协议、共享内存、文件或 JSONL。

**数据路径**：

```
fake_control_agent  --DDS Pub robot_control-->  CycloneDDS Domain 0
                                                      |
motion_planner_sim  <--DDS Sub robot_control--       |
       |
       v
HexaStateMachine / Walk -> IK -> MujocoExecutor -> mj_step
```

仿真端**不订阅** `mc/motor_control`；规划在进程内完成，再写入 `data_->ctrl[i]`。

---

## 2. DDS / 消息字段

### 2.1 `RobotControl::Control`（fake_control_agent → sim）

| 字段 | 有/无 | 说明 |
|------|--------|------|
| `motor_id` / `joint_id` | **无** | 仅机身/夹爪语义 |
| `q` / `dq` / `kp` / `kd` / `tau` | **无** | 非关节级命令 |
| `timestamp` / `sequence_id` | **无** | 无样本时效/序号 |
| `dry_run` / `virtual_only` / `sandbox` | **无** | 无沙箱标志 |

**子结构**（`mbus/idl/idl_robot_control.idl`）：

- `mode`: `Common::RobotMode`（发送端固定 `HEXAPOD`）
- `body.attitude`: roll/pitch/yaw（**弧度**，fake_control 由度转 rad）
- `body.linear_velocity`: x/y/z（**归一化 [-1,1]**，sim 内映射到 m/s）
- `body.angular_velocity`: x/y/z（yaw_rate 由 deg/s 转 rad/s；roll/pitch 为归一化轴）
- `body.height`: float（发送端常为 0）
- `left_gripper` / `right_gripper`（optional）：`is_valid`, attitude(rad), linear_velocity, `opening` [0,1]

**单位**：线速度为归一化；姿态 DDS 侧为 **rad**；fake_control TOML 里 body 姿态为 **度**；`hexa_config.toml` 将 [-1,1] 映射到 `max_vel`（默认 ±0.2 m/s）。

### 2.2 `MotorControl::Control`（`motion_planner` 真机路径）

| 字段 | 有/无 | 单位/备注 |
|------|--------|-----------|
| `timestamp_ns` | 有 | uint64，发送时用 **系统时钟** epoch ns |
| `sequence_id` | 有 | uint16，单调递增 |
| `motor_count` | 有 | ≤22 |
| `motors[i].pos` | 有 | **度**（DDSExecutor 从 rad 转换） |
| `motors[i].dq` | 有 | **度/s** |
| `motors[i].kp`, `kd` | 有 | 无量纲 |
| `motors[i].tau` | 有 | Nm（firmware torque） |
| `motor_id` | **无** | **纯数组下标 i** |

`MotorControl::State`：`sequence_id`, `motor_count`, `motors[]`, `imu`, `source_timestamp_ns`。

**仿真双终端路径不使用该 topic**；`DDSClient::get_motor_ctrl()` 恒返回 `nullopt`。

---

## 3. Topic / domain / transport

| 项目 | 值 |
|------|-----|
| **Domain ID** | `0`（`MbusContext(0)`） |
| **robot_control** | topic `robot_control`，QoS `mbus::RobotControl_Control`，fake=Pub，sim=Sub |
| **mc/motor_control** | Pub（真机 `motion_planner` + `DDSExecutor`） |
| **mc/motor_state** | Sub（真机读反馈） |
| **Transport** | CycloneDDS；`mbus_config.xml` 绑定 `lo` / `wlan0` / `eth0`，允许多播 |
| **配置路径** | 默认 `file:///etc/mbus/config/mbus_config.xml`（开发机需安装或改路径） |
| **Sandbox topic** | **无** |
| **硬件隔离** | **无** domain/topic/环境变量 隔离；同网段可发现真机 topic |

**能否连真机**：能。同 Domain 0 + 相同 topic 名即可互通；`fake_control_agent` 可驱动真机上的 `motion_planner`（若其在跑且订阅 `robot_control`）。

mbus demo 还用 `mbus/control`（与 motion-planner 的 `robot_control` **不一致**）。

---

## 4. 发送端逻辑（`fake_control_agent`）

- **构造**：TOML `[body]` / `[left_gripper]` / `[right_gripper]` → `RobotControl::Control`；或键盘 `--keyboard`（evdev，50Hz 可配）。
- **单关节命令**：**不支持**；无 per-joint DDS。
- **多关节命令**：**不支持**；仅机身 + 可选双夹爪。
- **默认姿态**：全零 body；gripper 默认 `enabled=false`（示例 toml 左爪 `enabled=true`）。
- **小 delta 测试**：无专用单关节 delta；可用 `vx=0.1` → 约 0.02 m/s（注释与 `max_vel=0.2` 一致）。
- **频率**：`[publish] rate_hz`（默认 50）、`loop_count`（-1 无限）；键盘模式 `sleep_until` 定周期。

---

## 5. 接收端逻辑（`motion_planner_sim`）

1. 每 sim tick：`dds_client->get_robot_ctrl()` → 有则更新 `cached_ctrl`（**无则沿用旧值**）。
2. `HexaControlInput::from_control_data()` 做 [-1,1]→物理量映射。
3. `HexaStateMachine::next_tick()` → `measure_legs_state_from_executor()`（MuJoCo `qpos/qvel`）→ 步态/IK → `sync_legs_state_to_executor()`。
4. **MujocoExecutor**：`τ = kp*(q_tgt-q) + kd*(dq_tgt-dq) + torque`，写入 `data_->ctrl[i]`；`max_torque` 限幅。
5. **target vs actual**：规划目标在 `LegChain`/joint 状态；实际来自 MuJoCo；**无**「DDS target buffer」层。
6. **错误关节日志**：无 wrong joint / spurious write 检测；可选诊断按腿汇总 pos/vel/torque 误差。

**关节顺序（关键）**：sim **故意不用** `exec_leg_sequence`，按 leg_id 排序 `0..5`（LF,LM,LR,RR,RM,RF）对齐 MJCF actuator。真机 `main.cpp` 用 `LF,RF,LM,LR,RM,RR`——**与 sim 不同**。

见 `motion-planner/cpp/src/main_sim.cpp` 注释（约 185–191 行）。

---

## 6. mapping / sign / zero

| 能力 | 状态 |
|------|------|
| motor_id → joint_name 表 | **无**（数组下标） |
| SDK index → motor_id | **无** |
| sign adapter（DDS 层） | **无**；几何 sign 在 `leg_chain.cpp`（`sign_lat`, `elbow_sign`）用于 IK/接触，非 DDS 审计 |
| zero offset（DDS） | **无** |
| 左右腿确认 | 腿 ID 枚举 LF..RF；无 automated L/R audit |
| visual sign audit | **无** JSONL/自动判定 |

---

## 7. 安全逻辑

| 机制 | 状态 |
|------|------|
| Emergency stop | 仅 SIGINT/SIGTERM；无 DDS e-stop |
| Stale command | **无**；`cached_ctrl` 永久有效直至新包 |
| sequence_id 单调性 | RobotControl **无**；MotorControl 有但 sim 路径不用 |
| max command age | **无** |
| 禁止真硬件 | **无** |
| 环境变量保护 | 仅 `CYCLONEDDS_URI`（指向 mbus XML）；无 `G0_*` / `VIRTUAL_ONLY` 类开关 |

MuJoCo：`--max-torque` 力矩钳位；body planner 有 `max_vel`/`max_acc` 等，属规划层非 DDS 沙箱。

---

## 8. 日志与验证

- **日志**：`mp::logger`；DDS 每 1000/3000 包打 info；fake_control 终端 UI；MujocoExecutor 可选 `[Diag]`。
- **JSONL / summary JSON**：**无**
- **自动 mapping/sign 判定**：**无**
- **测试**：`cpp/test/special_gait/` 为步态单元测试；**无** DDS/mapping 自动化测试。

---

## 9. 与 g0_robot_lab Phase 4.5 的关系

| 维度 | motion-planner | g0（Phase 1–4 描述） |
|------|----------------|----------------------|
| 中间件 | 直接 CycloneDDS | Phase 3A/3B virtual + Phase 4.5 拟真 DDS sandbox |
| 命令粒度 | 机身/夹爪高层 | Phase 4 单关节 q/kp/… |
| 隔离 | 弱（Domain 0 生产 topic） | hardware-free / 不发真 LowCmd |
| 审计 | 无 JSONL mapping audit | Visual + schema audit |

---

## A. 虚拟 DDS 联调相关文件清单

### motion-planner

| 路径 | 角色 |
|------|------|
| `cpp/src/main_sim.cpp` | 仿真接收端主程序 |
| `cpp/test/fake_control_agent/main.cpp` | DDS 发送端 |
| `cpp/test/fake_control_agent/fake_control_agent.toml` | 发送配置 |
| `cpp/test/fake_control_agent/CMakeLists.txt` | 构建 fake_control_agent |
| `cpp/include/motion_planner/dds_client.hpp` | DDS 客户端封装 |
| `cpp/src/dds_client.cpp` | topic 注册与读写 |
| `cpp/include/motion_planner/robot/mujoco_executor.hpp` | Sim 执行器 |
| `cpp/src/robot/mujoco_executor.cpp` | ctrl 写入与诊断 |
| `cpp/include/motion_planner/robot/dds_executor.hpp` | 真机 motor DDS（对照） |
| `cpp/src/robot/dds_executor.cpp` | rad↔deg、sequence_id |
| `cpp/src/hexa/control.cpp` | RobotControl → HexaControlInput |
| `cpp/src/main.cpp` | 真机规划器（对照 leg 序） |
| `config/hexa_config.toml` | body 速度/限幅 |
| `config/mujoco_pid_profile.toml` | 仿真 PID |

### mbus（IDL + 中间件）

| 路径 | 角色 |
|------|------|
| `mbus/idl/idl_robot_control.idl` | 高层控制 IDL |
| `mbus/idl/idl_motor_control.idl` | 关节 motor IDL |
| `mbus/idl/idl_common.idl` | 公共类型 |
| `mbus/cpp/wrapper/mbus_wrapper.hpp` | MbusContext、Topic |
| `mbus/install/etc/mbus/config/mbus_config.xml` | Domain 0 / 网卡 |
| `mbus/cpp/demo/control_dds_prot.cpp` | 独立 DDS demo（topic 名不同） |

### 工作区 DDS 栈（依赖）

`cyclonedds/`、`cyclonedds-cxx/`、`dds-install/`

---

## B. 两终端运行命令

```bash
# Terminal 1 — 仿真 + 可视化（在 motion-planner 根目录）
./cpp/build/motion_planner_sim --config ./config/hexa_config.toml

# 无界面
./cpp/build/motion_planner_sim --headless --dt 0.002 --config ./config/hexa_config.toml

# Terminal 2 — DDS 发送（在含 fake_control_agent 与 toml 的目录）
./cpp/build/test/fake_control_agent/fake_control_agent --config fake_control_agent.toml

# 键盘模式
./cpp/build/test/fake_control_agent/fake_control_agent --keyboard --config fake_control_agent.toml
```

**前提**：已构建；`/etc/mbus/config/` 或等效 `CYCLONEDDS_URI`；两进程同机 Domain 0。

---

## C. DDS message 字段表

### RobotControl::Control（双终端主路径）

| 字段 | 类型 | 单位/语义 |
|------|------|-----------|
| `mode` | enum | HEXAPOD/… |
| `body.attitude.roll/pitch/yaw` | float | **rad**（发送端由度转换） |
| `body.linear_velocity.x/y/z` | float | **归一化 [-1,1]** |
| `body.angular_velocity.x/y/z` | float | rad/s（yaw_rate 由 deg/s 转） |
| `body.height` | float | m（常用 0） |
| `left/right_gripper.*` | optional | `opening` [0,1]，vel 归一化，attitude rad |

### MotorControl::Control（真机路径，sim 双终端不用）

| 字段 | 类型 | 单位 |
|------|------|------|
| `timestamp_ns` | uint64 | ns |
| `sequence_id` | uint16 | — |
| `motor_count` | uint32 | — |
| `motors[i].pos` | float | **deg** |
| `motors[i].dq` | float | **deg/s** |
| `motors[i].kp/kd` | float | — |
| `motors[i].tau` | float | Nm |

---

## D. topic / domain / transport 表

| Topic | QoS 名 | Pub | Sub | 用于 |
|-------|--------|-----|-----|------|
| `robot_control` | `mbus::RobotControl_Control` | fake_control_agent | motion_planner_sim | **双终端联调** |
| `mc/motor_control` | `mbus::MotorControl_Control` | motion_planner (DDSExecutor) | （电机侧） | 真机 |
| `mc/motor_state` | `mbus::MotorControl_State` | （电机侧） | motion_planner | 真机反馈 |
| `mbus/control` | `mbus::RobotControl_Control` | mbus demo | mbus demo | 示例，**非** motion-planner |

| 参数 | 值 |
|------|-----|
| Domain ID | 0 |
| Transport | CycloneDDS multicast（lo/eth/wlan） |
| 虚拟总线 / JSONL | **无** |

---

## E. motor_id / joint mapping 逻辑总结

- **无 motor_id 字段**；`motors[i]` 与规划器 `sync_legs_state_to_executor()` 的 **向量顺序** 绑定。
- **真机顺序**：`LF, RF, LM, LR, RM, RR`（22 关节拼接）。
- **仿真顺序**：`LF, LM, LR, RR, RM, RF`（对齐 MJCF，**不可**与真机数组直接互换）。
- DDS 层 **无** joint_name / Isaac index 映射表。

**禁止**：将六足 motor 数组下标直接套到 G0；禁止假设 MuJoCo joint order = G0 Isaac joint order。

---

## F. sign / zero / unit 处理总结

| 层 | sign | zero | 单位 |
|----|------|------|------|
| RobotControl | 无 DDS sign 适配 | 无 | 归一化速度 + rad 姿态 |
| MotorControl 线缆 | 无 per-motor sign | 无 | pos/dq **度**，内部 **rad** |
| MuJoCo 执行 | 几何 `sign_lat`/`elbow_sign`（IK） | bootup record/play 有测量复位 | τ：Nm，q：rad |
| 与 G0/Isaac | **不可假设** joint order 或 motor_id 相同 | 需独立 G0 表 | G0 LowCmd 字段名/单位可能不同 |

---

## G. 安全边界总结

**已有**：进程信号退出；MuJoCo 力矩上限；body 规划速度/加速度限制；MotorControl `sequence_id`（仅真机发布）。

**缺失（对 G0 Phase 4.5 关键）**：

- sandbox domain/topic
- `virtual_only` / dry_run
- stale / age / seq 校验
- 禁止发布生产 `LowCmd` 的硬闸
- 单关节误写检测
- JSONL 审计
- 环境变量硬件锁

**风险**：开发机 Domain 0 可能与真机/其他 DDS 节点 **共域发现**；`cached_ctrl` 在无新包时 **持续执行旧命令**。

---

## H. 可迁移到 g0 Phase 4.5 的设计

1. **双进程解耦**：发送工具与仿真/规划接收端分离，便于 visual rehearsal。
2. **`mbus` 模式**：`MbusContext` + 强类型 topic + `take()` 非阻塞读。
3. **TOML/CLI 发布频率与循环次数**（`rate_hz`, `loop_count`）。
4. **归一化操作量 + 接收端 `map_range` 物理映射**（可改为 G0 的关节限幅）。
5. **Executor 抽象**（`BaseExecutor` / sim vs DDS）— 概念可映射到 Isaac Lab vs 真 DDS。
6. **MotorControl 帧**：`timestamp_ns` + `sequence_id` + 固定数组（G0 可换 IDL 但保留审计字段）。
7. **仿真侧显式文档化 joint 顺序**（`main_sim.cpp` 注释）— G0 应对 Isaac articulation 做同等显式表。

---

## I. 不能迁移或必须重写的设计

1. **整条 RobotControl 高层协议** → G0 Phase 4/4.5 需 **per-joint / LowCmd 级** schema。
2. **无 virtual DDS / JSONL bridge** → 不能替代 G0 Phase 3B；Phase 4.5 若「真 DDS sandbox」需 **独立 domain + topic 前缀**。
3. **数组下标 = motor 语义** → G0 必须 **motor_id / joint_name 显式表 + sign adapter**。
4. **无单关节 visual audit** → G0 Phase 4 核心能力需新建。
5. **hexapod leg 序 vs G0 人形/四足** → 禁止照搬 motor 索引。
6. **默认 `/etc/mbus` + Domain 0** → G0 需 **sandbox domain**（如 230）与 **非生产 topic 名**。
7. **stale ctrl 缓存策略** → G0 应 **超时归零 / 安全姿态**，不可沿用。
8. **Isaac Lab**：无 Isaac 集成；迁移的是 **流程与隔离思想**，不是 executors/MuJoCo 代码。

---

## J. 建议的 g0_robot_lab Phase 4.5 实现顺序

1. **沙箱契约**：独立 `CYCLONEDDS_DOMAIN_ID` + 仅 `rt/lowcmd_sandbox` / `rt/lowstate_sandbox`（名称按 G0 IDL）；环境变量 `G0_DDS_SANDBOX=1` 硬拒绝生产 topic。
2. **复用 Phase 3 packet builder**：在 **真 CycloneDDS** 上发 sandbox 包（非 3B JSONL，除非作 fallback）。
3. **单关节发送 CLI**（对标 G0 Phase 4）：一次只动 `motor_id=k`，小 Δq，带 `sequence_id`。
4. **Isaac / 仿真接收端**：显式 `motor_id → joint_name → dof_index` 表；记录 command vs measured JSONL。
5. **stale + seq + max_age**：接收端超时置零/冻结；单调 `sequence_id` 告警。
6. **Visual sign audit**：自动 + 人工 checklist（左/右、屈伸方向）。
7. **双终端文档化**：Terminal1 Isaac/sim，Terminal2 sandbox publisher（对齐 motion-planner 运维体验，但 **协议用 G0 LowCmd**）。
8. **最后** 才考虑与 Phase 3B bridge 互通或 policy-in-the-loop（明确在 mapping 签字之后）。

---

## 审查限制（复述）

- 未修改 motion-planner / mbus 代码
- 未连接真实硬件
- 未运行真实 motor command
- 未发布真实 LowCmd topic
- 未假设 MuJoCo joint order = G0 Isaac joint order
- 未将六足 motor 索引套用到 G0

---

## 与 special_gait / Phase B 的区分

`single_front_claw_gait` 的 Phase B stop-reposition 属于步态状态机内部逻辑，**与 DDS 双终端联调无关**。DDS 联调验证的是远程 `robot_control` → 行走/夹爪，不是单关节 motor mapping rehearsal。
