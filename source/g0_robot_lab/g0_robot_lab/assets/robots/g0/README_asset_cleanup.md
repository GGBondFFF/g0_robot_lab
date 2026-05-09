# G0 URDF Asset Cleanup

## 原始资产存在的问题

- 顶层资产目录使用中文长目录名。
- 原始 URDF 的 `robot name` 使用中文长名称。
- 原始 mesh 引用使用 `package://` 中文 package 路径。
- 原始资产中存在 macOS 缓存文件，例如 `._*`。

## 整理后的目录结构

```text
g0/
├── urdf/g0.urdf
├── meshes/
├── inspect_g0_urdf.py
└── README_asset_cleanup.md
```

## URDF 修改内容

- `robot name` 已改成 `g0`。
- 所有 `mesh filename` 已改成 `../meshes/xxx.STL` 相对路径。
- link 和 joint 名原则上保持不变。
- `inertial`、`mass`、`inertia`、`origin`、`axis`、`limit` 数值原则上不改。
- mesh 文件名和 `.STL` 扩展名大小写保持原样。

## 检查命令

```bash
cd g0_cleaned_asset/g0
python inspect_g0_urdf.py
```

## 手动拷贝到项目中的命令示例

```bash
cp -r g0_cleaned_asset/g0/* /home/lz/g0_robot_lab/g0_robot_lab/source/g0_robot_lab/g0_robot_lab/assets/robots/g0/
```

## 拷贝后下一步计划

- 在 Isaac Sim / Isaac Lab 中将 `urdf/g0.urdf` 转换为 `usd/g0.usd`。
- 编写或更新 `assets/robots/g0/g0.py`。
- 在 `velocity_env_cfg.py` 中将 `CARTPOLE_CFG` 替换为 `G0_CFG`。
- 再做 `G0-Velocity-v0` smoke test。
