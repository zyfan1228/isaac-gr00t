# Isaac-GR00T-N1.7-DEV

本项目是基于 Isaac-GR00T 1.7 版本的再开发版本（基于 tag n1.7-release），主要是在 GR00T 1.7 版本的基础上，针对后训练和推理服务部署添加适配，包括但不限于：

- 新增适配 G1_Inspire 的 SFT 脚本配置和教程说明
- 新增 N1.7 模型基于 hong_atom 的容器化推理服务部署脚本，支持自训练模型在容器内部署 + G1 真机上的实机推理

> *N1.7 官方原 README 文档入口：[GR00T 1.7 官方 README](./README_ORIGIN.md)*

## 快速跳转

### 自定义本体的 SFT 脚本配置和教程说明： [NEW_ROBOT_SFT 教程](./examples/NEW_ROBOT_SFT.md)

### N1.7 模型基于 hong_atom 的容器化推理服务部署脚本： [g1_inspire_deploy](./g1_inspire_deploy/README.md)

### (部署推荐) N1.7 模型基于 vla_infer 仓库的容器化推理服务部署脚本，参考 ssh://git@10.9.46.21:9022/ch_robot/alg_group/robot-api/vla_infer.git 仓库
