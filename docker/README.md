# Dockerfile.hong_atom_inference 使用说明

> 说明：包含 Hong Atom 相关代码的推理镜像基于 `gr00t:n1d7_latest` 构建。

**前提**
- 已构建并可使用基础镜像 `gr00t:n1d7_latest`（详见 [README_ORIGIN.md](./README_ORIGIN.md)）。
- 本地/CI 可运行 `docker` 与必要构建工具。
- 有权限访问公司内部 Git 仓库。

**准备**
- 将 `hong_atom` 仓库 clone 到与 `Isaac-GR00T/` 目录平级的位置，例如：
```bash
git clone ssh://git@10.9.46.21:9022/ch_robot/alg_group/hong/hong_atom.git
```
- 进入 `hong_atom`，切换到 `dev` 分支；同时确保子模块 `3rdparty/hong_protos` 也在 `dev` 分支。

- 确认宿主机上 `./hong_atom/` 路径与 Dockerfile 中 `COPY` 路径一致（镜像会把该目录复制到容器临时路径）。

**关键注意**
- Hong Atom 的以下三个子目录依赖 Python 3.10.x：`3rdparty/hong_protos`、`python/hong_atom`、`python/hong_atom_bridge`。
- GR00T 推理环境要求 Python 3.10.x；请在构建或镜像中确保上述依赖与目标 Python 版本兼容。此处仅作提示，具体按团队流程在 Dockerfile 或基础镜像中调整 Python 版本或使用合适的镜像标签。
- Dockerfile 中示例复制命令（请本地确认路径）：
```dockerfile
COPY ./hong_atom/ /tmp/hong_atom/
```

**构建镜像**
- Dockerfile 路径：`Isaac-GR00T/docker/Dockerfile.hong_atom_inference`。
- 构建示例（请按实际标签与路径替换）：
```bash
docker build -t gr00t-n1d7-infer:v1.0 Isaac-GR00T/docker/Dockerfile.hong_atom_inference
```

**使用与参考**
- 容器运行与部署参考：`../g1_inspire_deploy/README.md`。
- 若遇依赖或版本冲突，优先检查：复制路径是否正确、`hong_atom` 子模块分支是否一致、目标镜像的 Python 版本设置。

**简要故障排查**
- 找不到文件/构建失败：确认宿主机中 `hong_atom` 的相对位置与 Dockerfile 的 `COPY` 路径一致。
- 运行时报 Python 版本相关错误：检查容器内 Python 版本是否为 3.10.x，确认 `3rdparty/hong_protos`、`python/hong_atom`、`python/hong_atom_bridge` 等是否与该版本兼容。
- 子模块不在正确分支或缺失文件：检查 `hong_atom` 的子模块是否已正确初始化与更新（`git submodule update --init --recursive`）。
