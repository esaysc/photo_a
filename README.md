# IRVisionSys

## 平台简介

IRVisionSys 是一款基于 RuoYi-Vue 框架开发的红外与可见光图像融合系统。

本项目旨在提供一个集图像上传、多种融合算法调用、结果实时预览与分析于一体的 Web 应用平台。系统后端采用 Spring Boot 技术栈，前端基于 Vue 3 和 Element Plus 构建，实现了前后端分离的现代化架构。

## 主要功能

*   **图像管理**：支持红外与可见光图像的上传、预览和管理。
*   **算法融合**：集成多种经典的与基于深度学习的图像融合算法。
*   **效果预览**：实时展示融合前后的图像对比效果。
*   **参数调整**：支持用户对不同算法的参数进行在线调整以优化融合效果。
*   **结果分析**：提供客观的评价指标来量化融合图像的质量。

## 开发环境

*   **JDK**: `1.8`
*   **Node.js**: `v16.x` 或更高版本
*   **Maven**: `3.x`
*   **MySQL**: `5.7.x` 或更高版本
*   **Redis**

## 部署运行

1.  **后端启动**
    - `git clone` 克隆项目
    - 在 MySQL 中创建 `ry-vue` 数据库，并导入 `sql` 目录下的初始化脚本。
    - 打开 `ruoyi-admin/src/main/resources/application-druid.yml`，修改数据库连接信息。
    - 打开 `ruoyi-admin/src/main/resources/application.yml`，修改 Redis 连接信息。
    - 使用 IntelliJ IDEA 打开项目，运行 `RuoYiApplication.java` 启动后端服务。

2.  **前端启动**
    - 进入 `ruoyi-ui` 目录。
    - 执行 `npm install` 或 `yarn install` 安装依赖。
    - 执行 `npm run dev` 启动前端服务。
    - 浏览器访问 `http://localhost:80` (默认端口)。


