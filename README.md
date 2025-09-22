# Branched Flow Simulation (分支流模拟程序)

[![Language](https://img.shields.io/badge/Language-Python-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

一份用于模拟和可视化粒子在可演化随机势场中“分支流”现象的物理仿真程序。

---

## 目录

- [1. 项目简介](#1-项目简介)
- [2. 功能特性](#2-功能特性)
- [3. 环境配置](#3-环境配置)
  - [3.1. 硬件要求](#31-硬件要求)
  - [3.2. 软件依赖 (安装指南)](#32-软件依赖-安装指南)
- [4. 如何运行](#4-如何运行)
  - [4.1. 首次运行](#41-首次运行)
  - [4.2. 查看结果](#42-查看结果)
  - [4.3. 使用命令行参数](#43-使用命令行参数)
- [5. 程序配置与自定义](#5-程序配置与自定义)
  - [5.1. 理解核心配置文件 `config.py`](#51-理解核心配置文件-configpy)
  - [5.2. 更换物理模型 (高级)](#52-更换物理模型-高级)
- [6. 解读输出图像](#6-解读输出图像)
- [7. 常见问题 (FAQ)](#7-常见问题-faq)

---

### 1. 项目简介

本项目模拟了当一束粒子穿过一个不均匀的介质（势场）时发生的“分支流”现象。

**一个简单的比喻**：想象一下，光线穿过坑洼不平的玻璃，或者水流过布满卵石的河床。在特定的条件下，粒子（或波）的能量不会均匀散开，而是会汇聚成类似树枝分叉状的高强度细丝。本程序就是用来模拟这一物理过程的。

程序的核心特点是**粒子与场的相互作用**：
1.  场（势场/力场）决定了粒子的运动轨迹。
2.  粒子在运动过程中，反过来会修改和“雕刻”它们所穿过的场。
3.  随着一批批粒子不断穿过，这个系统会逐渐演化，形成稳定或动态的结构。

### 2. 功能特性

- **双后端支持**：可选择在 `GPU` (通过Numba和CuPy) 或 `CPU` (通过NumPy和SciPy) 上运行，充分利用硬件加速。
- **高度模块化**：物理模型（势场、相互作用、粒子分布）、计算核心与分析可视化完全分离，易于扩展。
- **参数化配置**：所有模拟参数都集中在 `config.py` 文件中，无需修改核心代码即可调整模拟的方方面面。
- **丰富的可视化**：自动生成多种分析图像，包括：
    - 每个批次的详细综合报告（力场、轨迹、相空间、密度）。
    - 整个模拟过程的统计演化图。
    - 最终形成的连续束流快照。
- **数据导出**：可选择将详细的模拟数据（如每个快照的力场和粒子轨迹）导出为 `.npz` 文件，以便进行后续的离线分析。

### 3. 环境配置

#### 3.1. 硬件要求

- **CPU模式**：任何现代计算机都可以运行。
- **GPU模式 (推荐)**：需要一块支持CUDA的NVIDIA显卡。

#### 3.2. 软件依赖 (安装指南)

建议使用 `conda` 创建一个独立的Python环境来管理这些包。

```bash
# 1. 创建一个新的conda环境 (推荐)
conda create -n branched_flow_env python=3.10
conda activate branched_flow_env

# 2. 安装基础科学计算包
pip install numpy scipy matplotlib
```

**接下来，根据您希望使用的模式进行安装：**

**选项A：仅使用 CPU 模式**

如果您没有NVIDIA显卡，或者不想配置GPU环境，这是最简单的方式。

```bash
# 安装Numba的CPU版本
pip install numba
```
安装完成后，请确保在 `config.py` 中设置 `'backend': 'cpu'`。

**选项B：使用 GPU 加速模式 (推荐)**

要启用GPU加速，您需要安装 `numba` 和 `cupy`。

```bash
# 1. 安装Numba (会自动检测CUDA)
pip install numba

# 2. 安装CuPy (关键步骤)
# CuPy的版本需要和您系统中安装的CUDA Toolkit版本对应。
# 请访问 CuPy 官网 ([https://cupy.dev/](https://cupy.dev/)) 获取最适合您系统的安装命令。
# 例如，如果您安装了CUDA 11.8, 安装命令为:
pip install cupy-cuda11x

# 如果您安装了CUDA 12.x, 安装命令为:
pip install cupy-cuda12x
```
安装完成后，在 `config.py` 中设置 `'backend': 'gpu'` 即可启用GPU加速。

---

### 4. 如何运行

#### 4.1. 首次运行

所有配置都已设置好默认值，您可以直接运行主程序。在项目的根目录（即 `main.py` 所在的目录）打开终端，执行以下命令：

```bash
python main.py
```

程序会开始运行，并在终端中打印出每个批次的模拟进度和统计信息。

#### 4.2. 查看结果

模拟完成后，在项目根目录下会自动生成两个文件夹：

- **`simulation_plots/`**: 里面包含了所有生成的 `.png` 格式的分析图像。
- **`exported_data/`**: 如果 `enable_export` 设为 `True`，这里会保存每个快照的详细数据。

#### 4.3. 使用命令行参数

您可以在运行时临时覆盖 `config.py` 中的任何参数，这对于快速实验非常方便。

**语法**: `python main.py --[参数名] [值]`

**示例**:
```bash
# 示例1：将模拟批次数改为50批
python main.py --num-batches 50

# 示例2：临时切换到CPU后端运行
python main.py --backend cpu

# 示例3：关闭数据导出功能
python main.py --no-enable-export
```

---

### 5. 程序配置与自定义

本程序的核心魅力在于其强大的可配置性。您几乎所有的想法都可以通过修改配置文件来实现。

#### 5.1. 理解核心配置文件 `config.py`

这个文件是整个模拟的“控制中心”。以下是一些最重要的参数说明：

- **高层配置**
    - `'backend'`: `gpu` 或 `cpu`，选择计算后端。
    - `'potential_config_name'`: 选择初始势场的类型，名字必须在 `physics/potentials.py` 中定义。
    - `'interaction_config_name'`: 选择粒子与场相互作用的模型。
    - `'distribution_config_name'`: 选择初始粒子的分布方式。
    - `'potential_update_rule'`: 力场更新规则，`accumulate` (累加) 或 `relax` (带衰减的累加)。

- **通用模拟参数**
    - `'num_particles'`: 每批次发射的粒子数量。
    - `'num_batches'`: 总共模拟的批次数。
    - `'x_end'`, `'y_end'`: 模拟区域的大小。
    - `'dx'`: 模拟步长，越小越精确，但计算越慢。

- **可视化开关**
    - `'use_log_scale_plots'`: `True` 或 `False`，决定密度图是否使用对数坐标。
    - `'plot_output_path'`: 指定保存图像的文件夹路径。

#### 5.2. 更换物理模型 (高级)

假设您想尝试一种新的初始势场，比如关联长度更长的随机场。得益于程序的模块化设计，您只需要两步：

**第一步：在 `physics/potentials.py` 中添加新配置**

打开该文件，在 `potential_configs` 字典中，复制一个现有的配置块并修改参数。例如，我们可以添加一个 `alpha` 值为 `6.0` 的新随机场配置：

```python
# physics/potentials.py

potential_configs = {
    'smooth_random': {
        'provider': 'fourier_synthesis',
        'args': {'alpha': 4.0, 'seed': 2025, 'amplitude': 0.1},
        'description': "一个标准的随机势能场。"
    },
    # --- 在这里添加您的新配置 ---
    'smooth_random_large_corr': {
        'provider': 'fourier_synthesis',
        'args': {'alpha': 6.0, 'seed': 2025, 'amplitude': 0.1}, # 将alpha改为6.0
        'description': "一个关联长度更长的随机势能场。"
    },
    # --------------------------
    'sinusoidal_default': { ... },
}
```

**第二步：在 `config.py` 中选用新配置**

现在，打开 `config.py`，将 `potential_config_name` 的值修改为您刚刚定义的新名字：

```python
# config.py

PARAMS = {
    # ...
    'potential_config_name': 'smooth_random_large_corr', # <-- 修改这里
    # ...
}
```
保存后再次运行 `main.py`，程序就会使用您定义的新物理条件进行模拟了！这个原则同样适用于 `interactions.py` 和 `distributions.py`。

---

### 6. 解读输出图像

- **`report_batch_XXXXX.png`**: 这是对特定批次的详细分析。
    - **左上**: 当前批次粒子感受到的背景力场。
    - **右上**: 粒子在该力场中的运动轨迹。
    - **左下**: 粒子在不同时刻的相空间分布图（位置 vs. 动量）。
    - **右下**: 该批次所有粒子轨迹的密度图，可以清晰地看到分支结构。
- **`summary_statistics.png`**: 展现了整个模拟过程中，各项统计数据（如总动能、平均力场强度、汇聚长度等）随批次数的演化趋势。
- **`beam_snapshot.png`**: 这是最终的、最重要的结果图之一。它将所有批次的粒子在某一瞬间的状态“冻结”并组合在一起，展示了连续粒子束形成的稳定分支流形态。

### 7. 常见问题 (FAQ)

- **Q: 终端出现 `NumbaPerformanceWarning: Grid size ... will likely result in GPU under-utilization`?**
- **A:** 这是一个来自Numba的性能提示，通常在模拟的某些阶段（如启动时）出现。它表示GPU的并行核心没有被完全占满，**这不是一个错误**，程序仍在正常运行，您可以忽略它。

- **Q: 我修改了 `config.py` 后程序报错 `SyntaxError`?**
- **A:** 这通常是因为小的语法错误，例如忘记在字典条目的末尾加逗号 `,`，或者在字符串后忘记加引号。请仔细检查您修改的那一行。

- **Q: 模拟速度很慢怎么办？**
- **A:**
    1.  优先确保您已正确配置并使用 `GPU` 后端。
    2.  尝试减小 `num_particles` (粒子数) 或增大 `dx` (步长)。
    3.  在 `config.py` 中，可以适当减小 `density_bins_x` 和 `density_bins_y` 的值，以降低分析和绘图的开销。

