# NNLBL: Neural Network Line-By-Line Absorption Calculator——基于神经网络的高速逐线积分气体吸收截面计算程序

## 📖 项目简介 (Introduction)  

NNLBL 是一个用于快速计算大气气体分子吸收截面（Absorption Cross-Section）的工具。它利用神经网络（NN）替代传统的 Voigt 线型卷积计算，并针对GPU进行优化，显著提升了计算速度，同时也可以在cpu上运行，算法会优先使用GPU加速。

本项目核心特性包括：  

- 高精度：以 HAPI (HITRAN Application Programming Interface) 计算结果为真值进行检验，单色吸收截面最大相对误差小于1%，平均相对误差远小于1%。  

- 特性支持：
  - 支持 Voigt 线型作为吸收线的线型。
  - 线翼截断为 25 cm$^{-1}$
  - 水汽 (H2O) 特化处理：完整集成了 自加宽 (Self-Broadening) 效应与 水汽连续吸收 (Continuum Absorption, MT-CKD模型)。
- 双模式运行：支持单层均匀大气 (SINGLE) 和 垂直大气廓线 (PROFILE) 计算。
- 混合模型架构：针对高压 (HP) 和低压 (LP) 环境分别优化，自动切换模型。  

## 📂 目录结构 (Directory Structure)  

在使用前，请确保项目目录包含 NNLBL_src 内的四个核心组件  

    ```text
    NNLBL_Project/
    │
    ├── environment.yml              # 环境依赖
    │
    ├── NNLBL_TEST.py                # [入口] 使用示例程序 (用户配置)
    │
    ├── NNLBL_src/                   # [源码] 核心算法模块
    │   └── NNLBL_main.py            # 计算主逻辑
    │   └── run_inference_and_save.py # 子模块
    │   └── mt_ckd_h2o.py            # python化的水汽连续吸收计算模块（来自fortran版的MT-CKD-H2O）
    │   └── hapi.py                  # HAPI包的源码 version: 1.2.2.4
    │
    ├── NNmodel&stats/               # [模型] 预训练的神经网络模型与统计参数
    │   ├── voigt_model_hp_*.pth     # 高压模型
    │   ├── voigt_model_lp_*.pth     # 低压模型
    │   └── *.npy                    # 归一化统计参数
    │
    ├── data/                        # [数据] 物理常数与基准数据
    │   ├── absco-ref_wv-mt-ckd.nc   # MT-CKD 水汽连续吸收数据
    │   └── *hapi/                   # HAPI 下载的 HITRAN 数据库缓存
    │
    ├── atmospheric_profile_for_testing/ # [输入] 大气廓线测试数据
    │   ├── pres_100.txt             # 气压层数据
    │   ├── US_STANDARD_ATMOSPHERE_T.txt    # 温度层数据
    │   └── US_STANDARD_ATMOSPHERE_H2O.txt  # 水汽浓度数据 (ppmv)
    │
    ├── sigma_output_filefold/       # [输出] 计算结果存储目录
    └── cache/                       # [缓存] HAPI 基准计算缓存 (加速二次运行)  
    ```  

## 🛠️ 环境依赖 (Requirements)  

见 environment.yml 文件

## 🚀 快速开始 (Quick Start)  

1. 配置参数

打开主程序文件（如 NNLBL_TEST.py），在 if __name__ == "__main__": 下方修改 用户控制面板 (USER CONFIGURATION)。

A. 计算目标与光谱范围

    ``` python

    # 选择分子同位素 ID (参考下文速查表)
    # 示例: 仅计算水汽主同位素 -> [1]
    # 示例: 计算 CO2 和 H2O 混合 -> [7, 1]
    TARGET_ISO_LIST = [1] 

    # 光谱范围 (波数 cm^-1)
    WN_MIN, WN_MAX, WN_STEP = 600, 700, 0.01

    # 水汽连续吸收开关 (仅当分子包含 H2O 时生效)
    ENABLE_H2O_CONTINUUM = True 
    ```  

B. 运行模式设置 (INPUT_MODE)
模式一：单层计算 (SINGLE)适用于实验室环境模拟或单一状态点测试。  

    ```python

    INPUT_MODE = "SINGLE"
    SINGLE_CONFIG = {
        "p_pa": 101325.0,  # 气压 (Pa)
        "t_k": 296.0,      # 温度 (K)
        "vmr": 0.01        # 体积混合比 (无量纲, 0.01 = 1%)
                        # 注意: 计算 H2O 时必须提供 vmr 以计算自加宽
    }
    ```  

模式二：廓线计算 (PROFILE)适用于全大气层辐射传输模拟。  

    ```python

    INPUT_MODE = "PROFILE"
    PROFILE_CONFIG = {
        "p_file": "path/to/pres.txt",  # 气压文件 (单位: mb/hPa, 代码自动转Pa)
        "t_file": "path/to/temp.txt",  # 温度文件 (单位: K)
        "ppmv_file": "path/to/h2o.txt",# 气体浓度文件 (单位: ppmv)
        "name": "US_STD_100"           # 输出文件名标识
    }
    ```

2. 运行程序  

    ```bash  
    python run_NNLBL.py
    ```  

- 如需跳过耗时的 HAPI 基准验证，可添加参数：  

        ```bash
        python run_NNLBL.py --skip-hapi
        ```  

💧 关于水汽 (H2O) 计算的特殊说明由于水汽分子具有强极性和显著的连续吸收效应，本程序对 Global ID = 1 (H2O) 及其同位素进行了特殊处理：自加宽 (Self-Broadening)：程序会强制检查输入数据中的 vmr (Volume Mixing Ratio)。在计算 Voigt 线型时，根据 vmr 动态调整 Diluent 参数 (self vs air)，而非简单的空气加宽。连续吸收 (Continuum)：若 ENABLE_H2O_CONTINUUM = True，程序将调用 MT-CKD 模型（基于 data/absco-ref_wv-mt-ckd.nc）。最终输出的吸收截面 = 神经网络预测 (线吸收) + MT-CKD (连续吸收)。📊 输出结果 (Output)结果将保存为 .h5 (HDF5) 格式，位于 sigma_output_filefold/ 目录下。文件名示例：MOL_1_600_700_0.01_US_STD_100.h5文件结构包含：wavenumber: 波数网格pressure: 各层气压temperature: 各层温度sigma_nn: NNLBL 计算的吸收截面 (含 Continuum)sigma_hapi: HAPI 计算的基准截面 (用于验证误差)
🔢 附录：HITRAN 全局同位素 ID 速查表
