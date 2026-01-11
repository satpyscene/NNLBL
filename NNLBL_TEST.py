import sys
from pathlib import Path
from NNLBL_src.NNLBL_main import NNLBL_API

# ==============================================================================
#                               HITRAN 同位素速查表
# ------------------------------------------------------------------------------
# H2O: [1, 2, 3, 4, 5, 6, 129]          |  CO2: [7, 8, 9, 10, 11, 12, 13, 14]
# O3:  [16, 17, 18, 19, 20]             |  N2O: [21, 22, 23, 24, 25]
# CO:  [26, 27, 28, 29, 30, 31]         |  CH4: [32, 33, 34, 35]
# O2:  [36, 37, 38]
# ==============================================================================
# 由于水汽计算的特殊性，计算水汽吸收截面时，要求用户必须输入水汽的相对湿度或者体积混合比，作为get_hapi_physical_params_new中背景大气成分设置以及mt-ckd中水汽连续吸收的设置

# ==============================================================================
# 用户配置区 (User Configuration)
# ==============================================================================

# 1. 计算目标,想计算什么分子，什么同位素，往里边写序号就行，写几个都成
TARGET_ISO_LIST = [1]
ENABLE_CONTINUUM = True

# 2. 光谱参数
SPECTRAL_CONFIG = {"min": 600, "max": 700, "step": 0.01}

# 3. 运行模式 ("SINGLE" 或 "PROFILE")
RUN_MODE = "SINGLE"

# 4. 环境参数配置
SINGLE_PARAMS = {"p_pa": 101325.0, "t_k": 296.0, "vmr": 0.01}

PROFILE_PARAMS = {
    "dir": "atmospheric_profile_for_testing",
    "p_file": "pres_100.txt",
    "t_file": "US_STANDARD_ATMOSPHERE_T.txt",
    "vmr_file": "US_STANDARD_ATMOSPHERE_h2o.txt",
    "name_tag": "US_STD_100",
}

# 5. 路径配置 (通常不需要动)
PATH_CONFIG = {
    "base_dir": Path(__file__).parent,  # 自动获取当前目录
    "model_dir": "NNmodel&stats",  # 神经网络模型文件夹名
    "output_dir": "sigma_output_filefold",  # 输出文件夹名
    "mtckd_file": "data/absco-ref_wv-mt-ckd.nc",  # MT-CKD 数据路径
}

# ==============================================================================
# 程序入口
# ==============================================================================
if __name__ == "__main__":

    # 一行代码启动，无需任何逻辑处理
    NNLBL_API(
        target_iso_list=TARGET_ISO_LIST,
        spectral_config=SPECTRAL_CONFIG,
        input_mode=RUN_MODE,
        single_config=SINGLE_PARAMS,
        profile_config=PROFILE_PARAMS,
        path_config=PATH_CONFIG,
        enable_continuum=ENABLE_CONTINUUM,
        skip_hapi=("--skip-hapi" in sys.argv),
    )
