import sys
import os
import numpy as np
from NNLBL_src.NNLBL_main import NNLBL_main

if __name__ == "__main__":

    # ==========================================
    # 1. 全局控制开关 (在这里切换模式！)
    # ==========================================
    # 可选模式: "SINGLE" (单层)  |  "PROFILE" (廓线文件)
    # INPUT_MODE = "SINGLE"
    INPUT_MODE = "PROFILE"
    # 全局物理参数
    MOLECULE = "CO2"
    WN_MIN = 600
    WN_MAX = 700
    WN_STEP = 0.01

    # ==========================================
    # 2. 模式参数配置
    # ==========================================

    # --- 配置 A: 单层模式参数 ---
    SINGLE_CONFIG = {"p_pa": 101325.0, "t_k": 296.0}

    # --- 配置 B: 廓线模式参数 ---
    # 建议使用相对路径或统一的 data 目录变量
    BASE_DATA_DIR = "atmospheric_profile_for_testing"
    PROFILE_CONFIG = {
        "p_file": os.path.join(BASE_DATA_DIR, "pres_100.txt"),
        "t_file": os.path.join(BASE_DATA_DIR, "US_STANDARD_ATMOSPHERE_T.txt"),
        "name": "US_STD_100",  # 给廓线起个名字，用于生成文件名
    }

    # ==========================================
    # 3. 自动化数据加载逻辑 (无需手动修改)
    # ==========================================
    input_p_vals = None
    input_t_vals = None
    filename_suffix = ""

    print(f"🚀 当前运行模式: {INPUT_MODE}")

    if INPUT_MODE == "SINGLE":
        # 加载单层数据
        input_p_vals = SINGLE_CONFIG["p_pa"]
        input_t_vals = SINGLE_CONFIG["t_k"]
        # 文件名后缀: _101325_296
        filename_suffix = f"{int(input_p_vals)}_{int(input_t_vals)}"

    elif INPUT_MODE == "PROFILE":
        # 加载廓线数据
        print("正在读取廓线文件...")
        if not os.path.exists(PROFILE_CONFIG["p_file"]):
            raise FileNotFoundError(f"找不到气压文件: {PROFILE_CONFIG['p_file']}")

        input_p_vals = np.loadtxt(PROFILE_CONFIG["p_file"]) * 100  # mb -> Pa
        input_t_vals = np.loadtxt(PROFILE_CONFIG["t_file"])
        # 文件名后缀: _US_STD_100
        filename_suffix = PROFILE_CONFIG["name"]

    else:
        raise ValueError(f"未知的模式: {INPUT_MODE}")

    # ==========================================
    # 4. 构建输出路径与模型路径
    # ==========================================

    # 自动生成输出文件名
    output_dir = "sigma_output_filefold"
    os.makedirs(output_dir, exist_ok=True)  # 自动创建输出文件夹

    output_filename = f"{MOLECULE}_{WN_MIN}_{WN_MAX}_{WN_STEP}_{filename_suffix}.h5"
    OUTPUT_PATH = os.path.join(output_dir, output_filename)

    print(f"📄 目标输出文件: {OUTPUT_PATH}")

    # 模型路径配置 (提取出来，代码更整洁)
    MODEL_DIR = "NNmodel&stats"
    model_paths = {
        "HP_MODEL": os.path.join(
            MODEL_DIR, "voigt_model_hp_Full-nonuniform-n0_1000_noshift.pth"
        ),
        "HP_STATS": os.path.join(
            MODEL_DIR, "voigt_stats_hp_Full-nonuniform-n0_1000_noshift.npy"
        ),
        "LP_MODEL": os.path.join(
            MODEL_DIR, "voigt_model_lp_Full-nonuniform-n0_1000_noshift.pth"
        ),
        "LP_STATS": os.path.join(
            MODEL_DIR, "voigt_stats_lp_Full-nonuniform-n0_1000_noshift.npy"
        ),
    }

    # ==========================================
    # 5. 调用主函数
    # ==========================================
    SKIP_HAPI = "--skip-hapi" in sys.argv

    NNLBL_main(
        MOLECULE,
        WN_MIN,
        WN_MAX,
        WN_STEP,
        input_pressures=input_p_vals,
        input_temperatures=input_t_vals,
        output_path=OUTPUT_PATH,
        HP_MODEL_PATH=model_paths["HP_MODEL"],
        HP_STATS_PATH=model_paths["HP_STATS"],
        LP_MODEL_PATH=model_paths["LP_MODEL"],
        LP_STATS_PATH=model_paths["LP_STATS"],
        skip_hapi=SKIP_HAPI,
    )
