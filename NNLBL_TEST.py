import sys
import os
import numpy as np
from NNLBL_src.NNLBL_main import NNLBL_main, generate_molecule_label

# ==============================================================================
#                               HITRAN 同位素速查表
# ------------------------------------------------------------------------------
# H2O: [1, 2, 3, 4, 5, 6, 129]          |  CO2: [7, 8, 9, 10, 11, 12, 13, 14]
# O3:  [16, 17, 18, 19, 20]             |  N2O: [21, 22, 23, 24, 25]
# CO:  [26, 27, 28, 29, 30, 31]         |  CH4: [32, 33, 34, 35]
# O2:  [36, 37, 38]
# ==============================================================================
# 由于水汽计算的特殊性，计算水汽吸收截面时，要求用户必须输入水汽的相对湿度或者体积混合比，作为get_hapi_physical_params_new中背景大气成分设置以及mt-ckd中水汽连续吸收的设置


if __name__ == "__main__":

    # ##########################################################################
    # 1. 用户控制面板 (USER CONFIGURATION)
    # ##########################################################################

    # [计算目标] 填写全局编号 (Global IDs)，支持多分子混合
    # 例如: CO2主同位素(7) + H2O主同位素(1) -> [7, 1]
    # ⚠️ 注意: 如果列表中包含 [1, 2, 3, 4, 5, 6, 129] (H2O), 系统将强制检查 VMR 输入
    TARGET_ISO_LIST = [1]
    ENABLE_H2O_CONTINUUM = True
    # ENABLE_H2O_CONTINUUM = False
    # [光谱范围] 波数 (cm^-1)
    WN_MIN, WN_MAX, WN_STEP = 600, 700, 0.01

    # [运行模式] 可选: "SINGLE" (单层均匀) | "PROFILE" (大气廓线)
    INPUT_MODE = "SINGLE"

    # 用于计算水汽连续吸收的基准数据文件
    MTCKD_DATA_PATH = "data/absco-ref_wv-mt-ckd.nc"

    # [环境参数]
    # > 模式 A: 单层设置
    # 如果是 H2O，vmr 必填 (例如 0.01 代表 1%)
    # 如果是其他气体，vmr 可选 (填 None 或 0.0 表示无限稀释)
    SINGLE_CONFIG = {"p_pa": 101325.0, "t_k": 296.0, "vmr": 0.00}

    # > 模式 B: 廓线设置 (使用相对路径)
    PROFILE_DIR = "atmospheric_profile_for_testing"
    PROFILE_CONFIG = {
        "p_file": os.path.join(
            PROFILE_DIR, "pres_100.txt"
        ),  # 单位: mb (代码会自动转Pa)
        "t_file": os.path.join(PROFILE_DIR, "US_STANDARD_ATMOSPHERE_T.txt"),
        "ppmv_file": os.path.join(PROFILE_DIR, "US_STANDARD_ATMOSPHERE_H2O.txt"),
        "name": "US_STD_100",  # 用于输出文件名标识
    }

    # [系统路径] 神经网络模型与输出目录
    MODEL_DIR = "NNmodel&stats"
    OUTPUT_DIR = "sigma_output_filefold"

    # ##########################################################################
    # 2. 自动化处理逻辑 (INTERNAL LOGIC) - 通常无需修改
    # ##########################################################################

    # --- A. 生成任务标识 ---
    MOLECULE_LABEL = generate_molecule_label(TARGET_ISO_LIST)
    print(f"任务标识自动生成: {MOLECULE_LABEL}")

    # --- B. 数据加载 ---
    print(f"当前运行模式: {INPUT_MODE}")
    input_p_vals, input_t_vals, input_vmr_vals = None, None, None
    file_suffix = ""

    if INPUT_MODE == "SINGLE":
        input_p_vals = np.array([SINGLE_CONFIG["p_pa"]])
        input_t_vals = np.array([SINGLE_CONFIG["t_k"]])
        val_vmr = SINGLE_CONFIG.get("vmr")
        input_vmr_vals = np.array([val_vmr if val_vmr is not None else 0.0])
        file_suffix = f"{int(input_p_vals[0])}_{int(input_t_vals[0])}"

    elif INPUT_MODE == "PROFILE":
        if not os.path.exists(PROFILE_CONFIG["p_file"]):
            raise FileNotFoundError(f"❌ 找不到文件: {PROFILE_CONFIG['p_file']}")

        print("正在读取廓线文件...")
        input_p_vals = np.loadtxt(PROFILE_CONFIG["p_file"]) * 100  # mb -> Pa
        input_t_vals = np.loadtxt(PROFILE_CONFIG["t_file"])
        ppmv_path = PROFILE_CONFIG.get("ppmv_file")
        if ppmv_path and os.path.exists(ppmv_path):
            input_ppmv_vals = np.loadtxt(ppmv_path)
            input_vmr_vals = input_ppmv_vals * 1e-6
        else:
            # 如果没有提供 VMR 文件，生成全 0 数组
            # (后续逻辑会检查：如果是算 H2O 但这里是全0，可能会报警)
            print("⚠️ 未找到 VMR 廓线文件，将使用默认值 0.0 (无限稀释)")
            input_vmr_vals = np.zeros_like(input_p_vals)
        file_suffix = PROFILE_CONFIG["name"]
    else:
        raise ValueError(f"未知的模式: {INPUT_MODE}")

    # --- C. 路径构建 ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_filename = f"{MOLECULE_LABEL}_{WN_MIN}_{WN_MAX}_{WN_STEP}_{file_suffix}.h5"
    OUTPUT_PATH = os.path.join(OUTPUT_DIR, output_filename)
    print(f"目标输出文件: {OUTPUT_PATH}")

    # 模型映射
    model_paths = {
        "hp_m": os.path.join(
            MODEL_DIR, "voigt_model_hp_Full-nonuniform-n0_1000_noshift.pth"
        ),
        "hp_s": os.path.join(
            MODEL_DIR, "voigt_stats_hp_Full-nonuniform-n0_1000_noshift.npy"
        ),
        "lp_m": os.path.join(
            MODEL_DIR, "voigt_model_lp_Full-nonuniform-n0_1000_noshift.pth"
        ),
        "lp_s": os.path.join(
            MODEL_DIR, "voigt_stats_lp_Full-nonuniform-n0_1000_noshift.npy"
        ),
    }

    # ##########################################################################
    # 3. 启动计算核心 (EXECUTION)
    # ##########################################################################
    SKIP_HAPI = "--skip-hapi" in sys.argv
    # 检查 MTCKD 文件是否存在
    if 1 in TARGET_ISO_LIST and not os.path.exists(MTCKD_DATA_PATH):
        print(
            f"⚠️ 警告: 检测到正在计算水汽 (ID=1)，但找不到 MT-CKD 数据文件: {MTCKD_DATA_PATH}"
        )

    NNLBL_main(
        MOLECULE=MOLECULE_LABEL,
        GLOBAL_WN_MIN=WN_MIN,
        GLOBAL_WN_MAX=WN_MAX,
        GLOBAL_WN_STEP=WN_STEP,
        input_pressures=input_p_vals,
        input_temperatures=input_t_vals,
        input_vmrs=input_vmr_vals,
        mtckd_path=MTCKD_DATA_PATH,
        output_path=OUTPUT_PATH,
        HP_MODEL_PATH=model_paths["hp_m"],
        HP_STATS_PATH=model_paths["hp_s"],
        LP_MODEL_PATH=model_paths["lp_m"],
        LP_STATS_PATH=model_paths["lp_s"],
        skip_hapi=SKIP_HAPI,
        global_iso_ids=TARGET_ISO_LIST,
        enable_continuum=ENABLE_H2O_CONTINUUM,
    )
