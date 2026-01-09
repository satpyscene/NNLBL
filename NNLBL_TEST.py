import sys
from NNLBL_src.NNLBL_main import NNLBL_main

if __name__ == "__main__":

    # --- 1. 定义用户输入 (User Inputs) ---
    # 这些是每次运行可能变化的参数
    MOLECULE = "CO2"

    GLOBAL_WN_MIN = 600
    GLOBAL_WN_MAX = 700
    GLOBAL_WN_STEP = 0.01

    # 示例 B: 直接定义数值 (单层模式 - 如果你想用这个，注释掉上面的文件读取)
    target_p = 101325.0
    target_t = 296.0

    # 示例 A: 从文件读取 (大气廓线模式)
    # PRESSURE_FILE = "/data/Dayaoyjy_GPU/NN_VOIGT_OLD_PAPER/2_moe_threshold_settings/标准廓线/pres_100.txt"
    # TEMPERATURE_FILE = "/data/Dayaoyjy_GPU/NN_VOIGT_OLD_PAPER/2_moe_threshold_settings/标准廓线/US_STANDARD_ATMOSPHERE_T.txt"

    OUTPUT_H5 = f"/sigma_output_filefold/{MOLECULE}_{GLOBAL_WN_MIN}_{GLOBAL_WN_MAX}_{GLOBAL_WN_STEP}_{target_p}_{target_t}.h5"

    # 示例 B: 直接定义数值 (单层模式 - 如果你想用这个，注释掉上面的文件读取)
    # target_p = 101325.0
    # target_t = 296.0

    # --- 2. 解析命令行标志 ---
    SKIP_HAPI_FLAG = "--skip-hapi" in sys.argv

    # --- 3. 准备数据 ---
    # print("正在读取源数据文件...")
    # 注意：此处进行了单位转换 (mb -> Pa)，这是数据准备的一部分
    # input_p_vals = np.loadtxt(PRESSURE_FILE) * 100
    # input_t_vals = np.loadtxt(TEMPERATURE_FILE)

    # 如果是单层模式，这里可以是:
    input_p_vals = target_p
    input_t_vals = target_t

    # --- 4. 调用主函数 ---
    NNLBL_main(
        MOLECULE,
        GLOBAL_WN_MIN,
        GLOBAL_WN_MAX,
        GLOBAL_WN_STEP,
        input_pressures=input_p_vals,
        input_temperatures=input_t_vals,
        output_path=f"sigma_output_filefold/{MOLECULE}_{GLOBAL_WN_MIN}_{GLOBAL_WN_MAX}_{GLOBAL_WN_STEP}_{target_p}_{target_t}.h5",
        HP_MODEL_PATH="NNmodel&stats/voigt_model_hp_Full-nonuniform-n0_1000_noshift.pth",
        HP_STATS_PATH="NNmodel&stats/voigt_stats_hp_Full-nonuniform-n0_1000_noshift.npy",
        LP_MODEL_PATH="NNmodel&stats/voigt_model_lp_Full-nonuniform-n0_1000_noshift.pth",
        LP_STATS_PATH="NNmodel&stats/voigt_stats_lp_Full-nonuniform-n0_1000_noshift.npy",
        skip_hapi=SKIP_HAPI_FLAG,
    )
