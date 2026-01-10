import sys
import os
import numpy as np
from NNLBL_src.NNLBL_main import NNLBL_main


def generate_molecule_label(global_iso_ids):
    # ==========================================
    # 1. 你的专属数据表
    # ==========================================
    TOP7_ISO_TABLE = {
        "H2O": [1, 2, 3, 4, 5, 6, 129],
        "CO2": [7, 8, 9, 10, 11, 12, 13, 14, 121, 15, 120, 122],
        "O3": [16, 17, 18, 19, 20],
        "N2O": [21, 22, 23, 24, 25],
        "CO": [26, 27, 28, 29, 30, 31],
        "CH4": [32, 33, 34, 35],
        "O2": [36, 37, 38],
    }

    if not global_iso_ids:
        return "Unknown_Empty"

    # 用于临时存储找到的分子及其包含的局部索引
    # 结构: { "CO2": [1, 2], "H2O": [1] } (这里存的是列表中的下标+1)
    found_map = {}

    # ==========================================
    # 2. 核心查找逻辑
    # ==========================================
    for gid in global_iso_ids:
        found = False
        for mol_name, iso_list in TOP7_ISO_TABLE.items():
            if gid in iso_list:
                if mol_name not in found_map:
                    found_map[mol_name] = []

                # 获取该ID在列表中的位置 (从1开始计数)，作为"IsoX"的标号
                local_index = iso_list.index(gid) + 1
                found_map[mol_name].append(local_index)
                found = True
                break

        if not found:
            print(f"⚠️ 警告: 全局编号 {gid} 不在 TOP7 表格中，将被忽略。")

    # ==========================================
    # 3. 生成描述性字符串
    # ==========================================
    name_segments = []

    # 按表格定义的顺序来生成文件名，保持整洁
    for mol_name in TOP7_ISO_TABLE.keys():
        if mol_name in found_map:
            indices = sorted(found_map[mol_name])
            total_count_in_table = len(TOP7_ISO_TABLE[mol_name])

            # --- 命名规则 ---
            if len(indices) == total_count_in_table:
                # 规则1: 如果包含该分子全部同位素 -> "_All"
                label = f"{mol_name}_All"
            elif len(indices) == 1 and indices[0] == 1:
                # 规则2: 如果只有第1个(通常是主同位素) -> "_Major"
                label = f"{mol_name}_Major"
            else:
                # 规则3: 其他情况 -> "_Iso1-2-5"
                str_indices = "-".join(map(str, indices))
                label = f"{mol_name}_Iso{str_indices}"

            name_segments.append(label)

    if not name_segments:
        return "Unknown_NoMatch"

    return "_".join(name_segments)


if __name__ == "__main__":
    # 对应分子同位素的全局编号，每个分子从前到后按丰度排序，使用全局编号放置在glable_iso_list里，即可控制目标分子和同位素，可以混合
    TOP7_ISO_TABLE = {
        "H2O": [1, 2, 3, 4, 5, 6, 129],
        "CO2": [7, 8, 9, 10, 11, 12, 13, 14, 121, 15, 120, 122],
        "O3": [16, 17, 18, 19, 20],
        "N2O": [21, 22, 23, 24, 25],
        "CO": [26, 27, 28, 29, 30, 31],
        "CH4": [32, 33, 34, 35],
        "O2": [36, 37, 38],
    }

    # ==========================================
    # 1. 全局控制开关 (在这里切换模式！)
    # ==========================================
    # 可选模式: "SINGLE" (单层)  |  "PROFILE" (廓线文件)
    # INPUT_MODE = "SINGLE"
    INPUT_MODE = "SINGLE"
    # 全局物理参数

    glable_iso_list = [1, 2, 7, 8, 9, 16, 17]
    MOLECULE = generate_molecule_label(glable_iso_list)
    print(f"自动生成的标识: {MOLECULE}")

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
        global_iso_ids=glable_iso_list,
    )
