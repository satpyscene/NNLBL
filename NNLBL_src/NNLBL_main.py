import sys
import os
import time
import numpy as np
import torch
import pickle
import hashlib
from joblib import Parallel, delayed
from tqdm import tqdm
from .run_inference_and_save import (
    create_non_uniform_grid,
    get_hapi_physical_params_new,
    load_model,
    process_mega_batch_gpu,
    process_superposition_from_gpu,
    calculate_hapi_benchmark_new,
    save_to_hdf5,
)


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


# ==============================================================================
# 1. 全局配置常量 (System Configuration)
#    这些参数定义了环境、模型路径和物理常数，通常在main内部直接调用
# ==============================================================================

# --- 路径配置 (服务器/环境相关) ---


# ==============================================================================
# 2. 主逻辑函数 (Core Logic)
#    这里只定义逻辑，不包含具体的数据文件路径。
#    输入：气压(Pa), 温度(K), 输出路径, 标志位
# ==============================================================================
def NNLBL_main(
    MOLECULE,
    GLOBAL_WN_MIN,
    GLOBAL_WN_MAX,
    GLOBAL_WN_STEP,
    input_pressures,
    input_temperatures,
    output_path,
    HP_MODEL_PATH="NNmodel&stats/voigt_model_hp_Full-nonuniform-n0_1000_noshift.pth",
    HP_STATS_PATH="NNmodel&stats/voigt_stats_hp_Full-nonuniform-n0_1000_noshift.npy",
    LP_MODEL_PATH="NNmodel&stats/voigt_model_lp_Full-nonuniform-n0_1000_noshift.pth",
    LP_STATS_PATH="NNmodel&stats/voigt_stats_lp_Full-nonuniform-n0_1000_noshift.npy",
    skip_hapi=False,
    global_iso_ids=None,
):
    print("!" * 80)
    print(
        "致谢，NNLBL算法的线参数订正模块，训练数据，以及结果验证均来自HAPI！HAPI version: 1.2.2.4"
    )
    print("!" * 80)

    PRESSURE_THRESHOLD_PA = 2200.0  # 高低压分界线
    WING_SIZE = 25.0
    TOTAL_POINTS_FULL = 5001
    CONCENTRATION_FACTOR = 6.0

    print("=" * 80)
    print(f"启动推理流程 | 输出目标: {output_path}")
    print("=" * 80)

    # ---------------------------------------------------
    # A. 环境与网格初始化
    # ---------------------------------------------------
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {DEVICE}")

    # 全局均匀网格 (用于最终输出)
    global_wavenumber_grid = np.arange(GLOBAL_WN_MIN, GLOBAL_WN_MAX, GLOBAL_WN_STEP)

    # 基础非均匀网格 (用于神经网络输入)
    base_wavenumber_grid = create_non_uniform_grid(
        1000.0, WING_SIZE, TOTAL_POINTS_FULL, CONCENTRATION_FACTOR
    )
    base_grid_gpu = torch.tensor(
        base_wavenumber_grid, dtype=torch.float32, device=DEVICE
    )

    # ---------------------------------------------------
    # B. 输入数据标准化 (处理单层/多层逻辑)
    # ---------------------------------------------------

    # 确保输入是 numpy 数组
    p_data = np.array(input_pressures, dtype=float)
    t_data = np.array(input_temperatures, dtype=float)

    # 自动判断：如果是单个数字(0维)，转为1维数组；如果是数组，保持原样
    if p_data.ndim == 0:
        p_data = p_data.reshape(1)
        t_data = t_data.reshape(1)
        print(">> 模式检测: 单层输入 (Single Layer)")
    else:
        print(f">> 模式检测: 大气廓线输入 (Profile, {len(p_data)} Layers)")

    # 校验维度一致性
    if p_data.shape != t_data.shape:
        raise ValueError("❌ 错误: 气压和温度数据的长度不一致！")

    # 直接使用输入值作为层的状态
    atmospheric_profile = [
        {"layer": i, "pressure_pa": p, "temperature_k": t}
        for i, (p, t) in enumerate(zip(p_data, t_data))
    ]
    print(f"大气分子参数加载: 共 {len(atmospheric_profile)} 层")

    # ---------------------------------------------------
    # C. 预计算 HAPI 参数
    # ---------------------------------------------------
    print("\n根据实际温压订正Voigt线型参数...")
    t_start = time.perf_counter()

    all_layers_lines_params = Parallel(n_jobs=16)(
        delayed(get_hapi_physical_params_new)(
            MOLECULE,
            GLOBAL_WN_MIN,
            GLOBAL_WN_MAX,
            layer["temperature_k"],
            layer["pressure_pa"],
            global_iso_ids=global_iso_ids,
        )
        for layer in tqdm(atmospheric_profile, desc="吸收线参数计算进程")
    )
    print(f"吸收线参数全部计算完成，耗时: {time.perf_counter() - t_start:.2f}秒")

    # ---------------------------------------------------
    # D. 加载神经网络模型
    # ---------------------------------------------------
    # print("\n加载神经网络模型...")
    statistics_pool = {
        "lp": np.load(LP_STATS_PATH, allow_pickle=True).item(),
        "hp": np.load(HP_STATS_PATH, allow_pickle=True).item(),
    }
    models = {
        "lp": load_model(
            LP_MODEL_PATH,
            2,
            [100, 500, 1000, 500],
            TOTAL_POINTS_FULL,
            DEVICE,
            statistics_pool["lp"],
        ),
        "hp": load_model(
            HP_MODEL_PATH,
            2,
            [100, 500, 1000, 500],
            TOTAL_POINTS_FULL,
            DEVICE,
            statistics_pool["hp"],
        ),
    }
    print("两个神经网络模型已加载至目标设备")

    # ---------------------------------------------------
    # E. GPU 推理 (保持原有逻辑)
    # ---------------------------------------------------
    # print("\n开始GPU推理...")
    t_infer_start = time.perf_counter()

    # 根据气压阈值分组
    hp_layer_indices = [
        i
        for i, l in enumerate(atmospheric_profile)
        if l["pressure_pa"] >= PRESSURE_THRESHOLD_PA
    ]
    lp_layer_indices = [
        i
        for i, l in enumerate(atmospheric_profile)
        if l["pressure_pa"] < PRESSURE_THRESHOLD_PA
    ]

    all_layer_absorptions = [None] * len(atmospheric_profile)
    mega_batch_size = 2

    # --- 处理低压层 (LP) ---
    if lp_layer_indices:
        print(f"处理低压层 ({len(lp_layer_indices)}层)...")
        for batch_start in tqdm(range(0, len(lp_layer_indices), mega_batch_size)):
            batch_end = min(batch_start + mega_batch_size, len(lp_layer_indices))
            batch_indices = lp_layer_indices[batch_start:batch_end]

            layer_gpu_results = process_mega_batch_gpu(
                batch_indices,
                all_layers_lines_params,
                models["lp"],
                base_grid_gpu,
                DEVICE,
            )

            # 后处理：插值回全局网格
            batch_absorptions = Parallel(n_jobs=-1, backend="threading")(
                delayed(process_superposition_from_gpu)(
                    p, w, global_wavenumber_grid, base_wavenumber_grid
                )
                for p, w in layer_gpu_results
            )

            for i, idx in enumerate(batch_indices):
                all_layer_absorptions[idx] = batch_absorptions[i]

    # --- 处理高压层 (HP) ---
    if hp_layer_indices:
        print(f"处理高压层 ({len(hp_layer_indices)}层)...")
        for batch_start in tqdm(range(0, len(hp_layer_indices), mega_batch_size)):
            batch_end = min(batch_start + mega_batch_size, len(hp_layer_indices))
            batch_indices = hp_layer_indices[batch_start:batch_end]

            layer_gpu_results = process_mega_batch_gpu(
                batch_indices,
                all_layers_lines_params,
                models["hp"],
                base_grid_gpu,
                DEVICE,
            )

            # 后处理：插值回全局网格
            batch_absorptions = Parallel(n_jobs=-1, backend="threading")(
                delayed(process_superposition_from_gpu)(
                    p, w, global_wavenumber_grid, base_wavenumber_grid
                )
                for p, w in layer_gpu_results
            )

            for i, idx in enumerate(batch_indices):
                all_layer_absorptions[idx] = batch_absorptions[i]

    print(
        f"NNLBL吸收截面光谱计算结束，耗时: {time.perf_counter() - t_infer_start:.2f}秒"
    )

    # ---------------------------------------------------
    # F. HAPI 基准计算与缓存
    # ---------------------------------------------------
    hapi_results = None
    if skip_hapi:
        print("\n⚠️  跳过HAPI吸收截面光谱计算")
    else:
        # 如果 global_iso_ids 为 None，记录为 'default'，否则将列表转为字符串
        iso_tag = (
            "_".join(map(str, sorted(global_iso_ids))) if global_iso_ids else "default"
        )
        # 缓存键生成：基于分子、范围以及输入数据的特征(避免过长文件名)
        # 使用数据摘要(长度 + 首元素)来区分不同输入
        data_signature = f"L{len(p_data)}_P{p_data[0]:.2f}_T{t_data[0]:.2f}"
        cache_key = (
            f"{MOLECULE}_{GLOBAL_WN_MIN}_{GLOBAL_WN_MAX}_{iso_tag}_{data_signature}"
        )
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:16]
        hapi_cache_path = os.path.join("cache", f"hapi_results_{cache_hash}.pkl")
        os.makedirs("cache", exist_ok=True)

        if os.path.exists(hapi_cache_path):
            print(f"\n发现HAPI吸收截面缓存数据: {hapi_cache_path}")
            print(f">> 缓存标识 (Isotopes): {iso_tag}")
            t_load_start = time.perf_counter()
            with open(hapi_cache_path, "rb") as f:
                hapi_results = pickle.load(f)
            print(f"缓存加载完成，耗时: {time.perf_counter() - t_load_start:.2f}秒")
        else:
            print("\n未找到缓存，开始HAPI计算...")
            t_hapi_start = time.perf_counter()
            hapi_results = Parallel(n_jobs=1)(
                delayed(calculate_hapi_benchmark_new)(
                    MOLECULE,
                    global_wavenumber_grid,
                    layer["temperature_k"],
                    layer["pressure_pa"],
                    GLOBAL_WN_MIN,
                    GLOBAL_WN_MAX,
                    global_iso_ids=global_iso_ids,
                )
                for layer in tqdm(atmospheric_profile, desc="HAPI计算")
            )
            print(f"HAPI计算完成，耗时: {time.perf_counter() - t_hapi_start:.2f}秒")

            print("保存HAPI缓存...")
            with open(hapi_cache_path, "wb") as f:
                pickle.dump(hapi_results, f, protocol=pickle.HIGHEST_PROTOCOL)

    # ---------------------------------------------------
    # G. 数据保存
    # ---------------------------------------------------
    save_to_hdf5(
        output_path,
        atmospheric_profile,
        all_layer_absorptions,
        hapi_results,
        global_wavenumber_grid,
        MOLECULE,
    )

    print("=" * 80)
    print("全部完成！")
    print("=" * 80)


# ==============================================================================
# 3. 脚本入口 (User Interface)
#    这里处理用户输入：读取哪些文件，还是直接给数值，保存到哪里
# ==============================================================================
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

    OUTPUT_H5 = f"../sigma_output_filefold/{MOLECULE}_{GLOBAL_WN_MIN}_{GLOBAL_WN_MAX}_{GLOBAL_WN_STEP}_{target_p}_{target_t}.h5"

    # 示例 B: 直接定义数值 (单层模式 - 如果你想用这个，注释掉上面的文件读取)
    # target_p = 101325.0
    # target_t = 296.0

    # --- 2. 解析命令行标志 ---
    SKIP_HAPI_FLAG = "--skip-hapi" in sys.argv

    # --- 3. 准备数据 ---
    print("正在读取源数据文件...")
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
        output_path=OUTPUT_H5,
        skip_hapi=SKIP_HAPI_FLAG,
        HP_MODEL_PATH="../NNmodel&stats/voigt_model_hp_Full-nonuniform-n0_1000_noshift.pth",
        HP_STATS_PATH="../NNmodel&stats/voigt_stats_hp_Full-nonuniform-n0_1000_noshift.npy",
        LP_MODEL_PATH="../NNmodel&stats/voigt_model_lp_Full-nonuniform-n0_1000_noshift.pth",
        LP_STATS_PATH="../NNmodel&stats/voigt_stats_lp_Full-nonuniform-n0_1000_noshift.npy",
    )
