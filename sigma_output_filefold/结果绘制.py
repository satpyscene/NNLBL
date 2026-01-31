import h5py
import matplotlib.pyplot as plt

file_path = "/Users/user/Desktop/0_NNLBL_main_use_阈值修改/sigma_output_filefold/H2O_Iso1-2_4800.0_5200.0_0.01_101325_296.h5"
with h5py.File(file_path, "r") as f:
    nnlbl_sigma = f["model_output/layer_000"][:]
    hapi_sigma = f["hapi_benchmark/layer_000"][:]
    wngrid = f["wavenumber_grid"][:]
    P = f["atmospheric_profile/pressure_pa"][:]
    T = f["atmospheric_profile/temperature_k"][:]
fig, ax1 = plt.subplots(1, 1, figsize=(6, 4), sharex=True)
ax1.plot(wngrid, hapi_sigma, label="HAPI", alpha=0.8)
ax1.plot(wngrid, nnlbl_sigma, label="NNLBL", linestyle="--", alpha=0.8)
ax1.set_ylabel("Absorption Cross Section (cm$^2$/molc.)")
ax1.set_xlabel("Wavenumber cm$^{-1}$")
ax1.set_yscale("log")
ax1.set_title(f"{P[0]/100}hpa, {T[0]}K")
ax1.legend()
ax1.grid(True, alpha=0.3)
