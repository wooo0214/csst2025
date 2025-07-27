from astropy.io import fits
import numpy as np

# 加载FITS文件
with fits.open("your_data.fits") as hdul:
    data_cube = hdul[0].data  # 数据立方（shape: [波长, y, x]）
    header = hdul[0].header   # 头文件

# 提取波长轴（假设头文件包含CDELT3/CRVAL3）
from astropy.wcs import WCS
wcs = WCS(header)
n_wave = header["NAXIS3"]
wave = wcs.spectral.pixel_to_world(np.arange(n_wave))  # 波长数组（单位：Å）

# 截取Hα波段（λ6400-6700Å）
import spectral_cube

# 转换为SpectralCube对象（方便截取波段）
cube = spectral_cube.SpectralCube(data=data_cube, wcs=wcs)
ha_band = cube.spectral_slab(6400 * u.AA, 6700 * u.AA)  # 截取波段

# 获取截取后的数据和波长
ha_data = ha_band.hdu.data  # 三维数组
ha_wave = ha_band.spectral_axis.value  # 波长值（Å）

# 逐像素拟合发射线（以Hα+[N II]为例）
from lmfit.models import GaussianModel, ConstantModel
import matplotlib.pyplot as plt

# 定义拟合模型（窄Hα + [N II]λλ6548,6583 + 宽Hα）
def fit_ha_spectrum(wave, flux, noise):
    model = ConstantModel(prefix='cont_')  # 连续谱背景
    
    # 窄线成分（约束[N II]λ6548/λ6583的强度比为1:3）
    narrow_ha = GaussianModel(prefix='nha_')
    nii6548 = GaussianModel(prefix='nii6548_')
    nii6583 = GaussianModel(prefix='nii6583_')
    
    # 宽线成分（可选）
    broad_ha = GaussianModel(prefix='bha_')
    
    # 组合模型
    full_model = model + narrow_ha + nii6548 + nii6583 + broad_ha
    
    # 设置初始参数
    params = full_model.make_params()
    params['cont_c'].set(value=np.median(flux), min=0)
    
    # Hα窄线（λ6563Å）
    params['nha_center'].set(value=6563, min=6550, max=6570)
    params['nha_sigma'].set(value=2.0, min=0.5, max=5.0)  # σ≈100 km/s
    
    # [N II]λ6548（固定与λ6583的波长比和强度比）
    params['nii6548_center'].set(expr='nha_center - 15.3')  # 6548Å与6563Å的差
    params['nii6548_amplitude'].set(expr='nii6583_amplitude / 3.0')  # 强度比1:3
    
    # [N II]λ6583
    params['nii6583_center'].set(expr='nha_center + 20.3')  # 6583Å与6563Å的差
    params['nii6583_sigma'].set(expr='nha_sigma')  # 与Hα窄线相同σ
    
    # Hα宽线（若存在）
    params['bha_center'].set(value=6563, min=6550, max=6570)
    params['bha_sigma'].set(value=10.0, min=5.0, max=30.0)  # σ≈500 km/s
    
    # 拟合
    result = full_model.fit(flux, params, x=wave, weights=1/noise)
    return result

# 示例：对单个像素（x=50, y=50）拟合
x, y = 50, 50
flux = ha_data[:, y, x]
noise = np.sqrt(np.abs(flux))  # 简单估计噪声（实际需更精确）
result = fit_ha_spectrum(ha_wave, flux, noise)

# 绘制拟合结果
plt.plot(ha_wave, flux, 'k-', label='Data')
plt.plot(ha_wave, result.best_fit, 'r-', label='Fit')
plt.plot(ha_wave, result.eval_components()['narrow_ha'], 'b--', label='Narrow Hα')
plt.legend()
plt.xlabel('Wavelength (Å)')
plt.ylabel('Flux')
plt.show()

from tqdm import tqdm  # 进度条

# 初始化结果存储数组
ny, nx = ha_data.shape[1], ha_data.shape[2]
v_narrow = np.zeros((ny, nx))  # 窄线速度场
sigma_narrow = np.zeros((ny, nx))  # 窄线速度弥散
v_broad = np.zeros((ny, nx))  # 宽线速度场

# 逐像素拟合
for y in tqdm(range(ny)):
    for x in range(nx):
        flux = ha_data[:, y, x]
        noise = np.sqrt(np.abs(flux))
        try:
            result = fit_ha_spectrum(ha_wave, flux, noise)
            # 保存窄线速度（km/s）
            v_narrow[y, x] = 3e5 * (result.params['nha_center'].value - 6563) / 6563
            sigma_narrow[y, x] = 3e5 * result.params['nha_sigma'].value / 6563
            # 保存宽线速度（若存在）
            if 'bha_center' in result.params:
                v_broad[y, x] = 3e5 * (result.params['bha_center'].value - 6563) / 6563
        except:
            v_narrow[y, x] = np.nan  # 拟合失败标记

# 保存结果到FITS
from astropy.io import fits
hdu_v = fits.PrimaryHDU(v_narrow, header=ha_band.wcs.celestial.to_header())
hdu_v.writeto('narrow_velocity_field.fits', overwrite=True)

import aplpy

# 绘制窄线速度场
fig = aplpy.FITSFigure('narrow_velocity_field.fits')
fig.show_colorscale(cmap='RdBu_r', vmin=-200, vmax=200)  # 假设速度范围±200 km/s
fig.add_colorbar()
fig.colorbar.set_label('Velocity (km/s)')
fig.save('velocity_map.png')
