# branched_flow/physics/potentials.py

import numpy as np
import os
from scipy.interpolate import RegularGridInterpolator

# ==============================================================================
# 1. 定义势场源函数 (Providers)
# ==============================================================================

def from_file(path, x_start, x_end_grid, y_start, y_end_grid, dx, dy, **kwargs):
    """
    从 .npz 文件加载势能场。
    如果文件中的网格与模拟所需的网格不匹配，将使用插值进行重采样。
    文件应包含 'potential' (矩阵) 和 'extents' (x_min, x_max, y_min, y_max)。
    """
    print(f"  [势场生成] 正在从文件 '{path}' 加载势能场...")
    if not os.path.exists(path):
        raise FileNotFoundError(f"错误：势能文件 '{path}' 不存在。")

    data = np.load(path)
    loaded_potential = data['potential']
    x_min_file, x_max_file, y_min_file, y_max_file = data['extents']

    ny_target = int(round((y_end_grid - y_start) / dy)) + 1
    nx_target = int(round((x_end_grid - x_start) / dx)) + 1
    x_target = np.linspace(x_start, x_end_grid, nx_target)
    y_target = np.linspace(y_start, y_end_grid, ny_target)
    
    ny_source, nx_source = loaded_potential.shape
    x_source = np.linspace(x_min_file, x_max_file, nx_source)
    y_source = np.linspace(y_min_file, y_max_file, ny_source)

    if (ny_source == ny_target and nx_source == nx_target and
            np.allclose(x_source, x_target) and np.allclose(y_source, y_target)):
        print("  [势场生成] 文件网格与模拟网格匹配，直接使用。")
        return loaded_potential
    else:
        print("  [势场生成] 文件网格与模拟网格不匹配，正在进行插值重采样...")
        interpolator = RegularGridInterpolator((y_source, x_source), loaded_potential,
                                               bounds_error=False, fill_value=0)
        
        yv, xv = np.meshgrid(y_target, x_target, indexing='ij')
        points_to_interpolate = np.vstack((yv.ravel(), xv.ravel())).T
        resampled_potential = interpolator(points_to_interpolate).reshape(ny_target, nx_target)
        
        print(f"  [势场生成] 成功将势能场重采样至 ({ny_target} x {nx_target})。")
        return resampled_potential

def sinusoidal(x_start, x_end_grid, y_start, y_end_grid, dx, dy, a, b, c, **kwargs):
    """
    生成一个 V(x,y) = a * sin(b*x) * sin(c*y) 形式的正弦势能场。
    """
    print(f"  [势场生成] 正在创建 a={a}, b={b}, c={c} 的正弦势能场...")
    ny = int(round((y_end_grid - y_start) / dy)) + 1
    nx = int(round((x_end_grid - x_start) / dx)) + 1

    x_coords = np.linspace(x_start, x_end_grid, nx)
    y_coords = np.linspace(y_start, y_end_grid, ny)

    xv, yv = np.meshgrid(x_coords, y_coords, indexing='xy')
    
    potential_matrix = a * np.sin(b * xv) * np.sin(c * yv)
    
    return potential_matrix.T

def fourier_synthesis(x_start, x_end_grid, y_start, y_end_grid, dx, dy, alpha, amplitude=1.0, seed=None, **kwargs):
    """
    【已修正】使用傅里叶空间滤波方法生成一个统计上均匀的、平滑的随机势能场。
    """
    print(f"  [势场生成] (已修正方法) 正在使用傅里叶滤波创建 alpha={alpha} 的随机势能场...")
    
    ny = int(round((y_end_grid - y_start) / dy)) + 1
    nx = int(round((x_end_grid - x_start) / dx)) + 1
    
    rng = np.random.default_rng(seed)
    
    # 1. 在真实空间生成高斯白噪声
    noise = rng.standard_normal(size=(ny, nx))
    
    # 2. 将其变换到频率空间
    noise_k = np.fft.fft2(noise)
    
    # 3. 创建频率坐标网格 (ky, kx) 并计算波数 k
    ky = np.fft.fftfreq(ny, d=dy) * 2 * np.pi
    kx = np.fft.fftfreq(nx, d=dx) * 2 * np.pi
    ky_mesh, kx_mesh = np.meshgrid(ky, kx, indexing='ij')
    k = np.sqrt(kx_mesh**2 + ky_mesh**2)
    
    # 避免除以零
    k[0, 0] = 1.0 
    
    # 4. 根据谱指数 alpha 计算振幅滤波器
    # P(k) ~ k^(-alpha), 所以振幅 A(k) ~ k^(-alpha/2)
    power_law_filter = k**(-alpha / 2.0)
    power_law_filter[0, 0] = 0  # 移除直流分量
    
    # 5. 将滤波器应用到噪声的频谱上
    filtered_noise_k = noise_k * power_law_filter
    
    # 6. 通过逆傅里叶变换，得到真实空间中的势能场
    potential_matrix = np.fft.ifft2(filtered_noise_k).real
    
    # 7. 标准化并应用最终的振幅
    std_dev = np.std(potential_matrix)
    if std_dev > 1e-9:
        potential_matrix = (potential_matrix / std_dev) * amplitude
        
    print(f"  [势场生成] 成功生成 ({ny} x {nx}) 的势能场。")
    return potential_matrix

# branched_flow/physics/potentials.py

# ... (在 fourier_synthesis 函式之後) ...

def fourier_synthesis_gpu(xp, ny, nx, dx, dy, alpha, amplitude=1.0, seed=None):
    """
    【GPU 版本】直接在 GPU 上使用 CuPy 產生隨機勢能場。
    """
    # 這裡的 xp 將會是 cupy 模組
    rng = xp.random.default_rng(seed)
    noise = rng.standard_normal(size=(ny, nx))
    noise_k = xp.fft.fft2(noise)
    
    ky = xp.fft.fftfreq(ny, d=dy) * 2 * np.pi
    kx = xp.fft.fftfreq(nx, d=dx) * 2 * np.pi
    ky_mesh, kx_mesh = xp.meshgrid(ky, kx, indexing='ij')
    k = xp.sqrt(kx_mesh**2 + ky_mesh**2)
    
    k[0, 0] = 1.0
    power_law_filter = k**(-alpha / 2.0)
    power_law_filter[0, 0] = 0
    
    filtered_noise_k = noise_k * power_law_filter
    potential_matrix = xp.fft.ifft2(filtered_noise_k).real
    
    std_dev = xp.std(potential_matrix)
    if std_dev > 1e-9:
        potential_matrix = (potential_matrix / std_dev) * amplitude
        
    return potential_matrix

# ==============================================================================
# 2. 注册源函数
# ==============================================================================

potential_source_registry = {
    'from_file': from_file,
    'sinusoidal': sinusoidal,
    'fourier_synthesis': fourier_synthesis,
    'fourier_synthesis_gpu': fourier_synthesis_gpu
}

# ==============================================================================
# 3. 定义配置
# ==============================================================================

potential_configs = {
    'smooth_random': {
        'provider': 'fourier_synthesis',
        'args': {'alpha': 2.0, 'seed': 20, 'amplitude': 0.1},
        'description': "一个透过傅立叶合成生成的、平滑的随机势能场（标准关联长度）。"
    },
    'smooth_random_gpu': {
        'provider': 'fourier_synthesis_gpu',
        'args': {'alpha': 2.0, 'seed': 20, 'amplitude': 0.1},
        'description': "一个透过傅立叶合成生成的、平滑的随机势能场（标准关联长度）。"
    },
    'smooth_random_large_corr': {
        'provider': 'fourier_synthesis',
        'args': {'alpha': 4.0, 'seed': 20, 'amplitude': 0.1},
        'description': "一个具有更大关联长度的、更平滑的随机势能场。"
    },
    'sinusoidal_default': {
        'provider': 'sinusoidal',
        'args': {'a': 1.0, 'b': 1.0, 'c': 1.0},
        'description': "一个标准的 V(x,y) = a*sin(b*x)*sin(c*y) 势能场。"
    },
    'from_file_example': {
        'provider': 'from_file',
        'args': {'path': 'my_potential.npz'},
        'description': "从名为 my_potential.npz 的文件中加载势能场。"
    }
}