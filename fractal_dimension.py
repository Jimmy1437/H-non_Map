import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import rcParams
from scipy.stats import linregress
import platform
import os

# 打印系统信息
print("=" * 60)
print(f"操作系统: {platform.system()} {platform.release()}")
print(f"Python 版本: {platform.python_version()}")
print(f"Matplotlib 版本: {mpl.__version__}")  # 使用 mpl 获取版本
print("=" * 60)


# 1. 系统字体检测与设置
def setup_fonts():
    """配置系统字体以确保正确显示"""
    # 尝试设置支持中文的字体
    chinese_fonts = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'SimSun', 'WenQuanYi Micro Hei',
                     'Arial Unicode MS', 'Noto Sans CJK SC', 'STKaiti', 'STSong']

    # 获取系统所有可用字体
    all_fonts = [f.name for f in fm.fontManager.ttflist]
    print(f"检测到系统安装字体数量: {len(all_fonts)}")

    # 查找推荐字体
    available_fonts = []
    for font in chinese_fonts:
        if any(f.lower() == font.lower() for f in all_fonts):
            available_fonts.append(font)

    # 设置字体
    if available_fonts:
        print(f"找到可用中文字体: {', '.join(available_fonts)}")
        # 使用第一个找到的字体
        selected_font = available_fonts[0]
        rcParams['font.family'] = 'sans-serif'
        rcParams['font.sans-serif'] = [selected_font]
        print(f"已设置字体: {selected_font}")
    else:
        print("警告: 未找到推荐的中文字体，尝试使用默认字体")
        # 打印前20个可用字体供参考
        print("前20个可用字体:")
        for i, font in enumerate(all_fonts[:20]):
            print(f"{i + 1}. {font}")

    # 设置其他字体参数
    rcParams['axes.unicode_minus'] = False  # 正确显示负号
    rcParams['font.size'] = 12
    print("=" * 60)


# 配置字体
setup_fonts()


# 2. Henon 映射函数
def henon_map(x, y, a=1.4, b=0.3):
    """计算单次 Henon 映射迭代"""
    x_next = 1 - a * x ** 2 + y
    y_next = b * x
    return x_next, y_next


# 3. 生成吸引子点集
def generate_henon_attractor(n_points=100000, transients=1000, a=1.4, b=0.3):
    """生成 Henon 吸引子点集"""
    # 初始化数组
    x_vals = np.zeros(n_points + transients)
    y_vals = np.zeros(n_points + transients)

    # 初始条件
    x_vals[0], y_vals[0] = 0.1, 0.1

    # 迭代生成轨迹
    for i in range(1, n_points + transients):
        x_vals[i], y_vals[i] = henon_map(x_vals[i - 1], y_vals[i - 1], a, b)

    # 丢弃瞬态部分
    return x_vals[transients:], y_vals[transients:]


# 4. 盒计数法计算分形维数
def box_counting_dimension(points, min_box_size=1e-3, max_box_size=0.5, num_sizes=20):
    """
    使用盒计数法计算分形维数

    参数:
    points - 形状为 (N, 2) 的点集
    min_box_size - 最小盒子尺寸
    max_box_size - 最大盒子尺寸
    num_sizes - 使用的盒子尺寸数量

    返回:
    dimension - 分形维数估计值
    log_scales - 对数尺度数组
    log_counts - 对数计数数组
    """
    # 提取坐标
    x, y = points[:, 0], points[:, 1]

    # 计算点集边界
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)

    # 生成对数尺度的盒子尺寸数组
    box_sizes = np.logspace(np.log10(min_box_size),
                            np.log10(max_box_size),
                            num_sizes)

    log_scales = np.log(1 / box_sizes)
    log_counts = np.zeros_like(log_scales)

    for i, size in enumerate(box_sizes):
        # 创建网格
        x_bins = np.arange(x_min, x_max + size, size)
        y_bins = np.arange(y_min, y_max + size, size)

        # 初始化盒子计数器
        box_grid = np.zeros((len(x_bins) - 1, len(y_bins) - 1), dtype=bool)

        # 将点分配到盒子中
        x_indices = np.digitize(x, x_bins) - 1
        y_indices = np.digitize(y, y_bins) - 1

        # 标记包含点的盒子
        valid = (x_indices >= 0) & (x_indices < len(x_bins) - 1) & \
                (y_indices >= 0) & (y_indices < len(y_bins) - 1)

        # 更新盒子状态
        np.add.at(box_grid, (x_indices[valid], y_indices[valid]), True)

        # 计算非空盒子数量
        non_empty = np.sum(box_grid)
        log_counts[i] = np.log(non_empty)

    # 线性回归估计分形维数
    slope, intercept, r_value, p_value, std_err = linregress(log_scales, log_counts)

    return slope, log_scales, log_counts


# 主程序
def main():
    # 参数设置
    a, b = 1.4, 0.3  # 经典混沌参数
    n_points = 50000  # 吸引子点数（减少点数以加快速度）

    print(f"生成 Henon 吸引子 (a={a}, b={b})...")
    x, y = generate_henon_attractor(n_points, a=a, b=b)
    points = np.column_stack((x, y))

    print("使用盒计数法计算分形维数...")
    dimension, log_scales, log_counts = box_counting_dimension(points)

    print(f"估计的分形维数: D = {dimension:.4f}")

    # 可视化结果
    plt.figure(figsize=(15, 10))

    # 1. 吸引子可视化
    plt.subplot(221)
    plt.plot(x, y, ',', markersize=0.1, alpha=0.7)
    plt.title(f"Henon 吸引子 (a={a}, b={b})", fontsize=14)
    plt.xlabel("x", fontsize=12)
    plt.ylabel("y", fontsize=12)
    plt.grid(alpha=0.3)

    # 2. 盒计数法结果
    plt.subplot(222)
    plt.scatter(log_scales, log_counts, c='red', s=50)
    fit_line = dimension * log_scales + np.mean(log_counts - dimension * log_scales)
    plt.plot(log_scales, fit_line, 'b--', label=f'拟合直线 (D={dimension:.4f})')
    plt.title("盒计数法结果", fontsize=14)
    plt.xlabel("log(1/ε)", fontsize=12)
    plt.ylabel("log(N(ε))", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)

    # 添加拟合方程
    equation = f"log(N) = {dimension:.4f} * log(1/ε) + {fit_line[0]:.4f}"
    plt.annotate(equation, xy=(0.05, 0.95), xycoords='axes fraction',
                 fontsize=12, backgroundcolor='white')

    # 3. 盒子尺寸示例 (中等尺寸)
    plt.subplot(223)
    plt.plot(x, y, ',', markersize=0.1, alpha=0.3)
    box_size = 0.05
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)

    # 添加网格
    for x_line in np.arange(x_min, x_max, box_size):
        plt.axvline(x_line, color='gray', alpha=0.2, lw=0.5)
    for y_line in np.arange(y_min, y_max, box_size):
        plt.axhline(y_line, color='gray', alpha=0.2, lw=0.5)

    plt.title(f"盒子尺寸示例 (ε={box_size})", fontsize=14)
    plt.xlabel("x", fontsize=12)
    plt.ylabel("y", fontsize=12)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.grid(alpha=0.3)

    # 4. 盒子尺寸示例 (小尺寸)
    plt.subplot(224)
    plt.plot(x, y, ',', markersize=0.1, alpha=0.3)
    small_box_size = 0.01

    # 添加网格
    for x_line in np.arange(x_min, x_max, small_box_size):
        plt.axvline(x_line, color='gray', alpha=0.2, lw=0.3)
    for y_line in np.arange(y_min, y_max, small_box_size):
        plt.axhline(y_line, color='gray', alpha=0.2, lw=0.3)

    plt.title(f"小盒子尺寸示例 (ε={small_box_size})", fontsize=14)
    plt.xlabel("x", fontsize=12)
    plt.ylabel("y", fontsize=12)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('henon_fractal_dimension.png', dpi=300, bbox_inches='tight')
    print("图表已保存为 'henon_fractal_dimension.png'")
    plt.show()

    # 附加分析：不同参数下的分形维数
    print("\n不同参数下的分形维数分析:")
    param_values = [
        (1.0, 0.3, "周期区域"),
        (1.2, 0.3, "过渡区域"),
        (1.4, 0.3, "混沌区域"),
        (1.4, 0.1, "小b值")
    ]

    plt.figure(figsize=(10, 8))

    for i, (a_val, b_val, label) in enumerate(param_values):
        # 生成吸引子
        print(f"生成参数 a={a_val}, b={b_val} 的吸引子...")
        x, y = generate_henon_attractor(30000, a=a_val, b=b_val)
        points = np.column_stack((x, y))

        # 计算分形维数
        dim, log_scales, log_counts = box_counting_dimension(points)

        # 绘制结果
        plt.plot(log_scales, log_counts, 'o-', markersize=5,
                 label=f'a={a_val}, b={b_val}: D={dim:.4f} ({label})')

        print(f"参数 a={a_val}, b={b_val}: 分形维数 = {dim:.4f} ({label})")

    plt.title("不同参数下的分形维数比较", fontsize=16)
    plt.xlabel("log(1/ε)", fontsize=14)
    plt.ylabel("log(N(ε))", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('henon_parameters_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import rcParams
from scipy.stats import linregress
import platform
import os

# 打印系统信息
print("=" * 60)
print(f"操作系统: {platform.system()} {platform.release()}")
print(f"Python 版本: {platform.python_version()}")
print(f"Matplotlib 版本: {mpl.__version__}")  # 使用 mpl 获取版本
print("=" * 60)


# 1. 系统字体检测与设置
def setup_fonts():
    """配置系统字体以确保正确显示"""
    # 尝试设置支持中文的字体
    chinese_fonts = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'SimSun', 'WenQuanYi Micro Hei',
                     'Arial Unicode MS', 'Noto Sans CJK SC', 'STKaiti', 'STSong']

    # 获取系统所有可用字体
    all_fonts = [f.name for f in fm.fontManager.ttflist]
    print(f"检测到系统安装字体数量: {len(all_fonts)}")

    # 查找推荐字体
    available_fonts = []
    for font in chinese_fonts:
        if any(f.lower() == font.lower() for f in all_fonts):
            available_fonts.append(font)

    # 设置字体
    if available_fonts:
        print(f"找到可用中文字体: {', '.join(available_fonts)}")
        # 使用第一个找到的字体
        selected_font = available_fonts[0]
        rcParams['font.family'] = 'sans-serif'
        rcParams['font.sans-serif'] = [selected_font]
        print(f"已设置字体: {selected_font}")
    else:
        print("警告: 未找到推荐的中文字体，尝试使用默认字体")
        # 打印前20个可用字体供参考
        print("前20个可用字体:")
        for i, font in enumerate(all_fonts[:20]):
            print(f"{i + 1}. {font}")

    # 设置其他字体参数
    rcParams['axes.unicode_minus'] = False  # 正确显示负号
    rcParams['font.size'] = 12
    print("=" * 60)


# 配置字体
setup_fonts()


# 2. Henon 映射函数
def henon_map(x, y, a=1.4, b=0.3):
    """计算单次 Henon 映射迭代"""
    x_next = 1 - a * x ** 2 + y
    y_next = b * x
    return x_next, y_next


# 3. 生成吸引子点集
def generate_henon_attractor(n_points=100000, transients=1000, a=1.4, b=0.3):
    """生成 Henon 吸引子点集"""
    # 初始化数组
    x_vals = np.zeros(n_points + transients)
    y_vals = np.zeros(n_points + transients)

    # 初始条件
    x_vals[0], y_vals[0] = 0.1, 0.1

    # 迭代生成轨迹
    for i in range(1, n_points + transients):
        x_vals[i], y_vals[i] = henon_map(x_vals[i - 1], y_vals[i - 1], a, b)

    # 丢弃瞬态部分
    return x_vals[transients:], y_vals[transients:]


# 4. 盒计数法计算分形维数
def box_counting_dimension(points, min_box_size=1e-3, max_box_size=0.5, num_sizes=20):
    """
    使用盒计数法计算分形维数

    参数:
    points - 形状为 (N, 2) 的点集
    min_box_size - 最小盒子尺寸
    max_box_size - 最大盒子尺寸
    num_sizes - 使用的盒子尺寸数量

    返回:
    dimension - 分形维数估计值
    log_scales - 对数尺度数组
    log_counts - 对数计数数组
    """
    # 提取坐标
    x, y = points[:, 0], points[:, 1]

    # 计算点集边界
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)

    # 生成对数尺度的盒子尺寸数组
    box_sizes = np.logspace(np.log10(min_box_size),
                            np.log10(max_box_size),
                            num_sizes)

    log_scales = np.log(1 / box_sizes)
    log_counts = np.zeros_like(log_scales)

    for i, size in enumerate(box_sizes):
        # 创建网格
        x_bins = np.arange(x_min, x_max + size, size)
        y_bins = np.arange(y_min, y_max + size, size)

        # 初始化盒子计数器
        box_grid = np.zeros((len(x_bins) - 1, len(y_bins) - 1), dtype=bool)

        # 将点分配到盒子中
        x_indices = np.digitize(x, x_bins) - 1
        y_indices = np.digitize(y, y_bins) - 1

        # 标记包含点的盒子
        valid = (x_indices >= 0) & (x_indices < len(x_bins) - 1) & \
                (y_indices >= 0) & (y_indices < len(y_bins) - 1)

        # 更新盒子状态
        np.add.at(box_grid, (x_indices[valid], y_indices[valid]), True)

        # 计算非空盒子数量
        non_empty = np.sum(box_grid)
        log_counts[i] = np.log(non_empty)

    # 线性回归估计分形维数
    slope, intercept, r_value, p_value, std_err = linregress(log_scales, log_counts)

    return slope, log_scales, log_counts


# 主程序
def main():
    # 参数设置
    a, b = 1.4, 0.3  # 经典混沌参数
    n_points = 50000  # 吸引子点数（减少点数以加快速度）

    print(f"生成 Henon 吸引子 (a={a}, b={b})...")
    x, y = generate_henon_attractor(n_points, a=a, b=b)
    points = np.column_stack((x, y))

    print("使用盒计数法计算分形维数...")
    dimension, log_scales, log_counts = box_counting_dimension(points)

    print(f"估计的分形维数: D = {dimension:.4f}")

    # 可视化结果
    plt.figure(figsize=(15, 10))

    # 1. 吸引子可视化
    plt.subplot(221)
    plt.plot(x, y, ',', markersize=0.1, alpha=0.7)
    plt.title(f"Henon 吸引子 (a={a}, b={b})", fontsize=14)
    plt.xlabel("x", fontsize=12)
    plt.ylabel("y", fontsize=12)
    plt.grid(alpha=0.3)

    # 2. 盒计数法结果
    plt.subplot(222)
    plt.scatter(log_scales, log_counts, c='red', s=50)
    fit_line = dimension * log_scales + np.mean(log_counts - dimension * log_scales)
    plt.plot(log_scales, fit_line, 'b--', label=f'拟合直线 (D={dimension:.4f})')
    plt.title("盒计数法结果", fontsize=14)
    plt.xlabel("log(1/ε)", fontsize=12)
    plt.ylabel("log(N(ε))", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)

    # 添加拟合方程
    equation = f"log(N) = {dimension:.4f} * log(1/ε) + {fit_line[0]:.4f}"
    plt.annotate(equation, xy=(0.05, 0.95), xycoords='axes fraction',
                 fontsize=12, backgroundcolor='white')

    # 3. 盒子尺寸示例 (中等尺寸)
    plt.subplot(223)
    plt.plot(x, y, ',', markersize=0.1, alpha=0.3)
    box_size = 0.05
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)

    # 添加网格
    for x_line in np.arange(x_min, x_max, box_size):
        plt.axvline(x_line, color='gray', alpha=0.2, lw=0.5)
    for y_line in np.arange(y_min, y_max, box_size):
        plt.axhline(y_line, color='gray', alpha=0.2, lw=0.5)

    plt.title(f"盒子尺寸示例 (ε={box_size})", fontsize=14)
    plt.xlabel("x", fontsize=12)
    plt.ylabel("y", fontsize=12)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.grid(alpha=0.3)

    # 4. 盒子尺寸示例 (小尺寸)
    plt.subplot(224)
    plt.plot(x, y, ',', markersize=0.1, alpha=0.3)
    small_box_size = 0.01

    # 添加网格
    for x_line in np.arange(x_min, x_max, small_box_size):
        plt.axvline(x_line, color='gray', alpha=0.2, lw=0.3)
    for y_line in np.arange(y_min, y_max, small_box_size):
        plt.axhline(y_line, color='gray', alpha=0.2, lw=0.3)

    plt.title(f"小盒子尺寸示例 (ε={small_box_size})", fontsize=14)
    plt.xlabel("x", fontsize=12)
    plt.ylabel("y", fontsize=12)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('henon_fractal_dimension.png', dpi=300, bbox_inches='tight')
    print("图表已保存为 'henon_fractal_dimension.png'")
    plt.show()

    # 附加分析：不同参数下的分形维数
    print("\n不同参数下的分形维数分析:")
    param_values = [
        (1.0, 0.3, "周期区域"),
        (1.2, 0.3, "过渡区域"),
        (1.4, 0.3, "混沌区域"),
        (1.4, 0.1, "小b值")
    ]

    plt.figure(figsize=(10, 8))

    for i, (a_val, b_val, label) in enumerate(param_values):
        # 生成吸引子
        print(f"生成参数 a={a_val}, b={b_val} 的吸引子...")
        x, y = generate_henon_attractor(30000, a=a_val, b=b_val)
        points = np.column_stack((x, y))

        # 计算分形维数
        dim, log_scales, log_counts = box_counting_dimension(points)

        # 绘制结果
        plt.plot(log_scales, log_counts, 'o-', markersize=5,
                 label=f'a={a_val}, b={b_val}: D={dim:.4f} ({label})')

        print(f"参数 a={a_val}, b={b_val}: 分形维数 = {dim:.4f} ({label})")

    plt.title("不同参数下的分形维数比较", fontsize=16)
    plt.xlabel("log(1/ε)", fontsize=14)
    plt.ylabel("log(N(ε))", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('henon_parameters_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()