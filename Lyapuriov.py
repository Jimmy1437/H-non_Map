import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, font_manager
from scipy.linalg import qr
import time
import warnings
import os


# =======================================================
# 解决中文字体显示问题
# =======================================================
# 查找系统中可用的中文字体
def find_chinese_font():
    """查找系统中可用的中文字体"""
    # 常见中文字体列表（按优先级排序）
    chinese_fonts = [
        'Microsoft YaHei',  # Windows 微软雅黑
        'SimHei',  # Windows 黑体
        'KaiTi',  # Windows 楷体
        'FangSong',  # Windows 仿宋
        'STSong',  # Mac 华文宋体
        'STKaiti',  # Mac 华文楷体
        'WenQuanYi Micro Hei',  # Linux 文泉驿微米黑
        'WenQuanYi Zen Hei',  # Linux 文泉驿正黑
        'AR PL UMing CN',  # Linux 文鼎明体
        'Noto Sans CJK SC',  # Google Noto字体
        'Source Han Sans SC',  # Adobe思源黑体
        'SimSun'  # Windows 宋体
    ]

    # 查找已安装的字体
    available_fonts = set(f.name for f in font_manager.fontManager.ttflist)

    # 找到第一个可用的中文字体
    for font in chinese_fonts:
        if font in available_fonts:
            return font

    # 如果找不到，尝试更通用的方法
    for f in font_manager.fontManager.ttflist:
        if any('cjk' in f.name.lower() or 'chinese' in f.name.lower() for f in f.name.split()):
            return f.name

    # 如果还是找不到，返回空
    return None


# 设置全局字体
chinese_font = find_chinese_font()
if chinese_font:
    print(f"使用中文字体: {chinese_font}")
    plt.rcParams['font.family'] = chinese_font
    plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
else:
    print("警告: 未找到中文字体，中文可能显示异常")

# 关闭LaTeX渲染
plt.rcParams['text.usetex'] = False

# 使用Matplotlib内置的数学字体
plt.rcParams['mathtext.fontset'] = 'cm'  # Computer Modern字体

# 设置全局绘图参数
plt.rcParams.update({
    'figure.figsize': (12, 8),
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
})


# =====================================
# 1. Henon映射函数（添加溢出保护）
# =====================================
def henon_map(x, y, a=1.4, b=0.3, max_value=1e10):
    """计算Henon映射的下一步，添加溢出保护"""
    try:
        x_new = 1 - a * x ** 2 + y
        y_new = b * x
    except OverflowError:
        # 处理溢出情况
        return np.nan, np.nan

    # 检查是否发散
    if np.isnan(x_new) or np.isnan(y_new) or abs(x_new) > max_value or abs(y_new) > max_value:
        return np.nan, np.nan

    return x_new, y_new


# =====================================
# 2. Lyapunov指数计算函数（添加安全措施）
# =====================================
def compute_lyapunov(a=1.4, b=0.3, n_iter=10000, trans=1000, max_value=1e10):
    """
    计算Henon映射的Lyapunov指数谱，添加安全措施
    参数:
        a, b: Henon参数
        n_iter: 总迭代次数
        trans: 瞬态迭代次数(丢弃)
        max_value: 最大允许值，防止溢出
    返回:
        lyap: Lyapunov指数列表 [λ₁, λ₂]
        history: 收敛历史记录
        status: 计算状态 (0:成功, 1:发散)
    """
    # 初始化状态和正交基
    x, y = 0.1, 0.1
    Q = np.eye(2)  # 初始正交基

    # 初始化累加器和历史记录
    lyap_sums = np.zeros(2)
    history = []
    status = 0  # 0 = 成功, 1 = 发散

    # 主迭代循环
    for i in range(n_iter):
        # 更新系统状态（带溢出保护）
        x, y = henon_map(x, y, a, b, max_value)

        # 如果发散则提前终止
        if np.isnan(x) or np.isnan(y):
            status = 1
            break

        # 计算雅可比矩阵
        J = np.array([[-2 * a * x, 1],
                      [b, 0]])

        # 应用雅可比矩阵到基向量
        V = J @ Q

        # 检查V是否包含无效值
        if np.any(np.isnan(V)) or np.any(np.isinf(V)):
            status = 1
            break

        # QR分解
        try:
            Q, R = qr(V)
        except:
            status = 1
            break

        # 更新累加器(跳过瞬态)
        if i > trans:
            diag_logs = np.log(np.abs(np.diag(R)))
            lyap_sums += diag_logs
            current_lyap = lyap_sums / (i - trans)
            history.append(current_lyap.copy())

    # 处理发散情况
    if status == 1 or len(history) == 0:
        return np.array([np.nan, np.nan]), np.array([]), status

    # 最终Lyapunov指数值
    lyap = history[-1] if history else np.array([np.nan, np.nan])
    return lyap, np.array(history), status


# =====================================
# 3. 参数空间扫描函数（带安全措施）
# =====================================
def scan_parameter_space(a_range, b_range, n_iter=5000, trans=1000):
    """
    扫描参数空间计算最大Lyapunov指数
    返回:
        lyap_grid: 最大Lyapunov指数网格
        status_grid: 状态网格 (0=成功, 1=发散)
        computation_time: 计算时间
    """
    start_time = time.time()
    lyap_grid = np.zeros((len(a_range), len(b_range)))
    status_grid = np.zeros((len(a_range), len(b_range)), dtype=int)

    for i, a in enumerate(a_range):
        for j, b in enumerate(b_range):
            # 计算最大Lyapunov指数
            lyap, _, status = compute_lyapunov(a, b, n_iter, trans)
            lyap_grid[i, j] = lyap[0]  # 取最大指数
            status_grid[i, j] = status

            # 打印进度
            progress = (i * len(b_range) + j) / (len(a_range) * len(b_range))
            if progress % 0.05 < 0.01:  # 每5%进度打印一次
                print(f"进度: {100 * progress:.1f}%", end='\r')

    computation_time = time.time() - start_time
    print(f"\n参数空间扫描完成! 耗时: {computation_time:.2f}秒")
    return lyap_grid, status_grid, computation_time


# =====================================
# 4. 可视化函数 - 使用中文标签
# =====================================
def plot_lyapunov_convergence(history, a, b, status):
    """绘制Lyapunov指数的收敛过程"""
    plt.figure(figsize=(10, 6))

    if status == 1 or len(history) == 0:
        plt.text(0.5, 0.5, '系统发散或计算失败',
                 ha='center', va='center', fontsize=16)
        plt.title(f"Henon映射 (a={a}, b={b}) - 系统发散")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        return

    # 使用mathtext格式的标签
    lambda1 = history[-1, 0]
    lambda2 = history[-1, 1]
    ln_b = np.log(abs(b))

    # 绘制两个指数的收敛曲线
    plt.plot(history[:, 0], 'b-', label=r'$\lambda_1$ (最终值: %.4f)' % lambda1, alpha=0.7)
    plt.plot(history[:, 1], 'r-', label=r'$\lambda_2$ (最终值: %.4f)' % lambda2, alpha=0.7)

    # 添加理论参考线
    plt.axhline(y=ln_b, color='purple', linestyle='--',
                label=r'理论值: $\ln|b|$ = %.4f' % ln_b)
    plt.axhline(y=0, color='k', linestyle=':', alpha=0.5)

    plt.title(r'Henon映射Lyapunov指数收敛过程 ($a=%.1f$, $b=%.1f$)' % (a, b))
    plt.xlabel("迭代次数 (丢弃前1000次瞬态)")
    plt.ylabel("Lyapunov指数估计值")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 显示理论关系验证
    theory_sum = ln_b
    actual_sum = lambda1 + lambda2
    print("理论关系验证: $\lambda_1 + \lambda_2 = \ln|b|$")
    print(f"实际计算值: {actual_sum:.6f}")
    print(f"理论值: {theory_sum:.6f}")
    print(f"绝对误差: {abs(actual_sum - theory_sum):.2e}")


def plot_parameter_space(a_range, b_range, lyap_grid, status_grid):
    """绘制参数空间的Lyapunov指数热力图"""
    plt.figure(figsize=(12, 8))

    # 创建网格
    A, B = np.meshgrid(a_range, b_range)

    # 创建掩码，隐藏发散区域
    masked_lyap = np.ma.masked_where(np.isnan(lyap_grid.T) | (status_grid.T == 1), lyap_grid.T)

    # 绘制热力图
    contour = plt.contourf(A, B, masked_lyap, 20, cmap=cm.coolwarm)

    # 添加颜色条
    cbar = plt.colorbar(contour)
    cbar.set_label(r'最大Lyapunov指数 ($\lambda_1$)')

    # 绘制混沌边界 (λ₁=0)
    try:
        zero_contour = plt.contour(A, B, masked_lyap, levels=[0], colors='black', linewidths=2)
        plt.clabel(zero_contour, inline=True, fontsize=10, fmt=r'$\lambda_1=0$')
    except:
        pass

    # 标记发散区域
    if np.any(status_grid == 1):
        plt.contourf(A, B, status_grid.T, levels=[0.5, 1.5], colors=['none', 'gray'], alpha=0.3,
                     hatches=['', '//'], label='发散区域')

    # 标记特殊点
    plt.plot(1.4, 0.3, 'ro', markersize=10, label='经典混沌参数')
    plt.plot(0.9, 0.3, 'go', markersize=10, label='周期区域')
    plt.plot(1.0, 0.3, 'yo', markersize=10, label='分岔点')

    plt.title(r"Henon映射参数空间分析 ($\lambda_1$ 热力图)")
    plt.xlabel("参数 a")
    plt.ylabel("参数 b")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()


def plot_attractor_comparison(a_values, b=0.3):
    """比较不同参数下的吸引子结构"""
    plt.figure(figsize=(15, 10))

    for i, a in enumerate(a_values):
        # 生成吸引子
        x, y = 0.1, 0.1
        x_vals, y_vals = [], []
        status = 0

        # 迭代 (去掉瞬态)
        for iter in range(10000):
            x, y = henon_map(x, y, a, b)

            # 检查发散
            if np.isnan(x) or np.isnan(y):
                status = 1
                break

            if iter > 1000:
                x_vals.append(x)
                y_vals.append(y)

        # 计算Lyapunov指数
        if status == 0:
            lyap, _, _ = compute_lyapunov(a, b, n_iter=5000)
            lambda1 = lyap[0] if not np.isnan(lyap[0]) else np.nan
            lambda2 = lyap[1] if not np.isnan(lyap[1]) else np.nan
        else:
            lambda1, lambda2 = np.nan, np.nan

        # 绘制吸引子
        plt.subplot(2, 2, i + 1)

        if status == 0 and len(x_vals) > 0:
            plt.scatter(x_vals, y_vals, s=0.1, alpha=0.5,
                        cmap='viridis', c=np.arange(len(x_vals)))
            plt.title(f"a={a}, b={b}" + "\n" +
                      r"$\lambda_1=" + f"{lambda1:.4f}$" + r", $\lambda_2=" + f"{lambda2:.4f}$")
        else:
            plt.text(0.5, 0.5, '系统发散',
                     ha='center', va='center', fontsize=14)
            plt.title(f"a={a}, b={b}" + "\n系统发散")

        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid(alpha=0.1)

    plt.tight_layout()
    plt.show()


# =====================================
# 5. 主程序（带安全参数）
# =====================================
def main():
    # 经典参数分析
    a_classic, b_classic = 1.4, 0.3
    print("=" * 60)
    print(f"分析经典Henon映射 (a={a_classic}, b={b_classic})")
    print("=" * 60)

    # 计算Lyapunov指数
    lyap, history, status = compute_lyapunov(a_classic, b_classic, n_iter=20000)

    if status == 0 and not np.isnan(lyap[0]):
        lambda1 = lyap[0]
        lambda2 = lyap[1]
        print(f"计算结果: λ₁ = {lambda1:.6f}, λ₂ = {lambda2:.6f}")
        print(f"理论关系验证: λ₁ + λ₂ = {lambda1 + lambda2:.6f}, ln|b| = {np.log(b_classic):.6f}")

        # 绘制收敛过程
        plot_lyapunov_convergence(history, a_classic, b_classic, status)
    else:
        print("计算失败：系统发散或数值不稳定")

    # 比较不同参数下的吸引子
    print("\n比较不同参数下的吸引子结构:")
    plot_attractor_comparison(a_values=[0.9, 1.0, 1.2, 1.4], b=0.3)

    # 参数空间扫描（使用安全范围）
    print("\n扫描参数空间...")
    a_range = np.linspace(0.8, 1.6, 30)  # a参数范围
    b_range = np.linspace(0.1, 0.5, 20)  # b参数范围

    lyap_grid, status_grid, comp_time = scan_parameter_space(a_range, b_range, n_iter=3000)

    # 绘制参数空间
    plot_parameter_space(a_range, b_range, lyap_grid, status_grid)

    # 分析特殊区域
    print("\n特殊区域分析:")
    print("1. 混沌区 (a=1.4, b=0.3): λ₁ > 0")
    print("2. 周期区 (a=0.9, b=0.3): λ₁ < 0")
    print("3. 分岔点 (a=1.0, b=0.3): λ₁ ≈ 0")
    print("4. 灰色区域: 系统发散")


if __name__ == "__main__":
    main()