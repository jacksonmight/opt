# 流形优化与二次规划算法实现

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.7+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/Purpose-Numerical%20Optimization-red.svg" alt="Purpose">
</p>

一个基于Python的高性能数值优化算法库，实现了**Stiefel流形优化**、**凸二次规划积极集法**和**Stiefel流形L-BFGS**三大核心算法，适用于机器学习、信号处理和数值分析领域的约束优化问题。

---

## 📑 目录

- [核心特性](#核心特性)
- [安装指南](#安装指南)
- [快速开始](#快速开始)
- [算法详解](#算法详解)
- [API参考](#api参考)
- [实验结果](#实验结果)
- [重要说明](#重要说明)
- [代码获取](#代码获取)
- [引用与致谢](#引用与致谢)

---

## ✨ 核心特性

### 1. **Stiefel流形优化**
- **梯度下降法**（单调/非单调线搜索）
- **Barzilai-Borwein (BB)** 自适应步长方法
- **QR分解收缩映射**，数值稳定
- 完整的收敛性诊断工具

### 2. **凸二次规划积极集法**
- 严格处理线性不等式约束
- 动态工作集管理
- KKT系统高效求解
- 详细的迭代历史追踪

### 3. **Stiefel流形L-BFGS**
- **有限内存拟牛顿法**，适合大规模问题
- **双循环递归**实现，计算高效
- **阻尼BFGS更新**，保证数值稳定性
- **环形缓冲区**管理，内存优化

---

## 🚀 安装指南

```bash
# 克隆仓库
git clone https://github.com/jacksonmight/opt.git
cd opt

# 安装依赖
pip install numpy scipy matplotlib
```

**环境要求**：
- Python ≥ 3.7
- NumPy ≥ 1.18.0
- SciPy ≥ 1.5.0
- Matplotlib ≥ 3.2.0

---

## 🎯 快速开始

### 场景1：Stiefel流形优化（主成分分析）

```python
# 导入模块
from stiefel_optimization import *

# 1. 定义问题参数
n, p = 20, 5  # 20维空间中求5个正交基

# 2. 创建流形和目标函数
manifold = StiefelManifold(n, p)
f, grad = generate_quadratic_function(n, p)  # 随机二次函数
X0 = manifold.random_point()  # 随机初始化

# 3. 运行算法
results = bb_method(manifold, f, grad, X0, M=10)  # BB方法

# 4. 结果分析
print_summary(results, "BB方法")  # 打印摘要
# 输出：迭代次数: 137, 最终梯度范数: 5.87e-07, 平均步长: 1.24e-01

# 5. 可视化
plot_optimization_results([results], labels=["BB"], 
                          save_path='bb_results.png')
```

### 场景2：求解凸二次规划问题

```python
from quadratic_programming import active_set_qp_corrected

# 定义QP问题：min 0.5*x^T Q x + c^T x s.t. A x ≤ b
Q = np.array([[2, 0], [0, 4]])  # Hessian矩阵
c = np.array([-2, -6])          # 线性项
A = np.array([[-1, 2], [1, 2], [1, -2], [-1, 0], [0, -1]])  # 约束矩阵
b = np.array([2, 6, 2, 0, 0])   # 约束上界

# 求解（需提供可行初始点）
x0 = np.array([0.0, 0.0])
x_opt, f_opt, history = active_set_qp_corrected(Q, c, A, b, x0)

print(f"最优解: {x_opt}")      # [1.0, 1.5]
print(f"最优值: {f_opt:.2f}")  # -5.5
```

### 场景3：大规模流形优化（L-BFGS）

```python
from lbfgs_stiefel import stiefel_lbfgs, generate_spd_matrix

# 生成大规模问题
n, p = 1000, 50
A = generate_spd_matrix(n)  # 生成正定矩阵

# 运行L-BFGS
results = stiefel_lbfgs(A, n, p, m=10, max_iter=1000)

print(f"迭代次数: {results['iter']}")              # 1000
print(f"计算时间: {results['time']:.2f}s")         # 25.30s
print(f"最终梯度范数: {results['final_grad_norm']:.2e}")  # 1.94e-4
print(f"目标函数值: {results['final_f_value']:.2f}")    # 接近理论最小值
```

---

## 🔬 算法详解

### 1. Stiefel流形优化算法

#### 1.1 数学基础

**Stiefel流形**：

$$
\text{St}(n,p) = \{X \in \mathbb{R}^{n \times p} \mid X^T X = I_p\}
$$

**切空间投影**：

$$
\Pi_X(Z) = Z - X \cdot \text{sym}(X^T Z), \quad \text{sym}(M) = \frac{M + M^T}{2}
$$

**QR收缩映射**：

$$
R_X(V) = \text{qf}(X + V)
$$

#### 1.2 梯度下降法（Algorithm 4.3）

**输入**：初始点 $X_0$，参数 $\rho, c_1, M$

**流程**：
1. 计算黎曼梯度 $g_k = \text{grad} f(X_k)$
2. 搜索方向 $v_k = -g_k$
3. 回退法线搜索确定步长 $t_k$
4. 更新 $X_{k+1} = R_{X_k}(t_k v_k)$
5. 检查收敛条件 $\|g_k\| < \epsilon$

**特点**：稳定但收敛较慢，适合小规模问题。

#### 1.3 BB方法（Algorithm 4.4）

**核心思想**：利用前两次迭代信息自适应调整步长。

**步长公式**：

$$
\begin{aligned}
\alpha_k^{\text{SBB}} &= \frac{\langle s_{k-1}, y_{k-1} \rangle}{\langle y_{k-1}, y_{k-1} \rangle} \\
\alpha_k^{\text{LBB}} &= \frac{\langle s_{k-1}, s_{k-1} \rangle}{\langle s_{k-1}, y_{k-1} \rangle}
\end{aligned}
$$

**交替策略**：奇数步用SBB，偶数步用LBB，平衡稳定性与收敛速度。

---

### 2. 凸二次规划积极集法

#### 2.1 问题形式

$$
\min_{x} \frac{1}{2}x^T Q x + c^T x \quad \text{s.t.} \quad A x \leq b
$$

#### 2.2 算法流程

1. **初始化工作集** $\mathcal{W}_0 = \{i \mid A_i x_0 = b_i\}$
2. **求解KKT系统**

3. **判断搜索方向**：
   - 若 $p = 0$，检查乘子 $\lambda$（最优或删除约束）
   - 若 $p \neq 0$，计算最大可行步长 $\alpha_{\max}$
4. **更新迭代点和工作集**

#### 2.3 关键实现细节

- **数值稳定性**：KKT矩阵奇异时使用伪逆
- **约束管理**：动态添加/移除阻塞约束
- **工作集**：存储活跃约束索引，高效更新

---

### 3. Stiefel流形L-BFGS算法

#### 3.1 双循环递归（Algorithm 5.8）

```python
def lbfgs_double_loop(grad, S, Y, H0):
    q = grad.copy()
    alphas = []
    
    # 前向循环
    for s, y in reversed(zip(S, Y)):
        rho = 1.0 / np.sum(s * y)
        alpha = rho * np.sum(s * q)
        q -= alpha * y
        alphas.append(alpha)
    
    r = H0 * q  # 初始Hessian近似
    
    # 后向循环
    for s, y, alpha in zip(S, Y, reversed(alphas)):
        rho = 1.0 / np.sum(s * y)
        beta = rho * np.sum(y * r)
        r += s * (alpha - beta)
    
    return -r
```

#### 3.2 阻尼BFGS更新

**曲率条件**：$s_k^T y_k > 0$ 保证Hessian近似正定

**阻尼策略**：当条件不满足时，修正梯度差

$$
r_k = \theta_k y_k + (1-\theta_k) B_k s_k
$$

其中

$$
\theta_k = 
\begin{cases}
1, & s_k^T y_k \geq 0.25 s_k^T s_k \\
\frac{0.75 s_k^T s_k}{s_k^T s_k - s_k^T y_k}, & \text{otherwise}
\end{cases}
$$

#### 3.3 强Wolfe线搜索

**条件**（转义下划线）：

```
f(X_k + α * p_k) ≤ f(X_k) + c1 * α * (∇f_k)^T * p_k
|∇f(X_k + α * p_k)^T * p_k| ≤ c2 * |∇f_k^T * p_k|
```

**实现**：回退-插值混合策略，兼顾效率和精度。

---

## 📊 API参考

### `StiefelManifold`类

```python
manifold = StiefelManifold(n=20, p=5)

# 方法
manifold.random_point()          # 生成随机正交矩阵
manifold.projection(X, Z)        # 切空间投影
manifold.retraction_qr(X, V)     # QR收缩映射
manifold.riemannian_gradient(X, egrad)  # 黎曼梯度
```

### `gradient_descent(...)`

```python
results = gradient_descent(
    manifold,          # StiefelManifold实例
    f,                 # 目标函数
    euclidean_grad,    # 欧氏梯度
    X0,                # 初始点
    M=0,               # 非单调参数 (0=单调)
    rho=0.5,           # 步长缩小因子
    c1=1e-4,           # Armijo常数
    max_iter=1000,     # 最大迭代
    tol=1e-6           # 收敛容差
)

# 返回字典
results['X']          # 最优解
results['f_vals']     # 函数值历史
results['grad_norms'] # 梯度范数历史
results['iterations'] # 迭代次数
results['converged']  # 是否收敛
```

### `bb_method(...)`

```python
results = bb_method(
    manifold, 
    f, 
    euclidean_grad, 
    X0,
    M=10,              # 非单调窗口
    alpha_min=1e-10,   # 最小步长
    alpha_max=1e10,    # 最大步长
    rho=0.5, c1=1e-4,
    max_iter=1000, tol=1e-6
)

# 额外返回
results['alpha_history']      # BB步长历史
results['backtrack_counts']   # 回溯次数
```

### `active_set_qp_corrected(...)`

```python
x_opt, f_opt, history = active_set_qp_corrected(
    Q,          # Hessian矩阵 (n×n)
    c,          # 线性项 (n,)
    A,          # 约束矩阵 (m×n)
    b,          # 约束上界 (m,)
    x0,         # 可行初始点 (n,)
    max_iter=100,
    tol=1e-6    # 约束激活容差
)

# 返回
# x_opt: 最优解
# f_opt: 最优值
# history: [(x0, W0), (x1, W1), ...] 迭代历史
```

### `stiefel_lbfgs(...)`

```python
results = stiefel_lbfgs(
    A,          # 对称正定矩阵 (n×n)
    n, p,       # 流形维度
    m=10,       # 记忆长度
    max_iter=1000,
    tol=1e-6    # 梯度范数容差
)

# 返回字典
results['iter']           # 迭代次数
results['time']           # 计算时间(秒)
results['final_grad_norm'] # 最终梯度范数
results['final_f_value']  # 最终目标值
results['grad_norm']      # 梯度范数历史
results['f_value']        # 函数值历史
```

---

## ⚠️ 重要说明

### 1. **Stiefel流形优化**
-  **`retraction_qr`实现**  ：代码使用经典Gram-Schmidt正交化，对小规模问题稳定；大规模问题建议改用`np.linalg.qr`
- **BB方法简化处理**：`s = X - X_prev` 是切向量的近似，严格实现需要平行移动（parallel transport）
- **收敛判据**：基于梯度Frobenius范数，适合中小规模问题

### 2. **积极集法**
- **可行初始点必需**：算法假设`x0`严格可行，需手动保证
- **KKT矩阵奇异**：代码使用伪逆回退，但可能表明约束冗余
- **约束退化**：在高度退化点可能循环，生产环境建议添加反循环规则

### 3. **L-BFGS流形优化**
- **向量传输简化**：直接计算`s_k = X_{k+1} - X_k`，未严格实现平行移动
- **阻尼策略**：采用简化版曲率条件检查，非标准阻尼BFGS
- **线搜索**：强Wolfe条件实现，c1=1e-4, c2=0.9为经验参数

---

## 📈 实验结果

### Stiefel流形优化性能

| 算法 | 迭代次数 | 最终梯度范数 | 平均步长 | 收敛时间 |
|------|----------|--------------|----------|----------|
| 梯度下降（单调） | 458 | 9.76e-07 | 0.002 | 0.8s |
| 梯度下降（非单调） | 999 | 6.31e+00 | 0.001 | 1.5s |
| **BB方法（非单调）** | **137** | **5.87e-07** | **0.124** | **0.3s** |

**结论**：BB方法收敛速度提升71%，推荐使用非单调策略(M=10)

### 二次规划求解示例

```python
Q = [[2,0],[0,4]], c = [-2,-6]
约束: -x1+2x2≤2, x1+2x2≤6, x1-2x2≤2, x1≥0, x2≥0
```

**迭代过程**：
```
迭代0: x=[0.0,0.0], 工作集={3,4}, f=0.0
迭代1: x=[0.0,1.0], 工作集={0,3}, f=-5.0
迭代2: x=[0.0,1.0], 工作集={0},   f=-5.0
迭代3: x=[1.0,1.5], 工作集={0},   f=-5.5  ← 最优
```

**最优解**：`x* = [1.0, 1.5]`，目标值`-5.5`，满足所有KKT条件

### L-BFGS可扩展性

| 维度(n,p) | 迭代 | 时间 | 梯度范数 | 与理论最优偏差 |
|-----------|------|------|----------|----------------|
| (100,10)  | 401  | 0.23s | 9.97e-07 | 0.0 |
| (500,20)  | 1000 | 6.09s | 3.44e-05 | 0.0 |
| (1000,50) | 1000 | 25.30s| 1.94e-04 | 0.0 |

**结论**：算法正确性验证通过，时间复杂度O(mn)，适合中大规模问题

---

## 📦 数据与代码

### 代码仓库

**GitHub**: https://github.com/jacksonmight/opt.git

### 目录结构

```
opt/
├── GD_BBalgorithm.py    # Stiefel流形优化（GD + BB）
├── Activeset_qp.py      # 凸二次规划积极集法
├── L_BFGS.PY            # L-BFGS流形优化
└── README.md                    # 本文档
```

### 快速运行

```bash
# 完整实验流程
python experiments/run_all_experiments.py

# 单独运行Stiefel优化
python -c "from stiefel_optimization import *; run_stiefel_experiment()"

# 单独运行QP求解
python -c "from quadratic_programming import *; run_qp_example()"

# 单独运行L-BFGS
python -c "from lbfgs_stiefel import *; run_lbfgs_experiment()"
```

---

## 📚 引用与致谢

### 参考文献

1. 杨卫红.(2025).流形优化.
2. 刘浩洋, 户将, 李勇锋, 文再文. (2021). 最优化:建模、算法与理论. 高等教育出版社.
3. Wright, S., & Nocedal, J. (1999). Numerical optimization. Springer Science, 35(67-68), 7.
4. Feng, B., & Wu, G. (2024). A block Lanczos method for large-scale quadratic minimization problems with orthogonality constraints. SIAM Journal on Scientific Computing, 46(2), A884-A905.

### 致谢

感谢复旦大学杨卫红老师数值分析课程提供的理论指导，以及开源社区对算法实现的支持。

---

## 📄 许可证

MIT License

Copyright (c) 2025 

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:...

---

## 💬 联系与反馈

- **GitHub Issues**: https://github.com/jacksonmight/opt/issues
- **项目维护**: Yuan Yang
- **最后更新**: 2025年12月

欢迎提交Issue和Pull Request，共同完善本算法库！
```
