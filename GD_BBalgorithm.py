import numpy as np
import scipy.linalg as la
import time
import matplotlib.pyplot as plt

class StiefelManifold:
    def __init__(self, n, p):
        self.n = n
        self.p = p
    
    def random_point(self):
        """生成随机Stiefel流形点"""
        X = np.random.randn(self.n, self.p)
        Q, _ = np.linalg.qr(X)
        return Q
    
    def projection(self, X, Z):
        """切空间投影: Π_X(Z) = Z - X sym(X^T Z) """
        sym = 0.5 * (X.T @ Z + Z.T @ X)
        return Z - X @ sym
    
    def retraction_qr(self, X, V):
        """基于QR分解的收缩映射 (4.1.14) """
        # 经典Gram-Schmidt正交化 
        Q = np.zeros_like(X)
        for j in range(self.p):
            v = X[:, j] + V[:, j]
            for i in range(j):
                r = np.dot(Q[:, i], v)
                v -= r * Q[:, i]
            r = la.norm(v)
            Q[:, j] = v / r
        return Q
    
    def riemannian_gradient(self, X, euclidean_grad):
        """黎曼梯度计算 """
        return self.projection(X, euclidean_grad)

def generate_quadratic_function(n, p):
    """生成Stiefel流形上的随机二次函数 """
    A = np.random.randn(n, n)
    A = A.T @ A + n * np.eye(n)  # 对称正定矩阵
    B = np.random.randn(n, p)
    
    def f(X):
        return 0.5 * np.trace(X.T @ A @ X) + np.trace(B.T @ X)
    
    def euclidean_grad(X):
        return A @ X + B
    
    return f, euclidean_grad

def backtracking_line_search(manifold, f, X, v, grad_fX_v, 
                            t_init=1.0, rho=0.5, c1=1e-4, M=0, 
                            f_history=None):
    """回退法线搜索 (Algorithm 4.2) """
    t = t_init
    # 非单调线搜索条件 (定义4.2.2)
    if M > 0 and f_history is not None and len(f_history) > 0:
        f_ref = max(f_history[-min(len(f_history), M):])
    else:  # 单调线搜索
        f_ref = f(X)
    
    while True:
        X_new = manifold.retraction_qr(X, t * v)
        f_new = f(X_new)
        condition = f_ref + c1 * t * grad_fX_v
        
        if f_new <= condition:
            return t
        t *= rho

def gradient_descent(manifold, f, euclidean_grad, X0, 
                   rho=0.5, c1=1e-4, M=0, max_iter=1000, tol=1e-6):
    """Algorithm 4.3: 梯度下降法 """
    X = X0.copy()
    f_vals = [f(X)]
    grad_norms = []
    backtrack_counts = []
    
    for k in range(max_iter):
        # 计算梯度
        egrad = euclidean_grad(X)
        grad = manifold.riemannian_gradient(X, egrad)
        grad_norm = la.norm(grad, 'fro')
        grad_norms.append(grad_norm)
        
        # 检查收敛条件
        if grad_norm < tol:
            break
        
        # 搜索方向
        v = -grad
        grad_fX_v = np.trace(grad.T @ v)  # <grad f(X), v>
        
        # 线搜索
        backtrack_count = 0
        t = backtracking_line_search(manifold, f, X, v, grad_fX_v, 
                                     rho=rho, c1=c1, M=M, f_history=f_vals)
        backtrack_counts.append(backtrack_count)
        
        # 更新迭代点
        X = manifold.retraction_qr(X, t * v)
        f_vals.append(f(X))
    
    return {
        'X': X, 'f_vals': f_vals, 'grad_norms': grad_norms,
        'backtrack_counts': backtrack_counts, 'iterations': k,
        'converged': grad_norm < tol, 'final_grad_norm': grad_norm
    }

def bb_method(manifold, f, euclidean_grad, X0, 
             rho=0.5, c1=1e-4, M=10, alpha_min=1e-10, 
             alpha_max=1e10, max_iter=1000, tol=1e-6):
    """Algorithm 4.4: 非单调线搜索的BB方法 """
    X = X0.copy()
    f_vals = [f(X)]
    grad_norms = []
    alpha_history = []
    backtrack_counts = []
    
    # 初始梯度计算
    egrad = euclidean_grad(X)
    grad = manifold.riemannian_gradient(X, egrad)
    grad_norm = la.norm(grad, 'fro')
    grad_norms.append(grad_norm)
    
    # 初始化BB步长
    alpha_hat = 1.0
    s_prev, y_prev = None, None
    grad_prev = grad.copy()
    X_prev = X.copy()
    
    for k in range(max_iter):
        if grad_norm < tol:
            break
        
        # 非单调线搜索 (4.2.23)
        alpha = alpha_hat
        backtrack_count = 0
        while True:
            X_new = manifold.retraction_qr(X, -alpha * grad)
            f_new = f(X_new)
            
            # 非单调条件 (定义4.2.2)
            if M > 0 and len(f_vals) > 0:
                f_ref = max(f_vals[-min(len(f_vals), M):])
            else:
                f_ref = f_vals[-1]
                
            condition = f_ref - c1 * alpha * grad_norm**2
            
            if f_new <= condition:
                break
                
            alpha *= rho
            backtrack_count += 1
            if alpha < alpha_min:  # 防止步长过小
                alpha = alpha_min
                break
        
        backtrack_counts.append(backtrack_count)
        alpha_history.append(alpha)
        
        # 保存旧梯度
        grad_prev = grad.copy()
        X_prev = X.copy()
        
        # 更新迭代点
        X = X_new
        f_vals.append(f_new)
        
        # 计算新梯度
        egrad = euclidean_grad(X)
        grad = manifold.riemannian_gradient(X, egrad)
        grad_norm = la.norm(grad, 'fro')
        grad_norms.append(grad_norm)
        
        # 计算BB步长 (简化平行移动) [20](@ref)
        if k >= 1:
            s = X - X_prev  # 简化切向量
            y = grad - grad_prev  # 简化梯度差
            
            # 计算BB步长 (4.4.54)
            sTy = np.trace(s.T @ y)
            if abs(sTy) > 1e-15:  # 避免除以零
                if k % 2 == 1:  # 奇数步用SBB
                    alpha_abb = sTy / np.trace(y.T @ y)
                else:  # 偶数步用LBB
                    alpha_abb = np.trace(s.T @ s) / sTy
            else:
                alpha_abb = alpha_hat  # 保持当前步长
            
            # 截断步长
            alpha_hat = min(alpha_max, max(alpha_min, alpha_abb))
    
    return {
        'X': X, 'f_vals': f_vals, 'grad_norms': grad_norms,
        'alpha_history': alpha_history, 'backtrack_counts': backtrack_counts,
        'iterations': k, 'converged': grad_norm < tol, 
        'final_grad_norm': grad_norm
    }

# 实验参数
n, p = 20, 5  # 减小问题规模
manifold = StiefelManifold(n, p)
f_func, euclidean_grad = generate_quadratic_function(n, p)
X0 = manifold.random_point()

# 运行算法
results_gd_mono = gradient_descent(manifold, f_func, euclidean_grad, X0, M=0)
results_gd_nonmono = gradient_descent(manifold, f_func, euclidean_grad, X0, M=10)
results_bb = bb_method(manifold, f_func, euclidean_grad, X0, M=10)

# 结果分析
def print_summary(results, name):
    print(f"\n===== {name} =====")
    print(f"迭代次数: {results['iterations']}")
    print(f"最终梯度范数: {results['final_grad_norm']:.4e}")
    if 'backtrack_counts' in results:
        avg_backtrack = np.mean(results['backtrack_counts']) if results['backtrack_counts'] else 0
        print(f"平均回溯次数: {avg_backtrack:.2f}")
    if 'alpha_history' in results:
        avg_alpha = np.mean(results['alpha_history']) if results['alpha_history'] else 0
        print(f"平均步长: {avg_alpha:.4e}")

print_summary(results_gd_mono, "梯度下降法 (单调)")
print_summary(results_gd_nonmono, "梯度下降法 (非单调, M=10)")
print_summary(results_bb, "BB方法 (非单调, M=10)")

# 可视化
plt.figure(figsize=(15, 10))

# 函数值下降曲线
plt.subplot(2, 2, 1)
min_f = min(min(results_gd_mono['f_vals']), min(results_gd_nonmono['f_vals']), min(results_bb['f_vals']))
plt.semilogy([f - min_f for f in results_gd_mono['f_vals']], 'b-', label='GD Monotonic')
plt.semilogy([f - min_f for f in results_gd_nonmono['f_vals']], 'g--', label='GD Nonmonotonic (M=10)')
plt.semilogy([f - min_f for f in results_bb['f_vals']], 'r-', label='BB Method (M=10)')
plt.xlabel('Iterations')
plt.ylabel('Function Value (log scale)')
plt.title('Function Value Reduction')
plt.legend()
plt.grid(True)

# 梯度范数下降曲线
plt.subplot(2, 2, 2)
plt.semilogy(results_gd_mono['grad_norms'], 'b-', label='GD Monotonic')
plt.semilogy(results_gd_nonmono['grad_norms'], 'g--', label='GD Nonmonotonic')
plt.semilogy(results_bb['grad_norms'], 'r-', label='BB Method')
plt.xlabel('Iterations')
plt.ylabel('Gradient Norm (log scale)')
plt.title('Gradient Norm Reduction')
plt.legend()
plt.grid(True)

# BB步长变化
plt.subplot(2, 2, 3)
if results_bb['alpha_history']:
    plt.semilogy(results_bb['alpha_history'], 'ro-')
    plt.xlabel('Iterations')
    plt.ylabel('Step Size (log scale)')
    plt.title('BB Method Step Size History')
    plt.grid(True)

# 回溯次数比较
plt.subplot(2, 2, 4)
if results_gd_mono.get('backtrack_counts'):
    plt.plot(results_gd_mono['backtrack_counts'], 'b-', label='GD Monotonic')
if results_gd_nonmono.get('backtrack_counts'):
    plt.plot(results_gd_nonmono['backtrack_counts'], 'g--', label='GD Nonmonotonic')
if results_bb.get('backtrack_counts'):
    plt.plot(results_bb['backtrack_counts'], 'r-', label='BB Method')
plt.xlabel('Iterations')
plt.ylabel('Backtracking Steps')
plt.title('Backtracking Steps per Iteration')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('stiefel_optimization_results.png')
plt.show()