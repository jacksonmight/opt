import numpy as np
import scipy.linalg as la
import time

def riemannian_gradient(X, A):
    """计算黎曼梯度"""
    euclidean_grad = A @ X
    Xt_grad = X.T @ euclidean_grad
    sym_term = (Xt_grad + Xt_grad.T) / 2
    return euclidean_grad - X @ sym_term

def retraction_QR(X, V):
    """基于QR分解的收缩映射"""
    Z = X + V
    Q, _ = la.qr(Z, mode='economic')
    return Q

def f(X, A):
    """目标函数: f(X) = 0.5 * tr(X^T A X)"""
    return 0.5 * np.trace(X.T @ A @ X)

def strong_wolfe(f, X, grad, p, A, c1=1e-4, c2=0.9, max_iters=20):
    """满足强Wolfe条件的线搜索"""
    alpha = 1.0
    alpha_min = 0.0
    alpha_max = 1e3
    f0 = f(X, A)
    grad_p = np.sum(grad * p)  # Frobenius内积
    
    for _ in range(max_iters):
        X_new = retraction_QR(X, alpha * p)
        f_alpha = f(X_new, A)
        grad_new = riemannian_gradient(X_new, A)
        grad_new_p = np.sum(grad_new * p)
        
        # Armijo条件检查
        if f_alpha > f0 + c1 * alpha * grad_p:
            alpha_max = alpha
            alpha = (alpha_min + alpha_max) / 2
            continue
        
        # 曲率条件检查
        if np.abs(grad_new_p) <= c2 * np.abs(grad_p):
            return alpha
        
        if grad_new_p < 0:
            alpha_min = alpha
            alpha = min(2 * alpha, (alpha_min + alpha_max) / 2)
        else:
            alpha_max = alpha
            alpha = (alpha_min + alpha_max) / 2
    
    return alpha

def lbfgs_double_loop(grad, S, Y, H0):
    """L-BFGS双循环递归算法"""
    q = grad.copy()
    alpha_list = []
    
    # 第一个循环：从最新到最旧的向量对
    for i in range(len(S)-1, -1, -1):
        s_i = S[i]
        y_i = Y[i]
        rho_i = 1.0 / np.sum(s_i * y_i)
        alpha_i = rho_i * np.sum(s_i * q)
        q -= alpha_i * y_i
        alpha_list.append(alpha_i)
    
    r = H0 * q
    
    # 第二个循环：从最旧到最新的向量对
    for i in range(len(S)):
        s_i = S[i]
        y_i = Y[i]
        rho_i = 1.0 / np.sum(s_i * y_i)
        beta = rho_i * np.sum(y_i * r)
        r += s_i * (alpha_list.pop() - beta)
    
    return -r

def stiefel_lbfgs(A, n, p, m=5, max_iter=1000, tol=1e-6):
    """Stiefel流形上的L-BFGS算法"""
    # 初始化随机Stiefel点
    X = np.random.randn(n, p)
    X, _ = la.qr(X, mode='economic')
    
    S = []  # 存储向量对 s_i = X_{k+1} - X_k
    Y = []  # 存储向量对 y_i = grad_{k+1} - grad_k
    
    grad = riemannian_gradient(X, A)
    grad_norm = la.norm(grad, 'fro')
    
    results = {
        'time': 0.0,
        'iter': 0,
        'grad_norm': [],
        'f_value': []
    }
    
    start_time = time.time()
    
    for k in range(max_iter):
        if grad_norm < tol:
            break
        
        # 计算L-BFGS方向
        if k == 0 or len(S) == 0:
            p_k = -grad
            H0 = 1.0  # 初始Hessian近似
        else:
            # 使用最近向量对计算初始Hessian近似
            s_prev = S[-1]
            y_prev = Y[-1]
            H0 = np.sum(s_prev * y_prev) / np.sum(y_prev * y_prev)
            p_k = lbfgs_double_loop(grad, S, Y, H0)
        
        # 强Wolfe线搜索
        alpha = strong_wolfe(f, X, grad, p_k, A)
        
        # 收缩映射更新
        X_new = retraction_QR(X, alpha * p_k)
        grad_new = riemannian_gradient(X_new, A)
        
        # 更新向量对
        s_k = X_new - X
        y_k = grad_new - grad
        
        # 曲率条件检查 (简化版阻尼技术)
        sTy = np.sum(s_k * y_k)
        if sTy > 1e-10:
            # 如果曲率条件不满足，应用简单阻尼
            if sTy < 0.25 * np.sum(s_k * s_k):
                theta = 0.75
                r_k = theta * y_k + (1 - theta) * s_k
            else:
                r_k = y_k
            
            S.append(s_k)
            Y.append(r_k)  # 存储修正后的向量对
            if len(S) > m:  # 保留最近m个向量对
                S.pop(0)
                Y.pop(0)
        
        # 更新迭代点
        X, grad = X_new, grad_new
        grad_norm = la.norm(grad, 'fro')
        
        # 记录性能
        results['grad_norm'].append(grad_norm)
        results['f_value'].append(f(X, A))
        results['iter'] = k + 1
    
    results['time'] = time.time() - start_time
    results['final_grad_norm'] = grad_norm
    results['final_f_value'] = f(X, A)
    return results

def generate_spd_matrix(n):
    """生成对称正定矩阵"""
    B = np.random.randn(n, n)
    return B.T @ B + n * np.eye(n)

# 数值实验
dimensions = [(100, 10), (500, 20), (1000, 50)]
results = {}

for n, p in dimensions:
    print(f"Running experiment for (n, p) = ({n}, {p})")
    A = generate_spd_matrix(n)
    
    # 计算理论最小值
    eigvals = np.sort(np.linalg.eigvalsh(A))[:p]
    theoretical_min = 0.5 * np.sum(eigvals)
    
    res = stiefel_lbfgs(A, n, p, m=10, max_iter=1000, tol=1e-6)
    
    # 计算与理论最小值的偏差
    deviation = res['final_f_value'] - theoretical_min
    
    results[(n, p)] = {
        'time': res['time'],
        'iterations': res['iter'],
        'final_grad_norm': res['final_grad_norm'],
        'final_f_value': res['final_f_value'],
        'theoretical_min': theoretical_min,
        'deviation': deviation
    }

# 输出结果
print("\nPerformance Results:")
print("Dimensions (n, p) | Time (s) | Iterations | Final Grad Norm | Final f Value | Theoretical Min | Deviation")
print("-" * 100)
for dim, res in results.items():
    n, p = dim
    print(f"({n:4d}, {p:2d})        | {res['time']:7.4f} | {res['iterations']:9d} | {res['final_grad_norm']:14.6e} | {res['final_f_value']:12.6f} | {res['theoretical_min']:14.6f} | {res['deviation']:10.6f}")