import numpy as np

def active_set_qp_corrected(Q, c, A, b, x0, max_iter=100, tol=1e-6):
    """
    凸二次规划积极集法
    """
    n = len(x0)
    x = x0.copy()
    W = set()
    for i in range(len(A)):
        if np.abs(A[i] @ x - b[i]) < tol:
            W.add(i)
    print(f"初始工作集: {W}")

    history = []

    for iter in range(max_iter):
        print(f"\n--- 迭代 {iter} ---")
        print(f"当前点: {x}")
        print(f"当前工作集: {W}")

        A_active = A[list(W)] if W else np.zeros((0, n))
        n_active = len(A_active)

        # 构建KKT矩阵
        KKT_top = np.hstack([Q, A_active.T])
        if n_active > 0:
            KKT_bottom = np.hstack([A_active, np.zeros((n_active, n_active))])
            KKT_matrix = np.vstack([KKT_top, KKT_bottom])
        else:
            # 工作集为空时，KKT系统退化为 Hessian 矩阵
            KKT_matrix = Q

        # 构建右端项
        g = Q @ x + c  # 当前点的梯度
        rhs_top = -g
        rhs_bottom = np.zeros(n_active) if n_active > 0 else np.array([])
        KKT_rhs = np.concatenate([rhs_top, rhs_bottom])

        try:
            # 求解KKT系统
            solution = np.linalg.solve(KKT_matrix, KKT_rhs)
        except np.linalg.LinAlgError:
            print("KKT矩阵奇异，尝试使用伪逆")
            solution = np.linalg.pinv(KKT_matrix) @ KKT_rhs

        p = solution[:n]
        lambda_active = solution[n:] if n_active > 0 else np.array([])

        print(f"搜索方向 p: {p}")
        print(f"工作集乘子 λ: {lambda_active}")

        if np.linalg.norm(p) < tol:
            print("p ≈ 0，检查最优性条件")
            if all(lambda_active >= -tol):  # 所有工作集乘子非负
                print("找到最优解！")
                break
            else:
                # 找到最负的乘子对应的约束索引
                j_min = np.argmin(lambda_active)
                # 将工作集列表化以便索引
                W_list = list(W)
                constraint_to_remove = W_list[j_min]
                print(f"移除约束 {constraint_to_remove}，其乘子为 {lambda_active[j_min]:.6f}")
                W.remove(constraint_to_remove)
                # x 保持不变
        else:
            print("p ≠ 0，计算步长")
            alpha_max = 1.0
            blocking_constraint = None
            for i in range(len(A)):
                if i not in W:
                    a_i_p = A[i] @ p
                    if a_i_p > tol:  # 只考虑会阻挡前进的约束
                        # 计算约束i允许的最大步长
                        a_i_x = A[i] @ x
                        alpha_i = (b[i] - a_i_x) / a_i_p
                        if alpha_i < alpha_max:
                            alpha_max = alpha_i
                            blocking_constraint = i
                            print(f"约束 {i} 是阻塞约束，α_{i} = {alpha_i:.6f}")

            # 更新迭代点
            x_new = x + alpha_max * p
            print(f"步长 α = {alpha_max:.6f}")
            print(f"新点: {x_new}")

            # 判断是否有阻塞约束被激活
            if alpha_max < 1.0 - tol and blocking_constraint is not None:
                print(f"添加阻塞约束 {blocking_constraint} 到工作集")
                W.add(blocking_constraint)
                x = x_new
            else:
                x = x_new
                print("走满步，工作集不变")

        history.append((x.copy(), set(W))) 

    obj_value = 0.5 * x.T @ Q @ x + c.T @ x
    return x, obj_value, history

# ===== 构造数值算例 =====
Q = np.array([[2, 0], [0, 4]])
c = np.array([-2, -6])
A = np.array([[-1, 2], [1, 2], [1, -2], [-1, 0], [0, -1]])
b = np.array([2, 6, 2, 0, 0])

# 选择一个可行的初始点，确保它在可行域内
x0 = np.array([0.0, 0.0])  

print("=== 使用积级集算法进行计算 ===")
x_opt_corrected, f_opt_corrected, history_corrected = active_set_qp_corrected(Q, c, A, b, x0)

print("\n=== 最终结果 ===")
print("最优解:", x_opt_corrected)
print("目标函数值:", f_opt_corrected)
print("\n迭代历史:")
for k, (x, W) in enumerate(history_corrected):
    print(f"迭代 {k}: x = {x}, 工作集 = {W}")