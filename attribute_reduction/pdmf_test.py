import numpy as np

from gaussian_pdmf import (
    calculate_membership,
    create_gaussian_pdmf,
    create_gaussian_pdmf_from_column,
    fuzzy_entropy,
    fuzzy_scalar_multiply,
    fuzzy_similarity,
    fuzzy_subtract,
    fuzzify_column_to_array,
    metrics,
)


if __name__ == "__main__":
    # # 1. 先验输入（非对称模糊数："中高收入"）
    # # x0 = 2.0  # 中心值：10万元
    # # p = (1.0, 0.8)  # 左控制点：(8万, 隶属度0.85)
    # # q = (3.0, 0.8)  # 右控制点：(15万, 隶属度0.4)
    # # t_minus = 0.5  # 左比例刻度0.6
    # # t_plus = 0.5  # 右比例刻度 0.3
    # x0 = 10.0  # 中心值：10万元
    # p = (8.0, 0.85)  # 左控制点：(8万, 隶属度0.85)
    # q = (15.0, 0.4)  # 右控制点：(15万, 隶属度0.4)
    # t_minus = 0.6  # 左比例刻度0.6
    # t_plus = 0.3  # 右比例刻度 0.3
    # # GaussianPDMF(x0=10.00, d-=5.00, d+=7.14, mu-=-0.7115, mu+=-0.4732)

    # # 2. 构造模糊数
    # fuzzy_num = create_gaussian_pdmf(x0, p, q, t_minus, t_plus)
    # print("构造的高斯-PDMF模糊数：", fuzzy_num)

    # # 3. 计算典型点隶属度
    # test_points = [5.0, 7.0, 8.0, 10.0, 15.0, 16.0, 17.14]
    # print("\n典型点隶属度计算结果：")
    # for x in test_points:
    #     mu = calculate_membership(fuzzy_num, x)
    #     print(f"x={x:.2f} → 隶属度={mu:.4f}")


    # # 4. 代数运算示例（论文示例1简化版）
    # fuzzy_1 = create_gaussian_pdmf(1.0, (0.5, 0.7), (1.5, 0.7), 0.5, 0.5)
    # fuzzy_3 = create_gaussian_pdmf(3.0, (1.0, 0.7), (5.0, 0.7), 0.5, 0.5)
    # # 求解 2⊗x ⊕ fuzzy_1 = fuzzy_3 → x = (fuzzy_3 ⊖ fuzzy_1) / 2
    # fuzzy_c_minus_b = fuzzy_subtract(fuzzy_3, fuzzy_1)
    # fuzzy_x = fuzzy_scalar_multiply(1/2, fuzzy_c_minus_b)
    # print("\n模糊方程2x ⊕ 1=3的解：", fuzzy_x)



    # a = create_gaussian_pdmf(10.0, (8.0, 0.85), (15.0, 0.4), 0.6, 0.3)
    # b = create_gaussian_pdmf(9.5, (7.5, 0.8), (14.0, 0.45), 0.6, 0.3)

    # print("E(a) =", fuzzy_entropy(a))
    # print("S(a,b) =", fuzzy_similarity(a, b, lam=0.5))

    print("H(0) =", metrics.h_mu(0.0,101,1e-6))
    print("H(-0.7115) =", metrics.h_mu(-0.7115,101,1e-6))
    print("H(-0.4732) =", metrics.h_mu(-0.4732,101,1e-6))
    # H(0)       = 0.3702632683
    # H(-0.7115) = 0.3308847925
    # H(-0.4732) = 0.3521619134

    # # 5. 按 create_d_miu.md：给一组属性数据，自动模糊化
    # data = np.array([2.1, 2.4, 2.8, 3.0, 4.2, 4.5, 6.0], dtype=float)
    # k = 3
    # print("\n原始数据列：", data.tolist())

    # for method in ("A", "B", "C"):
    #     print(f"\n--- 自动模糊化（mu_method={method}, k={k}）---")
    #     fuzzy_nums = create_gaussian_pdmf_from_column(data, k=k, mu_method=method)
    #     for idx, fn in enumerate(fuzzy_nums):
    #         print(
    #             f"i={idx}, x={fn.x0:.4f}, d-={fn.d_minus:.4f}, "
    #             f"d+={fn.d_plus:.4f}, mu-={fn.mu_minus:.4f}, mu+={fn.mu_plus:.4f}"
    #         )

    #     # 便捷矩阵输出：[x, d-, d+, mu-, mu+]
    #     arr = fuzzify_column_to_array(data, k=k, mu_method=method)
    #     print("矩阵输出 shape:", arr.shape)