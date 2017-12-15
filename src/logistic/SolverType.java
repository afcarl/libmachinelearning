package logistic;

public enum SolverType {

	/**
	 * 最简单的线性回归
	 * y = W * X
	 * 损失函数是：
	 * loss(W) = (W * X1 - Y1) ^ 2 + ... (W * Xn - Yn) ^ 2
	 * 这里没有使用sigmoid和bias，这个损失函数的推导可以从两种方法考虑：
	 * 1. 直接Min squre Error，非常直观
	 * 2. 从概率论出发，令Y = W * X + epsilon, epsilon满足标准正态分布，通过最大似然估计可以得到跟MSE一样的损失函数表达式
	 */
	MSE_LR,
	
	/**
	 * ridge线性回归
	 * y = W * X
	 * 损失函数是：
	 * loss(W) = (W * X1 - Y1) ^ 2 + ... (W * Xn - Yn) ^ 2 + W1 + ... + Wn
	 * 损失函数在LR_MSE的基础上增加了对W参数的平方约束，两种理解：
	 * 1. 直观理解，认为平方约束是为了限制W的参数浮动过大
	 * 2. 从概率论出发，令Y = W * X + epsilon，epsilon和W分别满足标准正态分布，通过最大后验估计推导得到
	 *    参看https://www.zhihu.com/question/20447622
	 */
	RIDGE_LR,
	
	/**
	 * L2正则化(W1^2 + ... + Wn^2)的分类器, Cross Entropy
	 */
	L2R_CE_C,

	/**
	 * L1正则化(|W1| + ... + |Wn|)的分类器, Cross Entropy
	 */
	L1R_CE_C,
	
	/**
	 * SVM, primal problem分类器
	 */
	SVM_PRI
}
