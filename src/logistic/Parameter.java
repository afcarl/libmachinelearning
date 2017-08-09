package logistic;

public class Parameter {
	/**
	 * C是增加在原始loss上的系数, 正则化项的系数保持为1
	 */
	public double C = 0.5;
	
	/**
	 * 特征数量
	 */
	public int n = 0;
	
	/**
	 * see SolverType
	 */
	public SolverType type;
	
	/**
	 * 结束条件
	 */
	public double eps = 0.01;
	
	/**
	 * 结束条件，超过迭代次数直接结束训练
	 */
	public int iter = 200;
}
