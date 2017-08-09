package logistic;

public class Model {
	//类型
	public SolverType type;
	//线性回归参数
	public double[] W;
	//偏置
	public double bias;
	//分类模型参数 one vs all
	public double[][] W_C;
	//分类的label
	public int[] labels;
	
	public void addWC(double[] values, int row) {
		if (values.length != W_C[0].length) {
			System.out.println("addWC error !");
			System.exit(1);
		}
		for (int i = 0; i < values.length; i++)
			W_C[row][i] = values[i];
	}
}
