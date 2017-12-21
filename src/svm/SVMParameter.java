package svm;

public class SVMParameter {

	//求解时的最多迭代次数
	public int iter = 10000000;
	
	//终止条件的差值,参看SMO代码
	public double eps = 0.001;
	
	//调整Soft-Margin中的sigma参数的权重,参看SVM的Primary问题的目标函数
	//C越大,越欠拟合(容忍更多的噪声分类点)   C越小,越过拟合(尽量将更多的点进行正确分类,包括噪声点)
	public double C = 0.05;
}
