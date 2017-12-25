package svm;

/**
 * SVM二分类模型,对偶问题的模型,其中:
 * w=SUM(alpha[i] * y[i] * x[i]), SUM表示N个训练样本的和
 * 
 * W*X包含了X的内积,这就是kernel引入的地方,X的内积替换成kernelFunction,既可以提高运算速度,也能对X进行升维
 * 
 * @author J.
 *
 */
public class SVMBinaryModel {
	//alpha参数
	public double[] alpha;
	//所有训练数据
	public double[][] x;
	//所有训练数据
	public int[] y;
	//通过alpha/x/y可以推导出w,这里可以不用存储
	public double[] w;
	//b参数
	public double b;
	//二分类模型y>0时对应到的实际分类lable,预测时进行label转换使用
	public int pLabel; //+1代表的实际label
	//二分类模型y<0时对应到的实际分类label,预测时记性label转换使用
	public int nLabel; //-1代表的实际label
	
	public SVMBinaryModel(double b, double[] alpha, double[][] x, int[] y){
		this.b = b;
		this.alpha = alpha;
		this.x = x;
		this.y = y;
	}
	
	public int getpLabel() {
		return pLabel;
	}

	public void setpLabel(int pLabel) {
		this.pLabel = pLabel;
	}

	public int getnLabel() {
		return nLabel;
	}

	public void setnLabel(int nLabel) {
		this.nLabel = nLabel;
	}
}
