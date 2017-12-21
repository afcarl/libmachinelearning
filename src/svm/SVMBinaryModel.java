package svm;

public class SVMBinaryModel {
	public double[] alpha;
	public double[][] x;
	public int[] y;
	public double[] w;
	public double b;
	public int pLabel; //+1代表的实际label
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
