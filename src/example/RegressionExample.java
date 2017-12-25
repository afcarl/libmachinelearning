package example;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import regression.Regression;
import regression.Regressioner;
import common.FeatureNode;
import common.Parameter;
import common.SolverType;

public class RegressionExample {
	private String f_train = "D://projectdata//libself//abalone.txt.train";
	private String f_test = "D://projectdata//libself//abalone.txt.test";
	private String f_model = "D://projectdata//libself//abalone.txt.model";

	/**
	 * abalone Regression (average square error)
	 * Liblinear(分类而非回归): 1.46, mylib LR_MSE:.1.66, mylib LR_Ridge:1.65
	 * @throws Exception
	 */
	public void run() throws Exception {
		Object[] objects = readIn(f_train);
		FeatureNode[][] x = (FeatureNode[][]) objects[0];
		double[] y = (double[]) objects[1];
		Parameter param = new Parameter();
		param.type = SolverType.RIDGE_LR;
		param.n = 10;
		//使用RIDGE_LR的时候,eps设置过小会报错,很可能跟LBFGS的代码实现有关,其只是提升梯度变化过小,应该提前结束训练
		param.eps = 0.001;
		Regressioner lr = Regression.getInstance(param);
		regression.Model model = lr.train(x, y, param);
		lr.dump(model, f_model);

		Object[] tobjects = readIn(f_test);
		FeatureNode[][] tx = (FeatureNode[][]) tobjects[0];
		double[] ty = (double[]) tobjects[1];
		double sum = 0;
		for (int i = 0; i < tx.length; i++) {
			double p = lr.predict(model, tx[i]);
			sum += ((p - ty[i]) * (p - ty[i]));
		}
		sum = sum / ty.length;
		System.out.println("mean square error:" + sum);
	}
	
	public Object[] readIn(String file) throws IOException {
		BufferedReader br = new BufferedReader(new FileReader(file));
		List<Double> yList = new ArrayList<Double>();
		List<FeatureNode[]> nodeArrayList = new ArrayList<FeatureNode[]>();
		String line = null;
		int c = 0;
		while (null != (line = br.readLine())) {
			String[] items = line.trim().split("\\s+");
			FeatureNode[] nodeArray = new FeatureNode[items.length - 1];
			for (int i = 1; i < items.length; i++) {
				String[] kv = items[i].split(":");
				nodeArray[i - 1] = new FeatureNode(Integer.valueOf(kv[0]), Double.valueOf(kv[1]));
			}
			nodeArrayList.add(nodeArray);
			yList.add(Double.valueOf(items[0]));
		}
		br.close();
		
		FeatureNode[][] x = new FeatureNode[nodeArrayList.size()][];
		for (int i = 0; i < nodeArrayList.size(); i++)
			x[i] = nodeArrayList.get(i);
		double[] y = new double[yList.size()]; 
		for (int i = 0; i < yList.size(); i++)
			y[i] = yList.get(i);
		Object[] object = new Object[2];
		object[0] = x;
		object[1] = y;
		return object;
	}

	public static void main(String[] args) throws Exception {
		RegressionExample lrce = new RegressionExample();
		lrce.run();
	}
}
