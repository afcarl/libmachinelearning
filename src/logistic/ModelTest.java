package logistic;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class ModelTest {
	private String f_train = "D://projectdata//libself//mnist.scale.train";
	private String f_test = "D://projectdata//libself//mnist.scale.test";
	private String f_model = "D://projectdata//libself//mnist.scale.model";
//	private String f_train = "D://projectdata//libself//covtype.libsvm.binary.scale.train";
//	private String f_test = "D://projectdata//libself//covtype.libsvm.binary.scale.test";
//	private String f_model = "D://projectdata//libself//covtype.libsvm.binary.scale.model";
//	private String f_train = "D://projectdata//libself//abalone.txt.train";
//	private String f_test = "D://projectdata//libself//abalone.txt.test";
//	private String f_model = "D://projectdata//libself//abalone.txt.model";

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
//		param.type = SolverType.L2R_CE_C;
//		param.n = 800;
//		param.eps = 0.0001;
//		param.C = 1;
		param.type = SolverType.SVM_PRI;
		param.n = 800;
		param.C = 1;
		Classifier lr = Classification.getInstance(param);
		Model model = lr.train(x, y, param);
		lr.dump(model, f_model);

		Object[] tobjects = readIn(f_test);
		FeatureNode[][] tx = (FeatureNode[][]) tobjects[0];
		double[] ty = (double[]) tobjects[1];
		double r = 0;
		for (int i = 0; i < tx.length; i++) {
			int p = (int) lr.predict(model, tx[i]);
			if (p == (int) ty[i])
				r++;
		}
		double precision = r / (double) tx.length;
		System.out.println("precision:" + precision);
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
			if (c++ > 1000)
				break;
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
		ModelTest lrTest = new ModelTest();
		lrTest.run();
	}
}
