package example;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import common.FeatureNode;
import common.Parameter;
import common.SolverType;
import logistic.Classification;
import logistic.Classifier;
import logistic.Model;

public class LRClassifcationExample {
	private String f_train = "D://projectdata//libself//mnist.scale.train";
	private String f_test = "D://projectdata//libself//mnist.scale.test";
	private String f_model = "D://projectdata//libself//mnist.scale.model";

	/**
	 * precision:0.9186
	 * @throws Exception
	 */
	public void run() throws Exception {
		Object[] objects = readIn(f_train);
		FeatureNode[][] x = (FeatureNode[][]) objects[0];
		double[] y = (double[]) objects[1];
		Parameter param = new Parameter();
		//L2R_CE_C/L1R_CE_C对应到Classification
		param.type = SolverType.L1R_CE_C;
		param.n = 800;
		param.eps = 0.0001;
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
		LRClassifcationExample lrce = new LRClassifcationExample();
		lrce.run();
	}
}
