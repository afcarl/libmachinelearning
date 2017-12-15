package logistic;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;

import optimizer.lbfgs.LBFGS;

public class MSE_LR implements Classifier {

	@Override
	public Model train(FeatureNode[][] nodes, double[] y, Parameter param) throws Exception {
		int m = 5;
		boolean diagco = false;
		double[] diag = new double[param.n];
		int[] iprint = new int [2];
		iprint [0] = 1;
		iprint [1] = 1;
		double eps = param.eps;
		double xtol = 1.0e-16;
		int iflag[] = new int[1];
		double[] W = initW(param.n);
		int call = 0;
		while (true) {
			double f = 0;
			double[] g = new double[param.n];
			for (int i = 0; i < nodes.length; i++) {
				double fsubsum = 0;
				for (int j = 0; j < nodes[i].length; j++) {
					FeatureNode node = nodes[i][j];
					fsubsum += W[node.index] * node.value;
				}
				f += Math.pow(fsubsum - y[i], 2);
				for (int j = 0; j < nodes[i].length; j++) {
					FeatureNode node = nodes[i][j];
					g[node.index] += ((fsubsum - y[i]) * nodes[i][j].value);
				}
			}
			f = f / (double) nodes.length;
			for (int i = 0; i < g.length; i++)
				g[i] = g[i] / (double) nodes.length; 
			LBFGS.lbfgs(param.n, m, W, f, g, diagco, diag, iprint, eps, xtol, iflag);
			if (call++ > param.iter || iflag[0] == 0)
				break;
		}
		
		Model model = new Model();
		model.W = W;
		model.type = SolverType.MSE_LR;

		return model;
	}

	@Override
	public void dump(Model model, String filePath) throws IOException {
		BufferedWriter bw = new BufferedWriter(new FileWriter(filePath));
		bw.write(model.type + "\r\n");
		bw.write("W " + model.W.length + "\r\n");
		for (double value : model.W)
			bw.write(value + "\r\n");
		bw.close();
	}
	
	public double predict(Model model, FeatureNode[] x) {
		double fx = 0;
		double[] W = model.W;
		for (FeatureNode node : x)
			fx += (W[node.index] * node.value);
		return fx;
	}

	@Override
	public Model load(String filePath) {
		// TODO Auto-generated method stub
		return null;
	}

	private static double[] initW(int length) {
		double[] W = new double[length];
		for (int i = 0; i < length; i++)
			W[i] = 1.0 / (double) length;
		return W;
	}
}
