package regression;

import common.FeatureNode;
import common.Parameter;
import common.SolverType;

public class Regression {
	public static Regressioner getInstance(Parameter param) {
		if (param.type == SolverType.MSE_LR)
			return new MSE_LR();
		if (param.type == SolverType.RIDGE_LR)
			return new RIDGE_LR();
		return null;
	}

	public static void predictProbability(Model model, FeatureNode[] x, double[] prob_estimate) {
		double fx = 0;
		double[] W = model.W;
		for (FeatureNode node : x)
			fx += (W[node.index] * node.value);
		prob_estimate[0] = fx;
	}
	
	public static double predict(Model model, FeatureNode[] x) {
		double fx = 0;
		double[] W = model.W;
		for (FeatureNode node : x)
			fx += (W[node.index] * node.value);
		return fx;
	}
}
