package logistic;

import svm.SVM_PRI;

public class Classification {
	public static Classifier getInstance(Parameter param) {
		if (param.type == SolverType.MSE_LR)
			return new MSE_LR();
		if (param.type == SolverType.RIDGE_LR)
			return new RIDGE_LR();
		if (param.type == SolverType.L2R_CE_C)
			return new L2R_CE_C();
		if (param.type == SolverType.L1R_CE_C)
			return new L1R_CE_C();
		if (param.type == SolverType.SVM_PRI)
			return new SVM_PRI();
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
