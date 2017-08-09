package optimizer.joptimizer;

import com.joptimizer.functions.PDQuadraticMultivariateRealFunction;

public class SimpleQP {
	
	public void doQp() {
		double[][] p = new double[][] {{ 1., 0.4 }, { 0.4, 1. }};
		PDQuadraticMultivariateRealFunction objectiveFunction2 = 
				new PDQuadraticMultivariateRealFunction(p, null, (double) 0.0);
		
	}

	public static void main(String[] args) {
		SimpleQP simpleQP = new SimpleQP();
		simpleQP.doQp();
	}
}
