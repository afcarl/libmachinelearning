//package optimizer.joptimizer;
//
//QP工具来求解SVM问题不太现实，耗费的内存太大，改为SMO算法
//import com.joptimizer.exception.JOptimizerException;
//import com.joptimizer.functions.ConvexMultivariateRealFunction;
//import com.joptimizer.functions.LinearMultivariateRealFunction;
//import com.joptimizer.functions.PDQuadraticMultivariateRealFunction;
//import com.joptimizer.optimizers.JOptimizer;
//import com.joptimizer.optimizers.OptimizationRequest;
//
//public class SimpleQP {
//	
//	public void doQp() throws JOptimizerException {
//		// Objective function ( (1/2)^TPx+q^Tx+r)
//		double[][] p = new double[][] {{ 1., 0.4 }, { 0.4, 1. }};
//		//三个参数分别是P/Q/r
//		PDQuadraticMultivariateRealFunction objectiveFunction = 
//				new PDQuadraticMultivariateRealFunction(p, null, (double) 0.0);
//		//equalities (Ax = b)
//		double[][] A = new double[][]{{1,1}};
//		double[] b = new double[]{1};
//		//inequalities(Gx <= h)
//		ConvexMultivariateRealFunction[] inequalities = new ConvexMultivariateRealFunction[2];
//		inequalities[0] = new LinearMultivariateRealFunction(new double[]{-1, 0}, -0.6);
//		inequalities[1] = new LinearMultivariateRealFunction(new double[]{0, -1}, 0);
//		//optimization problem
//		OptimizationRequest or = new OptimizationRequest();
//		or.setF0(objectiveFunction);
////		or.setInitialPoint(new double[] { 0.1, 0.9});
//		or.setFi(inequalities); //if you want x>0 and y>0
////		or.setA(A);
////		or.setB(b);
//		or.setToleranceFeas(1.E-12);
//		or.setTolerance(1.E-12);
//		//optimization
//		JOptimizer opt = new JOptimizer();
//		opt.setOptimizationRequest(or);
//		opt.optimize();
//		
//		double[] sol = opt.getOptimizationResponse().getSolution();
//		System.out.println(sol[0]);
//		System.out.println(sol[1]);
//	}
//
//	public static void main(String[] args) throws JOptimizerException {
//		SimpleQP simpleQP = new SimpleQP();
//		simpleQP.doQp();
//	}
//}
