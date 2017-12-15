package svm;

import java.io.IOException;
import java.util.HashSet;
import java.util.Set;

import cern.colt.matrix.tdouble.DoubleFactory1D;
import cern.colt.matrix.tdouble.DoubleFactory2D;
import cern.colt.matrix.tdouble.DoubleMatrix1D;
import cern.colt.matrix.tdouble.DoubleMatrix2D;

import com.joptimizer.exception.JOptimizerException;
import com.joptimizer.functions.ConvexMultivariateRealFunction;
import com.joptimizer.functions.LinearMultivariateRealFunction;
import com.joptimizer.functions.PDQuadraticMultivariateRealFunction;
import com.joptimizer.optimizers.JOptimizer;
import com.joptimizer.optimizers.OptimizationRequest;

import logistic.Classifier;
import logistic.FeatureNode;
import logistic.Model;
import logistic.Parameter;

/**
 * 使用Primary进行SVM构建，对偶问题暂时不考虑.
 * 这里实现的是soft-margin svm, 采用QP工具进行求解，参看台大的机器学习技法
 * 使用Joptimizer来求解SVM这个QP问题似乎不行，报错
 */
public class SVM_PRI implements Classifier {

	@Override
	public Model train(FeatureNode[][] nodes, double[] y, Parameter param)
			throws Exception {
		Model model = new Model();
		Set<Integer> yset = new HashSet<Integer>();
		for (int i = 0; i < y.length; i++)
			yset.add((int) y[i]);
		int[] labels = new int[yset.size()]; 
		if (yset.size() == 2)
			model.W_C = new double[yset.size() - 1][param.n + 1];
		else
			model.W_C = new double[yset.size()][param.n + 1];
		int row = 0;
		for (Integer key : yset) {
			labels[row] = key;
			if (yset.size() == 2 && row >= 1)
				break;
			double[] suby = new double[y.length]; 
			for (int i = 0; i < nodes.length; i++) {
				if (((int) y[i]) == key)
					suby[i] = 1;
				else
					suby[i] = -1;
			}

			Model subModel = train_one(nodes, suby, param);
			System.exit(1);
			model.addWC(subModel.W, row++);
		}
		model.labels = labels;
		return null;
	}
	
	private Model train_one(FeatureNode[][] nodes, double[] y, Parameter param) throws JOptimizerException {
		int N = y.length, d = param.n;
		int plen = d + 1 + N; //W(d), b(1), sigma(N)
		DoubleMatrix2D p = DoubleFactory2D.sparse.make(plen, plen);
		for (int i = 0; i < plen; i++)
			p.set(i, i, 1.0);
		DoubleMatrix1D q = DoubleFactory1D.sparse.make(plen);
		for (int i = param.n + 1; i < plen; i++)
			q.set(i, 1.0);
		//X.shape=[d+1+N,1]
		//三个参数分别是P/Q/r
		PDQuadraticMultivariateRealFunction objectiveFunction = 
				new PDQuadraticMultivariateRealFunction(p, q, (double) 0.0);	
		//inequalities(Gx <= h)
		ConvexMultivariateRealFunction[] inequalities = new ConvexMultivariateRealFunction[2 * N];
		int k = 0;
		//G.shape=[N,d+1+N],其中d表示特征维度,N表示训练样本数
		//h.shape=[N,1]
		//G的第一行是x11,...,x1n,1,0,0,...,0
		//G的第二行是x21,...,x2n,0,1,0,...,0 也就是x的输入矩阵append一个单位矩阵
		//不等式约束条件1：Y(WX+b)>=1-SIGMA
		for (int i = 0; i < N; i++) {
			DoubleMatrix1D gvector = DoubleFactory1D.sparse.make(d + 1 + N);
			for (int j = 0; j < nodes[i].length; j++) {
				FeatureNode fn = nodes[i][j];
				gvector.set(fn.index, y[i] * fn.value);
			}
			gvector.set(d, 1.0);//b对应的x输入为1
			//append单位矩阵
			gvector.set(d + 1 + i, 1.0);
			inequalities[k++] = new LinearMultivariateRealFunction(gvector, 1.0);
		}
		//不等式约束条件2：SIGMA>=0
		for (int i = 0; i < N; i++) {
			DoubleMatrix1D gvector = DoubleFactory1D.sparse.make(d + 1 + N);
			gvector.set(d + 1 + i, -1.0);
			inequalities[k++] = new LinearMultivariateRealFunction(gvector, 0);
		}
		//初始化InitialPoint
		double ip[] = new double[d + 1 + N];
		for (int i = 0; i < d + 1 + N; i++)
			ip[i] = 0.01;
		
		OptimizationRequest or = new OptimizationRequest();
		or.setF0(objectiveFunction);
		or.setFi(inequalities);
//		or.setInitialPoint(ip);
		or.setToleranceFeas(1.E-12);
		or.setTolerance(1.E-12);
		JOptimizer opt = new JOptimizer();
		opt.setOptimizationRequest(or);
		System.out.println("go optimize");
		opt.optimize();
		double[] sol = opt.getOptimizationResponse().getSolution();
		
		return null;
	}
	
	@Override
	public void dump(Model model, String filePath) throws IOException {
		// TODO Auto-generated method stub
		
	}

	@Override
	public Model load(String filePath) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public double predict(Model model, FeatureNode[] x) {
		// TODO Auto-generated method stub
		return 0;
	}

}
