package logistic;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.HashSet;
import java.util.Set;

import optimizer.lbfgs.LBFGS;
import optimizer.lbfgs.LBFGS.ExceptionWithIflag;

/**
 * 对分类问题采用one-vs-all的方式训练N-1个二分类模型,预测的时候取分值最大的一个类别
 * 最为最终的分类结果 
 */
public class L2R_CE_C implements Classifier {

	@Override
	public Model train(FeatureNode[][] nodes, double[] y, Parameter param)
			throws Exception {
		Model model = new Model();
		Set<Integer> yset = new HashSet<Integer>();
		for (int i = 0; i < y.length; i++)
			yset.add((int) y[i]);
		int[] labels = new int[yset.size()]; 
		if (yset.size() == 2)
			model.W_C = new double[yset.size() - 1][param.n];
		else
			model.W_C = new double[yset.size()][param.n];
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
					suby[i] = 0;
			}

			Model subModel = train_one(nodes, suby, param);
			model.addWC(subModel.W, row++);
		}
		model.labels = labels;
		return model;
	}

	@Override
	public void dump(Model model, String filePath) throws IOException {
		BufferedWriter bw = new BufferedWriter(new FileWriter(filePath));
		bw.write("L2R_CE_C" + "\r\n");
		bw.write("labels");
		for (int label : model.labels)
			bw.write(" " + String.valueOf(label));
		bw.write("\r\n");
		bw.write("W " + model.W_C.length + " " + model.W_C[0].length + "\r\n");
		//每一列是一个W
		for (int col = 0; col < model.W_C[0].length; col++) {
			StringBuffer sb = new StringBuffer();
			for (int row = 0; row < model.W_C.length; row++)
				sb.append(" " + model.W_C[row][col]);
			bw.write(sb.toString().replaceFirst(" ", "") + "\r\n");
		}
		bw.close();
	}

	@Override
	public Model load(String filePath) {
		// TODO Auto-generated method stub
		return null;
	}

	private static Model train_one(FeatureNode[][] nodes, double[] y, Parameter param) throws Exception {
		int m = 5;
		boolean diagco = false;
		double[] diag = new double[param.n];
		int[] iprint = new int [2];
		iprint [0] = 1;
		iprint [1] = 2;
		double eps = param.eps;
		double xtol = 0.000001;
		int iflag[] = new int[1];
		double[] W = initW(param.n);
		int call = 0;
		while (true) {
			double f = 0;
			double[] g = new double[param.n]; // W and b
			for (int i = 0; i < nodes.length; i++) {
				double p = 0;
				for (int j = 0; j < nodes[i].length; j++)
					p += W[nodes[i][j].index] * nodes[i][j].value;
				p = 1 / (1 + Math.exp(0 - p));
				f += y[i] * Math.log(p) + (1 - y[i]) * Math.log(1 - p);
				for (int j = 0; j < nodes[i].length; j++) {
					g[nodes[i][j].index] += ((y[i] * (1 - p) + (y[i] - 1) * p) * nodes[i][j].value);
				}
			}
			//需要将负数的argmax转换成正数的argmin(f需要是正数,argmin(f)),否则会不收敛, why?
			f = param.C * (0 - f);
			for (int i = 0; i < g.length; i++)
				g[i] = param.C * (0 - g[i]);
			//放在前面,经过f和g除以nodes.length之后,其权重就自然变小了
			for (int i = 0; i < g.length; i++) {
				g[i] += W[i];
				f += (0.5 * W[i] * W[i]);
			}
			f = f / (double) nodes.length;
			for (int i = 0; i < g.length; i++)
				g[i] = g[i] / (double) nodes.length; 
			/* 这个放在后面的话,C需要设置得比较大,来降低L2R的权重,否则W会太小,导致效果不好
			for (int i = 0; i < g.length; i++) {
				g[i] += W[i];
				f += (0.5 * W[i] * W[i]);
			}
			*/
			try {
				LBFGS.lbfgs(param.n, m, W, f, g, diagco, diag, iprint, eps, xtol, iflag);
			} catch (ExceptionWithIflag e) {
				//当iflag==-1 info=3时,表示f变换很小
				//<code>info = 3</code> Number of function evaluations has reached <code>maxfev</code>.
				if (iflag[0] == -1)
					break;
			}
			if (call++ > param.iter || iflag[0] == 0)
				break;
		}
		Model model = new Model();
		model.W = W;

		return model;
	}
	
	private static double[] initW(int length) {
		double[] W = new double[length];
		for (int i = 0; i < length; i++)
			W[i] = 1.0 / (double) length;
		return W;
	}

	//2分类和多分类要区别对待,2分类只需要一个W
	//回归问题，返回的是预测值，分类时返回的是lable
	@Override
	public double predict(Model model, FeatureNode[] x) {
		Integer label_max = null;
		if (model.labels.length == 2) {
			double p = 0;
			for (FeatureNode node : x)
				p += (model.W_C[0][node.index] * node.value);
			p = 1 / (double) (1 + Math.exp(0 - p));
			if (p > 0.5)
				label_max = model.labels[0];
			else
				label_max = model.labels[1];
		} else {
			double p_max = 0;
			for (int i = 0; i < model.W_C.length; i++) {
				double p = 0;
				for (FeatureNode node : x) {
					p += (model.W_C[i][node.index] * node.value);
				}
				p = 1 / (1 + Math.exp(0 - p));
				if (p > p_max) {
					p_max = p;
					label_max = model.labels[i];
				}
			}
		}
		return label_max;
	}
}
