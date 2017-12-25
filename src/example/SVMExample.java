package example;

import java.io.IOException;

import common.FeatureNode;

import svm.SVMModel;
import svm.SVMParameter;
import svm.SVM_dual;
import libsvm.svm_model;
import libsvm.svm_node;
import libsvm.svm_parameter;
import libsvm.svm_problem;

/**
 * 对比libsvm Linear Kernel的效果
 * train one svm
 * 测试准确率 : 0.8518518518518519
 * optimization finished, #iter = 555
 * nu = 0.3636883201231288
 * obj = -47.23945210918802, rho = -0.9862072311222366
 * nSV = 103, nBSV = 91
 * Total nSV = 103
 * 测试准确率 : 0.8518518518518519
 * @author J.
 *
 */
public class SVMExample {
//	private String f_train = "D://projectdata//libself//mnist.scale.train";
	private String f_train = "D:/projectdata/libself/heart_scale.txt";
	
	public void classify() throws IOException {
		DataReader dataReader = new DataReader();
		Object[] objects = dataReader.readIn(f_train);
		FeatureNode[][] x = (FeatureNode[][]) objects[0];
		int dimension = (int) objects[2];
		double[][] x1 = new double[x.length][dimension];
		for (int i = 0; i < x.length; i++) {
			for (int j = 0; j < x[i].length; j++) {
				int index = x[i][j].index - 1;
				x1[i][index] = x[i][j].value;
			}
		}
		double[] y = (double[]) objects[1];
		int[] y1 = new int[y.length];
		for (int i = 0; i < y.length; i++)
			y1[i] = (int) y[i];
		SVMParameter parameter = new SVMParameter();
		parameter.C = 0.05;
		parameter.eps = 0.001;
		SVM_dual svm = new SVM_dual();
		SVMModel model = svm.train(x1, y1, parameter);
		int correct = 0;
		for (int i = 0; i < x1.length; i++) {
			int py = svm.predict(model, x1[i]);
			if (py == y1[i])
				correct++;
		}
		System.out.println("测试准确率 : " + (correct / (double) x1.length));
	}
	
	/**
	 * libsvm分类效果
	 * @throws IOException
	 */
	public void libsvmClassify() throws IOException {
		DataReader dataReader = new DataReader();
		Object[] objects = dataReader.readIn(f_train);
		FeatureNode[][] x = (FeatureNode[][]) objects[0];
		svm_node[][] x1 = new svm_node[x.length][];
		for (int i = 0; i < x.length; i++) {
			x1[i] = new svm_node[x[i].length];
			for (int j = 0; j < x[i].length; j++) {
				svm_node node = new svm_node();
				node.index = x[i][j].index;
				node.value = x[i][j].value;
				x1[i][j] = node;
			}
		}
		double[] y = (double[]) objects[1];
		svm_parameter parameter = new svm_parameter();
		parameter.C = 0.5;
		parameter.eps = 0.01;
		parameter.kernel_type = svm_parameter.LINEAR;
		svm_problem problem = new svm_problem();
		problem.x = x1;
		problem.y = y;
		problem.l = x.length;
		svm_model model = libsvm.svm.svm_train(problem, parameter);
		int correct = 0;
		for (int i = 0; i < x1.length; i++) {
			int py = (int) libsvm.svm.svm_predict(model, x1[i]);
			if (py == y[i])
				correct++;
		}
		System.out.println("测试准确率 : " + (correct / (double) x1.length));
	}
	
	public static void main(String[] args) throws IOException {
		SVMExample svmExample = new SVMExample();
		svmExample.classify();
		svmExample.libsvmClassify();
	}
}
