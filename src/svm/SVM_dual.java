package svm;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

import optimizer.smo.SMO;

/**
 * SVM对偶问题,使用SMO算法求解,没有使用QP方法,因为对内存和时间要求较高
 * 采用one-vs-one的vote方法进行多个二分类模型整合
 * @author J.
 *
 */
public class SVM_dual {
	
	public SVMModel train(double[][] x, int[] y, SVMParameter parameter) {
		Set<Integer> yset = new HashSet<Integer>();
		for (int i = 0; i < y.length; i++)
			yset.add((int) y[i]);
		List<Integer> yList = new ArrayList<Integer>();
		for (Integer value : yset)
			yList.add(value);
		SVMModel svmModel = new SVMModel(yset.size() * (yset.size() - 1) / 2);
		int index = 0;
		//构建N*(N-1)/2个vote子模型
		for (int i = 0; i < yList.size() - 1; i++) {
			for (int j = i + 1; j < yList.size(); j++) {
				int y1 = yList.get(i), y2 = yList.get(j);
				List<double[]> subNodeList = new ArrayList<double[]>();
				List<Integer> subYList = new ArrayList<Integer>();
				for (int k = 0; k < y.length; k++) {
					if (y[k] == y1) {
						subNodeList.add(x[k]);
						subYList.add(1);
					}
					if (y[k] == y2) {
						subNodeList.add(x[k]);
						subYList.add(-1);
					}
				}
				double[][] subx = new double[subNodeList.size()][];
				for (int m = 0; m < subNodeList.size(); m++)
					subx[m] = subNodeList.get(m);
				int[] suby = new int[subYList.size()];
				for (int m = 0; m < subYList.size(); m++)
					suby[m] = subYList.get(m);
				System.out.println("train binary svm, number is : " + index);
				SVMBinaryModel subModel = train_one(subx, suby, parameter);
				subModel.setpLabel(y1);
				subModel.setnLabel(y2);
				svmModel.setModel(index++, subModel);
			}
		}
		return svmModel;
	}
	
	public int predict(SVMModel model, double[] x) {
		Map<Integer, Integer> voteMap = new HashMap<Integer, Integer>();
		for (SVMBinaryModel subModel : model.modelList) {
			double p = SMO.predictOneStatic(subModel, x);
			int label = subModel.getnLabel();
			if (p > 0)
				label = subModel.getpLabel();
			if (null == voteMap.get(label))
				voteMap.put(label, 0);
			voteMap.put(label, voteMap.get(label) + 1);
		}
		int maxLable = -1, maxVote = -1;
		for (Entry<Integer, Integer> entry : voteMap.entrySet()) {
			if (entry.getValue() > maxVote) {
				maxVote = entry.getValue();
				maxLable = entry.getKey();
			}
		}
		return maxLable;
	}
	
//	public SVMModel train(SvmNode[][] nodes, int[] y, SVMParameter parameter) {
//		SVMModel svmModel = null;
//		Set<Integer> yset = new HashSet<Integer>();
//		for (int i = 0; i < y.length; i++)
//			yset.add((int) y[i]);
//		int[] labels = new int[yset.size()]; 
//		//对于二分类问题,可以训练两个重复的模型,预测的时候只使用第一个模型即可,方便写代码
//		svmModel = new SVMModel(yset.size());
//		int row = 0;
//		for (Integer key : yset) {
//			labels[row] = key;
//			int[] suby = new int[y.length]; 
//			for (int i = 0; i < nodes.length; i++) {
//				if (y[i] == key)
//					suby[i] = 1;
//				else
//					suby[i] = -1;
//			}
//
//			SVMBinaryModel subModel = train_one(nodes, suby, parameter);
//			svmModel.setModel(row, subModel, key);
//			row++;
//		}
//		return svmModel;
//	}
	
	private SVMBinaryModel train_one(double[][] x, int[] y, SVMParameter parameter) {
		SMO svm = new SMO(x, y, parameter);
		svm.train();
		SVMBinaryModel subModel = new SVMBinaryModel(svm.getB(), svm.getAlph(),
				svm.getTrainSamples(), svm.getTrainLabels()); 
		return subModel;
	}
	
//	public int predict(SVMModel model, SvmNode[] x) {
//		int maxLabel = 0;
//		double max = -99999999;
//		//分类数为2的时候单独处理
//		if (model.N == 2) {
//			SVMBinaryModel subModel = model.modelList[0];
//			int label = model.labels[0];
//			double y = SMO.predict(subModel, x);
//			if (y > 0)
//				maxLabel = label;
//			else
//				maxLabel = model.labels[1];
//			return maxLabel;
//		}
//
//		for (int i = 0; i < model.N; i++) {
//			SVMBinaryModel subModel = model.modelList[i];
//			int label = model.labels[i];
//			double y = SMO.predict(subModel, x);
//			if (y > max) {
//				max = y;
//				maxLabel = label;
//			}
//		}
//		return maxLabel;
//	}
}
