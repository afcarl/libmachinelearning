package optimizer.smo;

import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.Set;

import svm.SVMBinaryModel;
import svm.SVMParameter;

public class SMO {
	private double[][] trainSamples;
	private int[] trainLabels;
	private double[] w;
	private double b;
	private double[] alph;
	private double C;
	private Map<Integer, Double> errorMap;
	
	public SMO(double[][] trainSamples, int[] trainLabels, SVMParameter parameter) {
		this.setTrainSamples(trainSamples);
		this.setTrainLabels(trainLabels);
		this.C = parameter.C;
		
		b = 0;
		int attSize = this.getTrainSamples()[0].length;
		int nSamples = this.getTrainSamples().length;
		w = new double[attSize];
		alph = new double[nSamples];
		
		errorMap = new HashMap<Integer, Double>();
	}
	
	public int takeStep(int i1, int i2) {
		if(i1 == i2) 
			return 0;
		double alph1 = this.alph[i1];
		double alph2 = this.alph[i2];
		
		int y1 = this.getTrainLabels()[i1];
		int y2 = this.getTrainLabels()[i2];
		
		double[] v1 = this.getTrainSamples()[i1];
		double[] v2 = this.getTrainSamples()[i2];
		
		double e1 = innerProduct(this.w, v1) - this.b - y1;
		double e2 = innerProduct(this.w, v2) - this.b - y2;
		int s = y1 * y2;
		
		double L; 				//下界
		double H;				//上界
		if(s < 0) {
			L = alph2 - alph1 > 0 ? alph2 - alph1 : 0;
			H = this.C > this.C + alph2 - alph1 ? this.C + alph2 - alph1 : this.C;
		} else {
			L = alph1 + alph2 - this.C > 0 ? alph1 + alph2 - this.C: 0;
			H = this.C > alph1+ alph2 ? alph1 + alph2 : this.C;
		}
		
		if(L == H) {
			return 0;
		}
		
		double k11 = innerProduct(v1, v1);
		double k12 = innerProduct(v1, v2);
		double k22 = innerProduct(v2, v2);
		double eta = 2 * k12 - k11 - k22;
		
		double a1;
		double a2;
		double eps = 0.001;
		if(eta < 0) {
			a2 = alph2 - ( (y2 * (e1 - e2) ) / eta);
			if(a2 < L) {
				a2 = L;
			} else if (a2 > H) {
				a2 = H;
			}
		} else {
			System.out.println("eta < 0");
			
			double r = -y1 * sumAY(i1, i2);
			double vv1 = innerProduct(v1, this.w) - y1 * alph1 * k11 - y2 * alph2 * k12;
			double vv2 = innerProduct(v2, this.w) - y1 * alph1 * k12 - y2 * alph2 * k22;
			
			double lobj = r - s * L + L - 0.5 * k11 * (r - s*L) * (r - s*L) - 0.5 * k22 * L * L - s*k12*(r - s*L)*L - y1*(r - s*L)*vv1 - y2*L*vv2;
			double hobj = r - s * H + H - 0.5 * k11 * (r - s*H) * (r - s*H) - 0.5 * k22 * H * H - s*k12*(r - s*H)*H - y1*(r - s*H)*vv1 - y2*H*vv2;
			
			if(lobj > hobj + eps) {
				a2 = L;
			} else if (lobj < hobj - eps) {
				a2 = H;
			} else {
				a2 = alph2;
			}
		}
		
		if(Math.abs(a2 - alph2) < eps * (a2 + alph2 + eps) ) {
			return 0;
		}
		
		a1 = alph1 + s * (alph2 - a2);
		
		
		double b1 = e1 + y1 * (a1 - alph1) * k11 + y2 * (a2 - alph2) * k12 + b;
		double b2 = e2 + y1 * (a1 - alph1) * k12 + y2 * (a2 - alph2) * k22 + b;
		
		double bOld = this.b;				//未更新之前的b;			
		this.b = (b1 + b2) / 2;
		
		
		this.alph[i1] = a1;
		this.alph[i2] = a2;
		
		if(a1 < this.C && a1 > 0) {							//非边界点
			errorMap.put(i1, 0.0);
		}
		
		if(a2 < this.C && a2 > 0) {
			errorMap.put(i2, 0.0);
		}
		
		Set<Integer> alphs = errorMap.keySet();
		Iterator<Integer> it = alphs.iterator();
		while(it.hasNext()) {
			int a = it.next();
			if(a == i1 || a == i2) {
				continue;
			}
			double[] vk = getTrainSamples()[a];
			double errorOld = errorMap.get(a);
			double errorNew = errorOld + y1 * (a1 - alph1) * innerProduct(v1, vk) + y2 * (a2 - alph2) * innerProduct(v2, vk) + bOld - this.b;
			
			errorMap.put(a, errorNew);
		}	//更新error 缓存
		
		
		double theta1 = y1 * (a1 - alph1);					//线性核可以这样做
		double theta2 = y2 * (a2 - alph2);
		double[] v11 = product(theta1, v1);
		double[] v22 = product(theta2, v2);
		double[] delta = add(v11, v22);
		this.w = add(this.w, delta);
		
//		System.out.println("i1 = " + i1 + ", i2 = " + i2);
		return 1;
	}
	
	//求和alph i * y i
	public double sumAY(int i1, int i2) {
		double result = 0;
		for(int i = 0; i < this.alph.length; i++) {
			if(i != i1 && i != i2) {
				result += this.alph[i] * this.getTrainLabels()[i];
			}
		}
		return result;
	}
	
	//向量数乘
	public double[] product(double a, double[] b) {
		double[] result = new double[b.length];
		for(int i = 0; i < b.length; i++) {
			result[i] = a * b[i];
		}
		return result;
	}
	
	//数组相加
	public double[] add(double[] a, double[] b) {
		if(a.length != b.length) {
			return null;
		}
		double[] result = new double[a.length];
		for(int i = 0; i < a.length; i++) {
			result[i] = a[i] + b[i];
		}
		return result;
	}
	
	//两向量内积
	public double innerProduct(double[] a, double[] b) {
		if(a.length != b.length) {
			return 0;
		}
		int length = a.length;
		double result = 0;
		for(int i = 0; i < length; i++) {
			result += a[i] * b[i];
		}
		return result;
	}

	
	// 计算输入向量输出
	public double calculateOutput(double[] input) {
		double result = innerProduct(this.w, input) - this.b;
		return result;
	}
	
	public int predictOne(double[] input) {
		double result = calculateOutput(input);
		if(result >= 0) {
			return 1;
		} else {
			return -1;
		}
	}
	
	public static double predictOneStatic(SVMBinaryModel model, double[] x) {
		//double p = innerProductStatic(model., x) - model.getB(); 
		double[][] tx = model.x;
		double sum = 0;
		for (int i = 0; i < tx.length; i++) {
			sum += innerProductStatic(tx[i], x) * model.y[i] * model.alpha[i];
		}
		sum -= model.b;
		return sum;
	}
	
	//两向量内积
	public static double innerProductStatic(double[] a, double[] b) {
		if(a.length != b.length) {
			return 0;
		}
		int length = a.length;
		double result = 0;
		for(int i = 0; i < length; i++) {
			result += a[i] * b[i];
		}
		return result;
	}
	
	public int[] predict(double[][] testData) {
		double[] input;
		int num = testData.length;
		int[] predictedLabel = new int[num];
		for(int i = 0; i < num; i++) {
			input = testData[i];
			predictedLabel[i] = predictOne(input);
		}
		return predictedLabel;
	}
	
	public double accuracy(int[] labels, int[] predictedLabels) {
		int labelsLength = labels.length;
		int predictedLabelsLength = predictedLabels.length;
		if(labelsLength != predictedLabelsLength) {
			System.out.println("different length.");
			return 0;
		}
		double totleCorrect = 0;
		for(int i = 0; i < labelsLength; i++) {
			if(labels[i] == predictedLabels[i]) {
				totleCorrect++;
			}
		}
		return totleCorrect / labelsLength;
	}
	
	public double[] getAlph() {
		return this.alph;
	}
	
	public double[] getW() {
		return this.w;
	}
	
	public double getB() {
		return this.b;
	}
	
	public int  examineExample(int i2) {
		double tol = 0.001;
		int y2 = this.getTrainLabels()[i2];
		double alph2 = this.alph[i2];
		double[] v2 = this.getTrainSamples()[i2];
		double output = calculateOutput(v2);
		double r2 = y2 * (output - y2);
		if((r2 < -tol && alph2 < this.C) || (r2 > tol && alph2 > 0)) {
			if(this.errorMap.size() > 1) {
				double min = Double.MAX_VALUE;
				double max = Double.MIN_VALUE;
				int minIndex = 0;
				int maxIndex = 0;
				
				Set<Integer> indexs = this.errorMap.keySet();
				Iterator<Integer> it = indexs.iterator();
				int i1 = 0;
				while(it.hasNext()) {
					int currentAlph = it.next();
					double value = errorMap.get(currentAlph);
					if(value > max) {
						max = value;
						maxIndex = currentAlph;
					}
					
					if(value < min) {
						min = value;
						minIndex = currentAlph;
					}					
				}
				if(output - y2 < 0) {
					i1 = maxIndex;
				} else {
					i1 = minIndex;
				}
				
				if(takeStep(i1, i2) == 1) {
					return 1;
				}	
			}
			
			
			//loop over all non-zero and non-C alpha, starting at random point
			Set<Integer> nonZeroC = this.errorMap.keySet();
			RangeRandom rr = new RangeRandom();
			int[] als = rr.rSequence(nonZeroC);
			for(int i = 0; i < als.length; i++) {
				int i1 = als[i];
				if(takeStep(i1, i2) == 1) {
					return 1;
				}
			}
			
			//loop over all possible i1, starting at a random point
			int[] allAls = rr.rangeRandom(this.getTrainSamples().length);
			for(int i = 0; i < allAls.length; i++) {
				int i1 = i;
				if(takeStep(i1, i2) == 1) {
					return 1;
				}
			}
			
		}
		return 0;
	}

	public void train() {
		int numChanged = 0;
		boolean examineAll = true;
		while(numChanged > 0 || examineAll) {
			numChanged = 0;
			if(examineAll) {
				this.errorMap.clear();
				for(int i = 0; i < this.getTrainSamples().length; i++) {
					numChanged += examineExample(i);
				}
			} else {			
				Set<Integer> indexOfAlph = this.errorMap.keySet();
				RangeRandom rr = new RangeRandom();
				int[] alps = rr.rSequence(indexOfAlph);
				for(int i = 0; i < alps.length; i++) {
					numChanged += examineExample(alps[i]);
				}
			}
			
			if(examineAll) {
				examineAll = false;
			} else if (numChanged == 0) {
				examineAll = true;
			}
		}
		System.out.println("train one svm");
	}

	public double[][] getTrainSamples() {
		return trainSamples;
	}

	public void setTrainSamples(double[][] trainSamples) {
		this.trainSamples = trainSamples;
	}

	public int[] getTrainLabels() {
		return trainLabels;
	}

	public void setTrainLabels(int[] trainLabels) {
		this.trainLabels = trainLabels;
	}
}
