package svm;

/**
 * 多分类模型,包括多个SVMBinaryModel
 * @author J.
 */
public class SVMModel {
	//modeList[i]和labels[i]对应, one vs all模型
	SVMBinaryModel[] modelList = null;
	int N = 0;
	
	public SVMModel(int size) {
		modelList = new SVMBinaryModel[size];
		N = size;
	}
	
	public void setModel(int index, SVMBinaryModel svmBinaryModel) {
		modelList[index] = svmBinaryModel;
	}
}
