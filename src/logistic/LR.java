package logistic;

import java.io.IOException;

public interface LR {

	public Model train(FeatureNode[][] nodes, double[] y, Parameter param) throws Exception;
	
	public void dump(Model model, String filePath) throws IOException;
	
	public Model load(String filePath);
	
	public double predict(Model model, FeatureNode[] x);
}
