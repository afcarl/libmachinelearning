package regression;

import java.io.IOException;

import common.FeatureNode;
import common.Parameter;

public interface Regressioner {
	public Model train(FeatureNode[][] nodes, double[] y, Parameter param) throws Exception;
	
	public void dump(Model model, String filePath) throws IOException;
	
	public double predict(Model model, FeatureNode[] x);
}
