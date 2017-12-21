package example;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import logistic.FeatureNode;

public class DataReader {

	public Object[] readIn(String file) throws IOException {
		BufferedReader br = new BufferedReader(new FileReader(file));
		List<Double> yList = new ArrayList<Double>();
		List<FeatureNode[]> nodeArrayList = new ArrayList<FeatureNode[]>();
		String line = null;
		int c = 0, maxIndex = 0;
		while (null != (line = br.readLine())) {
			String[] items = line.trim().split("\\s+");
			FeatureNode[] nodeArray = new FeatureNode[items.length - 1];
			for (int i = 1; i < items.length; i++) {
				String[] kv = items[i].split(":");
				nodeArray[i - 1] = new FeatureNode(Integer.valueOf(kv[0]), Double.valueOf(kv[1]));
				if (Integer.valueOf(kv[0]) > maxIndex)
					maxIndex = Integer.valueOf(kv[0]);
			}
			nodeArrayList.add(nodeArray);
			yList.add(Double.valueOf(items[0]));
			if (c++ > 2000)
				break;
		}
		br.close();
		
		FeatureNode[][] x = new FeatureNode[nodeArrayList.size()][];
		for (int i = 0; i < nodeArrayList.size(); i++)
			x[i] = nodeArrayList.get(i);
		double[] y = new double[yList.size()]; 
		for (int i = 0; i < yList.size(); i++)
			y[i] = yList.get(i);
		Object[] object = new Object[3];
		object[0] = x;
		object[1] = y;
		object[2] = maxIndex;
		return object;
	}
}
