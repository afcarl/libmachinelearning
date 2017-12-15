package word2vec4j;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import java.util.Map.Entry;

/**
 * 句子向量训练部分
 * @author jiangwen
 *
 */
public class Paragraph2Vec {
	//学习率
	private float alpha = Constants.PARA_STARTING_ALPHA;
	//词典
	private Map<String, VocabWord> vocab;
	//数组形式的词典
	private ArrayList<VocabWord> vocabSorted;
	//词频概率分布数组
	private int[] table;
	//指数运算结果
	private float[] expTable;
	//句子向量
	private float[] syn0new;
	private float[] syn1new;
	private float[] neu1e;
	//word2vec训练出来的词向量
	private float[] syn0;
	private float[] syn1;
	private float[] syn1neg;
	//使用cbow还是skip-gram模型
	private boolean cbow = Constants.CBOW;
	//是否使用hierachy softmax
	private boolean hs = Constants.HS;
	//随机数
	private long nextRandom = 1;
	private Random random = new Random();
	
	//词典大小（去重后）
	private int vocabSize;
	//向量维数
	private int layerSize;
	
	//模型文件
	private String modelFile = Constants.MODEL_FILE;
	
	public Paragraph2Vec() {
	}
	
	public Paragraph2Vec(String modelFile) {
		this.modelFile = modelFile;
	}
	
	//初始化和神经网络关系不大的数据
	private void initOther() {
		expTable = new float[Constants.EXP_TABLE_SIZE];
		for (int i = 0; i < Constants.EXP_TABLE_SIZE; i++) {
			expTable[i] = (float) Math.exp((i / (float)Constants.EXP_TABLE_SIZE * 2 - 1) *
					(float)Constants.MAX_EXP);
			expTable[i] = expTable[i] / (expTable[i] + 1);
		}
		
		//按词频排序的词表
		vocabSorted = new ArrayList<VocabWord>();
		int i = 0;
		for (Entry<String, VocabWord> entry : vocab.entrySet()) {
			if (i > vocabSize) {
				System.out.println("vocabSize is not right!");
				break;
			}
			vocabSorted.add(entry.getValue());
		}
		
		//从大到小
		Comparator<VocabWord> comparator = new Comparator<VocabWord>() {
			public int compare(VocabWord o1, VocabWord o2) {
				if (o1.getCn() == o2.getCn())
					return 0;
				else if (o1.getCn() > o2.getCn())
					return -1;
				else
					return 1;
			}
		};
		
		Collections.sort(vocabSorted, comparator);
	}
	
	public void initNet() throws IOException {

		//从模型文件中把整个词向量的神经网络读入内存
		BufferedReader br = new BufferedReader(new FileReader(modelFile));
		String line = br.readLine();
		if (null == line || 2 != line.split("\\s+").length) {
			System.out.println("Bad word vector model!");
			br.close();
			return ;
		}
		
		String[] items = line.split("\\s+");
		vocabSize = Integer.valueOf(items[0]);
		layerSize = Integer.valueOf(items[1]);
		
		//跳过说明文
		while (null != (line = br.readLine())) {
			line = line.trim();
			if (0 == line.compareTo("vocabulary"))
				break;
		}
		//读入词典
		System.out.println("reading " + line + " ......");
		vocab = new HashMap<String, VocabWord>();
		for (int i = 0; i < vocabSize; i++) {
			items = br.readLine().trim().split("\\s+");
			String word = items[0];
			int id = Integer.valueOf(items[1]);
			int cn = Integer.valueOf(items[2]);
			int codeLen = Integer.valueOf(items[3]);
			char[] code = new char[Constants.MAX_CODE_LENGTH];
			int[] point = new int[Constants.MAX_CODE_LENGTH];
			for (int j = 0; j < codeLen; j++)
				code[j] = items[j + 4].charAt(0);
			for (int j = 0; j < codeLen; j++)
				point[j] = Integer.valueOf(items[j + 4 + codeLen]);
			vocab.put(word, new VocabWord(word, cn, codeLen, id, code, point));
		}
		
		//读入神经网络向量(syn0)
		//跳过section开头的说明文
		while (null != (line = br.readLine())) {
			line = line.trim();
			if (0 == line.compareTo("word vector"))
				break;
		}
		System.out.println("reading " + line + " ......");
		syn0 = new float[vocabSize * layerSize];
		for (int i = 0; i < vocabSize; i++) {
			items = br.readLine().trim().split("\\s+");
			VocabWord vocabWord = vocab.get(items[0]);
			int id = vocabWord.getId();
			for (int j = 0; j < layerSize; j++)
				syn0[id * layerSize + j] = Float.valueOf(items[j + 1]);
		}
		
		//读入神经网络隐藏层参数(syn1)
		syn1 = new float[vocabSize * layerSize];
		while (null != (line = br.readLine())) {
			line = line.trim();
			if (0 == line.compareTo("hidden layer"))
				break;
		}
		System.out.println("reading " + line + " ......");
		for (int i = 0; i < vocabSize; i++) {
			items = br.readLine().trim().split("\\s+");
			for (int j = 0; j < layerSize; j++) {
				syn1[i * layerSize + j] = Float.valueOf(items[j]);
			}
		}
		
		//读入神经网络negative sampling网络参数(syn1neg)
		syn1neg = new float[vocabSize * layerSize];
		while (null != (line = br.readLine())) {
			line = line.trim();
			if (0 == line.compareTo("negative sampling layer"))
				break;
		}
		System.out.println("reading " + line + " ......");
		for (int i = 0; i < vocabSize; i++) {
			items = br.readLine().trim().split("\\s+");
			for (int j = 0; j < layerSize; j++) {
				syn1neg[i * layerSize + j] = Float.valueOf(items[j]);
			}
		}

		initOther();
		br.close();
	}
	
	public void trainModelFromFile() throws IOException {
		//初始化句子向量需要用到的参数
		syn0new = new float[layerSize];
		syn1new = new float[layerSize];
		neu1e = new float[layerSize];

		BufferedReader br = new BufferedReader(new FileReader(Constants.PARA_TRAIN_FILE));
		BufferedWriter bw = new BufferedWriter(new FileWriter(Constants.PARA_MODEL_FILE));
		int counter = 0;
		String line = null;
		while (null != (line = br.readLine())) {
			if (0 == counter++ % 1000)
				System.out.println("Progress:" + counter);
			String[] items = line.trim().split("##"); //以"##"为分隔符，后面部分是注释
			trainParagraphVector(items[0]);
			line = line.replace("##", "##seperator##");
			bw.write(line + " ##seperator## ");
			for (int i = 0; i < layerSize; i++)
				bw.write(syn0new[i] + " ");
			bw.write("\n");
		}
		
		br.close();
		bw.close();

		System.out.println("trainParagraphVector done, progress:" + counter);
	}
	
	
	public void trainParagraphVector(String line) {
		for (int b = 0; b < layerSize; b++)
			//RAND_MAX应该取java int的上限?
			syn0new[b] = (float) ((random.nextInt() / (float) Integer.MAX_VALUE - 0.5) / layerSize);
		for (int b = 0; b < layerSize; b++)
			syn1new[b] = (float) 0.0; 
		
		String[] words = line.split("\\s+");
		int senLen = words.length;
		//句子向量的训练，需要对一个句子进行反复迭代收敛
		for (int i = 0; i < Constants.PARA_ITERATOR; i++) {
			alpha = (1 - (float) i / (float) Constants.PARA_ITERATOR) * Constants.PARA_STARTING_ALPHA;
			if (alpha < Constants.PARA_STARTING_ALPHA * 0.0001)
				alpha = Constants.PARA_STARTING_ALPHA * (float) 0.0001;
			//alpha = Constants.PARA_STARTING_ALPHA;
			for (int senPosition = 0; senPosition < senLen; senPosition++) {
				if (cbow) {
				} else {
					skipGram(words, senPosition);
				}
			}
		}
	}
	
	private void skipGram(String[] words, int senPosition) {
		int senLen = words.length;
		VocabWord word = vocab.get(words[senPosition]);
		if (null == word)
			return ;
			
		for (int c = 0; c < layerSize; c++)
			neu1e[c] = (float) 0.0;
		//这里选择固定窗口大小，而不是随机窗口大小
//		int b = Constants.PARA_WINDOW - 1;
		int b = 0;
		for (int a = b; a < Constants.PARA_WINDOW * 2 + 1 - b; a++) {
			if (a == Constants.PARA_WINDOW)
				continue;
			int c = senPosition - Constants.PARA_WINDOW + a;
			if (c < 0 || c >= senLen)
				continue;
			VocabWord lastWord = vocab.get(words[c]);
			if (null == lastWord)
				continue;
			int l1 = lastWord.getId() * layerSize;
			for (c = 0; c < layerSize; c++)
				neu1e[c] = 0;
			if (hs) for (int d = 0; d < word.getCodeLen(); d++) {
				float f = (float) 0.0;
				int l2 = word.getPoint()[d] * layerSize;
				for (c = 0; c < layerSize; c++)
					f += syn0[c + l1] * syn1[c + l2];
				for (c = 0; c < layerSize; c++)
					f += syn0new[c] * syn1[c + l2];
				if (f <= -Constants.MAX_EXP)
					continue;
				else if (f >= Constants.MAX_EXP)
					continue;
				else
					f = expTable[(int) ((f + (float) Constants.MAX_EXP) * (Constants.EXP_TABLE_SIZE /
						Constants.MAX_EXP / 2))];
				//上面的计算是否要转成float？几个常量都是int，除法会损失精度
				//float g = (word.getCode()[d] - '0' - f) * f * (1 - f) * alpha;
				float g = (1 - (word.getCode()[d] - '0') - f) * (float) alpha;
				for (c = 0; c < layerSize; c++)
					neu1e[c] += g * syn1[c + l2];
			}
			
			if (Constants.NEGATIVE > 0) {
				for (int d = 0; d < Constants.NEGATIVE + 1; d++) {
					int targetWordId = 0;
					float label = (float) 0.0;
					if (0 == d) {
						targetWordId = word.getId();
						label = (float) 1.0;
					} else {
						nextRandom = Math.abs(nextRandom * (long) 25214903917l + 11);
//						nextRandom = Math.abs(random.nextInt());
						int index = (int) (Math.abs(nextRandom >> 16) % table.length);
						if (0 == vocabSorted.get(table[index]).getWord().compareTo("</s>"))
							continue;
						targetWordId = vocabSorted.get(table[index]).getId();
						if (targetWordId == word.getId())
							continue;
					}
					int l2 = targetWordId * layerSize;
					float f = (float) 0.0, g = (float) 0.0;
					for (c = 0; c < layerSize; c++)
						f += (syn0[c + l1] * syn1neg[c + l2]);
					for (c = 0; c < layerSize; c++)
						f += syn0new[c] * syn1neg[c + l2];
					if (f > Constants.MAX_EXP)
						g = (label - (float) 1) * alpha;
					else if (f < -Constants.MAX_EXP)
						g = (label - (float) 0) * alpha;
					else 
						g = (label - expTable[(int) ((f + (float) Constants.MAX_EXP) * (Constants.EXP_TABLE_SIZE /
								Constants.MAX_EXP / 2))]) * alpha;
					for (c = 0; c < Constants.LAYER_1_SIZE; c++)
						neu1e[c] += (g * syn1neg[c + l2]);
				}
			}
			
			for (c = 0; c < layerSize; c++)
				syn0new[c] += neu1e[c];
		}
	}
	
	public Map<String, float[]> getWordVectors() {
		Map<String, float[]> vectors = new HashMap<String, float[]>();
		for (Entry<String, VocabWord> entry : vocab.entrySet()) {
			float[] vec = getWordVec(entry.getKey());
			vectors.put(entry.getKey(), vec);
		}

		return vectors;
	}
	
	private float[] getWordVec(String word) {
		VocabWord vocabWord = vocab.get(word);
		if (null == vocabWord)
			return null;
		
		float[] vec = new float[layerSize];
		int wordId = vocabWord.getId();
		for (int j = 0; j < layerSize; j++)
			vec[j] = syn0[wordId * layerSize + j];

		return vec;
	}

	public float[] getSyn0new() {
		return syn0new;
	}
	public int getLayerSize() {
		return layerSize;
	}
	public Map<String, VocabWord> getVocab() {
		return vocab;
	}
	public ArrayList<VocabWord> getVocabSorted() {
		return vocabSorted;
	}
	public void setVocabSorted(ArrayList<VocabWord> vocabSorted) {
		this.vocabSorted = vocabSorted;
	}
	public float[] getExpTable() {
		return expTable;
	}
	public void setExpTable(float[] expTable) {
		this.expTable = expTable;
	}
	public float[] getSyn0() {
		return syn0;
	}
	public void setSyn0(float[] syn0) {
		this.syn0 = syn0;
	}
	public float[] getSyn1() {
		return syn1;
	}
	public void setSyn1(float[] syn1) {
		this.syn1 = syn1;
	}
	public float[] getSyn1neg() {
		return syn1neg;
	}
	public void setSyn1neg(float[] syn1neg) {
		this.syn1neg = syn1neg;
	}
	public void setVocab(Map<String, VocabWord> vocab) {
		this.vocab = vocab;
	}
	public int getVocabSize() {
		return vocabSize;
	}
	public void setVocabSize(int vocabSize) {
		this.vocabSize = vocabSize;
	}
	public void setLayerSize(int layerSize) {
		this.layerSize = layerSize;
	}
	public void setTable(int[] table) {
		this.table = table;
	}

	public void checkVocab() throws IOException {
		String filename = "E:\\data\\word2vec\\test\\vocab.dat";
		BufferedWriter bw = new BufferedWriter(new FileWriter(filename));
		int i = 0;
		for (Entry<String, VocabWord> entry : vocab.entrySet()) {
			if (i > vocabSize){
				System.out.println("Bad vocab!");
				return ;
			}

			VocabWord vocabWord = entry.getValue();
			bw.write(vocabWord.getWord() + " ");
			bw.write(vocabWord.getId() + " ");
			bw.write(vocabWord.getCn() + " ");
			bw.write(vocabWord.getCodeLen() + " ");
			int codeLen = vocabWord.getCodeLen();
			for (int j = 0; j < codeLen; j++)
				bw.write(vocabWord.getCode()[j] + " ");
			for (int j = 0; j < codeLen; j++)
				bw.write(vocabWord.getPoint()[j] + " ");
			bw.write("\n");

			i++;
		}
		
		bw.close();
	}

	public static void main(String[] args) throws IOException {
		Paragraph2Vec paragraph2vec = new Paragraph2Vec();
		paragraph2vec.initNet();
		paragraph2vec.trainModelFromFile();
	}
}
