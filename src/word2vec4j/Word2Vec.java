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
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.PriorityQueue;
import java.util.Queue;
import java.util.Random;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * 词向量训练(跟c版本的是一样的)
 * @author jiangwen
 *
 */
public class Word2Vec {
	//预先计算好的指数运算结果，只是为了提升后续的计算效率
	private float[] expTable;
	//词典
	private Map<String, VocabWord> vocab;
	//词典大小
	private int vocabSize = 0;
	//数组形式的词典,在负例采样的时候需要随机选一个词,只有词频计数和词id是有效的,其他变量是无效的
	private ArrayList<VocabWord> vocabSorted;
	//训练的词语计数
	private int trainWords;
	//训练语料输入文件
	private String trainFile = Constants.TRAIN_FILE;
	//输出模型文件
	private String modelFile = Constants.MODEL_FILE;
	//隐藏层
	private float[] syn1;
	//输入层，也就是词向量
	private float[] syn0;
	//负例采样的神经网络数据
	private float[] syn1neg;
	//huffman编码字符串，用于在huffman树深度遍历的时候记录当前编码串
	private char[] huffmanCoding;
	//huffman树节点编码串，用户在构建huffmanCoding的时候同时记录treeNode节点编号
	private int[] pointCoding;
	//negative sampling使用的采样数组
	private int[] table;
	
	//全局。词id计数
	private int wordId = 0;
	//全局。huffman树的所有节点编号
	private int treeNodeId = 0;
	//全局。线程共享，需要写的数据
	private AtomicInteger wordCountActual = new AtomicInteger();
	
	public Word2Vec() {
	}
	
	public Word2Vec(String trainFile, String modelFile) {
		this.trainFile = trainFile;
		this.modelFile = modelFile;
	}

	/**
	 * 初始化一些类变量
	 */
	public void init() {
		//这里是将这个e^x的连续值切成EXP_TABLE_SIZE个区间，每个区间有一个固定的e^x值可理解为将连续的函数值转换成离散数据,
		//对精度有一定损失，这样做纯粹是为了提高运行效率预先计算好e^x的值，x>MAX_EXP和x<-MAX_EXP的不做处理，这个范围内
		//sigmoid趋近于1或0
		expTable = new float[Constants.EXP_TABLE_SIZE];
		for (int i = 0; i < Constants.EXP_TABLE_SIZE; i++) {
			expTable[i] = (float) Math.exp((i / (float)Constants.EXP_TABLE_SIZE * 2 - 1) *
					(float)Constants.MAX_EXP);
			expTable[i] = expTable[i] / (expTable[i] + 1);
		}
		
		//初始化
		vocab = new HashMap<String, VocabWord>();
	}
	
	/**
	 * 初始化神经网络
	 */
	private void initNet() {
		long a, b;
		syn0 = new float[vocabSize * Constants.LAYER_1_SIZE];
		syn1 = new float[vocabSize * Constants.LAYER_1_SIZE];
		for (b = 0; b < Constants.LAYER_1_SIZE; b++) {
			for (a = 0; a < vocabSize; a++) {
				syn1[(int) (a * Constants.LAYER_1_SIZE + b)] = 0;
			}
		}
		
		if (Constants.NEGATIVE > 0) {
			syn1neg = new float[vocabSize * Constants.LAYER_1_SIZE];
			for (b = 0; b < Constants.LAYER_1_SIZE; b++) {
				for (a = 0; a < vocabSize; a++) {
					syn1neg[(int) (a * Constants.LAYER_1_SIZE + b)] = (float) 0.0;
				}
			}
		}
		
		Random random = new Random();
		for (b = 0; b < Constants.LAYER_1_SIZE; b++) {
			for (a = 0; a < vocabSize; a++) {
				int key = (int) (a * Constants.LAYER_1_SIZE + b);
				float value = (float) (random.nextInt() / ((float)Constants.RAND_MAX - 0.5) /
						(float) Constants.LAYER_1_SIZE);
				syn0[key] = value;
			}
		}
		
		createBinaryTree();
	}
	
	/**
	 * 构建huffman编码，思路如下：
	 * 1. 根据huffman编码规则，自底向上构建huffman树
	 * 2. 从根节点采用深度遍历，得到所有叶子节点的编码(所有叶子节点就是待编码的词)
	 */
	private void createBinaryTree() {
		Comparator<HuffmanTreeNode> wordComparator = new Comparator<HuffmanTreeNode>() {
			public int compare(HuffmanTreeNode o1, HuffmanTreeNode o2) {
				//warning, 这里强制转int，如果词频足够大，或出现精度损失
				return (int) (o1.getVocabWord().getCn() - o2.getVocabWord().getCn());
			}
		};
		
		Queue<HuffmanTreeNode> pq = new PriorityQueue<HuffmanTreeNode>(1, wordComparator);
		//叶子节点的编号范围是[0,vocabSize]
		for (Entry<String, VocabWord> entry : vocab.entrySet()) {
			VocabWord vocabWord = entry.getValue();
			pq.add(new HuffmanTreeNode(vocabWord, null, null, null, treeNodeId++));
		}
		/*
		System.out.println("pq size=" + pq.size());
		System.out.println(pq.peek().getVocabWord());
		System.exit(1);
		*/
		
		//取出两个频率最小的点，创建一个它们的父节点，词频为它们的和
		//如果优先级队列里只有一个节点，则它是root节点，已完成huffman树构建
		while (pq.size() > 1) {
			HuffmanTreeNode node1 = pq.poll();
			HuffmanTreeNode node2 = pq.poll();
			VocabWord vocabWord = new VocabWord(null, 0, 0, -1);
			//频次较大的节点放到右孩子节点，这样做有效果吗?
			HuffmanTreeNode left = node1, right = node2;
			if (node1.getVocabWord().getCn() > node2.getVocabWord().getCn()) {
				left = node2;
				right = node1;
			}
			vocabWord.setCn(node1.getVocabWord().getCn() + node2.getVocabWord().getCn());
			//由于没有使用叶子节点，所以可以将非叶子节点的编号调整到[0,vocabSize]区间内，原来[0, vocabSize]区间
			//存储的是叶子节点的编号
			HuffmanTreeNode newNode = new HuffmanTreeNode(vocabWord, left, right, null,
					treeNodeId - vocabSize);
			treeNodeId++;
			pq.add(newNode);
		}

		huffmanCoding = new char[Constants.MAX_CODE_LENGTH];
		pointCoding = new int[Constants.MAX_CODE_LENGTH];
		//最后仅会剩一个root节点在优先级队列中
		HuffmanTreeNode root = pq.poll();
		//深度遍历时记录当前节点的深度
		int depth = 0;
		//深度遍历huffman树，得到编码
		depthTrieve(root, depth);
	}
	
	//这里的编码方式，是按边来编码的，也就是编码长度=树的深度-1
	private void depthTrieve(HuffmanTreeNode node, int depth) {
		int id = node.getId();
		pointCoding[depth] = id; 
		//达到子节点，不再递归
		if (null == node.getLeft() && null == node.getRight()) {
			VocabWord vocabWord = node.getVocabWord();
			//必须是指针，才能更新到vocab中的vocabWord
			vocabWord.setCode(huffmanCoding.clone());
			vocabWord.setPoint(pointCoding.clone());
			//codeLen设置为实际huffman编码长度-1，没有使用叶子节点的编码，这个是按照c版代码逻辑来写的，暂时还不知道原因
			vocabWord.setCodeLen(depth);
			return ;
		} 
		
		//左递归,左边标0，右边标1。对应到hierachy softmax，左边为正1,右边为-1
		if (null != node.getLeft()) {
			huffmanCoding[depth] = '0';
			depthTrieve(node.getLeft(), depth + 1);
		}
		
		//右递归
		if (null != node.getRight()) {
			huffmanCoding[depth] = '1';
			depthTrieve(node.getRight(), depth + 1);
		}
	}
	
	/**
	 * 对过长的词，只截取前MAX_STRING长度的部分。只是为了避免词过长，对算法没有什么影响
	 * @param word
	 * @return
	 */
	private String getWordPrefix(String word) {
		if (word.length() > Constants.MAX_STRING) {
			System.out.println("addWordToVocab word is too long, word=" + word);
			word = word.substring(0, word.length());
		}
		
		return word;
	}
	
	/**
	 * 将新词添加到hash词典中
	 * @param word
	 */
	private void addWordToVocab(String word) {
		//新增新词时，词频计数为1，词id依次递增
		VocabWord vocabWord = new VocabWord(word, 1, 0, wordId++);
		vocab.put(word, vocabWord);
	}
	
	/**
	 * 读取词典数据，目前支持从训练文件中读取词典
	 * @throws IOException 
	 */
	private void learnVocabFromTrainFile() throws IOException {
		trainWords = 0;
		BufferedReader br = new BufferedReader(new FileReader(trainFile));
		
		addWordToVocab("</s>");
		String line = null;
		while (null != (line = br.readLine())) {
			//把断句的标点符号去掉
			String[] words = line.split("##")[0].split("\\s+");
			for (String word : words) {
				word = getWordPrefix(word).trim();
				trainWords++;
				if (0 == trainWords % 100000) {
					System.out.println(trainWords);
				}
				
				if (word.length() <= 0 )
					continue;
				
				VocabWord vocabWord = vocab.get(word);
				if (null == vocabWord) {
					addWordToVocab(word);
				} else {
					vocabWord.setCn(vocabWord.getCn() + 1);
				}
			}
		}
		
		vocabSize = vocab.size();
		
		System.out.println("Vocab Size:" + vocabSize);
		System.out.println("Words in train file:" + trainWords);
		
		br.close();

		//初始化数组形式的词典
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
	
	/**
	 * 获取训练文件的行数
	 * @return
	 * @throws IOException 
	 */
	private int getTrainFileLines() throws IOException {
		BufferedReader br = new BufferedReader(new FileReader(trainFile));
		int count = 0;
		while (null != (br.readLine())) {
			count++;
		}
		br.close();
		return count;
	}
	
	public void trainModel() throws IOException, InterruptedException {
		learnVocabFromTrainFile();
		//saveVocab();
		initNet();
		//如果需要使用负例采样方法，则预先计算好采样分布table
		if (Constants.NEGATIVE > 0)
			initUnigramTable();
		int lines = getTrainFileLines();
		List<Thread> threadList = new ArrayList<Thread>();
		for (int a = 0; a < Constants.NUM_THREADS; a++) {
			threadList.add(new TrainModelThread(a, lines, trainFile, vocab, expTable,
					trainWords, wordCountActual, vocabSorted, syn0, syn1, syn1neg, table));
		}
		for (Thread thread : threadList) {
			thread.start();
		}
		for (Thread thread : threadList) {
			thread.join();
		}
		
		saveModel();
	}
	
	//初始化词的概率分布数组(按词频计算的概率分布)
	private void initUnigramTable() {
		long trainWordsPow = (long) 0;
		float power = (float) 0.75;
		table = new int[Constants.tableSize];
		//在这里，vocab在之前必须按词频从大到小排序
		for (int a = 0; a < vocabSize; a++)
			trainWordsPow += Math.pow(vocabSorted.get(a).getCn(), power);
		int i = 0;
		float d1 = (float) (Math.pow(vocabSorted.get(i).getCn(), power) / (float) trainWordsPow);
		for (int a = 0; a < Constants.tableSize; a++) {
			table[a] = i;
			//vocabSorted[i]这个词的概率分布已填充完成，轮到下一个词
			if (a / (float) Constants.tableSize > d1) {
				i++;
				d1 += Math.pow(vocabSorted.get(i).getCn(), power) / (float) trainWordsPow;
			}
			if (i >= vocabSize)
				i = vocabSize - 1;
		}
	}
	
	private void saveModel() throws IOException {
		BufferedWriter bw = new BufferedWriter(new FileWriter(modelFile));
		bw.write(vocabSize + " " + Constants.LAYER_1_SIZE + "\n");

		//存储词典信息。以一个说明文分隔，便于人去分析这个模型数据
		bw.write("vocabulary\n");
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

		//存储训练出的词向量模型,syn0
		if (vocabSize != vocab.size()) {
			System.out.println("Bad Vocab, vocabSize!=vocab.size()");
			System.exit(1);
		}
		bw.write("word vector\n");
		for (Entry<String, VocabWord> entry : vocab.entrySet()) {
			VocabWord vocabWord = entry.getValue();
			bw.write(vocabWord.getWord() + " ");
			int id = vocabWord.getId();
			for (int b = 0; b < Constants.LAYER_1_SIZE; b++)
				bw.write(syn0[id * Constants.LAYER_1_SIZE + b] + " ");
			bw.write("\n");
		}
		
		//存储神经网络的网络结构（参数,syn1）,syn0是词向量，已经在第一步存储完成
		bw.write("hidden layer\n");
		for (int a = 0; a < vocabSize; a++) {
			for (int b = 0; b < Constants.LAYER_1_SIZE; b++) {
				bw.write(syn1[a * Constants.LAYER_1_SIZE + b] + " ");
			}
			bw.write("\n");
		}
		
		//存储negative sampling网络参数
		bw.write("negative sampling layer\n");
		for (int a = 0; a < vocabSize; a++) {
			for (int b = 0; b < Constants.LAYER_1_SIZE; b++) {
				bw.write(syn1neg[a * Constants.LAYER_1_SIZE + b] + " ");
			}
			bw.write("\n");
		}
		
		bw.close();
	}
	
	private void trainParagraphModel() {
		Paragraph2Vec paragraph2vec = new Paragraph2Vec();
		paragraph2vec.setExpTable(expTable);
		paragraph2vec.setSyn0(syn0);
		paragraph2vec.setSyn1(syn1);
		paragraph2vec.setSyn1neg(syn1neg);
		paragraph2vec.setVocab(vocab);
		paragraph2vec.setVocabSorted(vocabSorted);
		paragraph2vec.setTable(table);
		paragraph2vec.setVocabSize(vocabSize);
		paragraph2vec.setLayerSize(Constants.LAYER_1_SIZE);
		try {
			paragraph2vec.trainModelFromFile();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}


	public Map<String, VocabWord> getVocab() {
		return vocab;
	}
	public float[] getSyn0() {
		return syn0;
	}
	public int getLayerSize() {
		return Constants.LAYER_1_SIZE;
	}

	public static void main(String[] args) throws IOException, InterruptedException {
		Word2Vec word2vec = new Word2Vec();
		word2vec.init();
		word2vec.trainModel();
		System.out.println("word2vec train done ................................");
		word2vec.trainParagraphModel();
		System.out.println("para2vec train done ................................");
	}
}

