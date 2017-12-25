package word2vec4j;

public class Constants {
	//训练文件
	public static final String TRAIN_FILE = "";
	//模型文件
	public static final String MODEL_FILE = "";

	public static final int EXP_TABLE_SIZE = 1000;
	
	public static final int MAX_EXP = 6;
	
	//一个词的最大长度限制
	public static final int MAX_STRING = 100;
	
	public static final int VOCAB_HASH_SIZE = 30000000;
	
	public static final int LAYER_1_SIZE = 100;
	
	//整形最大值，求随机值的使用使用
	public static final long RAND_MAX = 2147483647;
	
	//huffman编码最大长度
	public static final int MAX_CODE_LENGTH = 40;
	
	//模型训练的线程数
	public static final int NUM_THREADS = 10;
	
	//模型训练时的选词窗口大小
	public static final int WINDOW = 5;
	
	//梯度下降学习率
	public static final float STARTING_ALPHA = (float) 0.025;
	
	//断句标点符号，用来讲一句话再进行断句切分
	public static final String SEN_DELIMETOR = "[,，?？。！!]";
	
	//使用负例，0表示不使用负例
	public static final int NEGATIVE = 10;
	
	//是否使用Bag of Words模型
	public static final boolean CBOW = false;
	
	//是否使用hierachy softmax
	public static final boolean HS = true;
	
	//negative sampling使用的词概率分布数组
	public static final int tableSize = 100000000;
	
	/**
	 * Paragraoh2vec使用
	 */
	//待训练的句子，已完成分词
	//public static final String PARA_TRAIN_FILE = "D:\\data\\word2vec\\test\\verify_cut.dat";
	public static final String PARA_TRAIN_FILE = "D:\\data\\cluster\\phone\\raw\\raw.data.cut";
	
	//句子向量模型输出文件
	//public static final String PARA_MODEL_FILE = "D:\\data\\word2vec\\para_vectors.bin";
	public static final String PARA_MODEL_FILE = "D:\\data\\cluster\\phone\\para_vectors.bin";
	
	//句子向量训练时的学习率
	public static final float PARA_STARTING_ALPHA = (float) 0.05;
	
	//句子向量训练的迭代次数
	public static final int PARA_ITERATOR = 100;
	
	//句子向量的窗口大小
	public static final int PARA_WINDOW = 3;
	
	/**
	 * Distance.java使用
	 */
	public static final int topN = 40;
	
}
