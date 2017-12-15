package word2vec4j;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;

//TODO 负例采样中需要使用table的概率分布进行采样
public class TrainModelThread extends Thread {
	
	//线程号
	private int threadNo;
	//训练文件的行数,用于对每个线程分配处理的文件行范围
	private int lines;
	//训练文件
	private String trainFile;
	//词典数据
	private Map<String, VocabWord> vocab;
	//指数计算结果
	private float[] expTable;
	private float[] syn0;
	private float[] syn1;
	private float[] syn1neg;
	//整个训练文件的词数量
	private int trainWords;
	//全局。线程之间共享
	//实际处理的词计数，如果设置了根据词频过滤低频词，则低频词不做处理，不算在这个计数里
	private AtomicInteger wordCountActual;
	//数组形式的词典
	private ArrayList<VocabWord> vocabSorted;
	//词频概率分布数组
	private int[] table;

	//误差项
	private float[] neu1;
	private float[] neu1e;
	//模型类型,cbow或者skip-gram
	private boolean cbow = Constants.CBOW;
	//是否使用hierachy soft
	private boolean hs = Constants.HS;
	//随机数
	private long nextRandom = 0;
	//学习率
	private float alpha = Constants.STARTING_ALPHA;
	
	public TrainModelThread(int threadNo, int lines, String trainFile,
			Map<String, VocabWord> vocab, float[] expTable, int trainWords,
			AtomicInteger wordCountActual, ArrayList<VocabWord> vocabSorted,
			float[] syn0, float[] syn1, float[] syn1neg, int[] table) {
		this.threadNo = threadNo;
		this.lines = lines;
		this.trainFile = trainFile;
		this.vocab = vocab;
		this.expTable = expTable;
		this.trainWords = trainWords;
		this.wordCountActual = wordCountActual;
		this.vocabSorted = vocabSorted;
		this.syn0 = syn0;
		this.syn1 = syn1;
		this.syn1neg = syn1neg;
		this.table = table;

		neu1 = new float[Constants.LAYER_1_SIZE];
		neu1e = new float[Constants.LAYER_1_SIZE];
	}

	@Override
	public void run() {
		nextRandom = (long)threadNo;
		int pageSize = lines / Constants.NUM_THREADS + 1;
		int begin = pageSize * threadNo;
		try { 

		System.out.println("threadNo:" + threadNo + ", begin:" + begin);
		BufferedReader br = new BufferedReader(new FileReader(trainFile));
		int count = 0;
		while (true) {
			count++;
			if (count >= begin)
				break;
			br.readLine();
		}
		String line = null;
		for (int i = 0; i < pageSize; i++) {
			if (0 == wordCountActual.get() % 10000) {
				alpha = Constants.STARTING_ALPHA * (1 - wordCountActual.get() / (float) (trainWords + 1));
				if (alpha < Constants.STARTING_ALPHA * (float) 0.0001)
					alpha = Constants.STARTING_ALPHA * (float) 0.0001;
				
				System.out.println("Alpha:" + alpha + " Progress:" + (float) wordCountActual.get() / 
						(float) (trainWords + 1) * 100);
				System.out.flush();
			}

			if (null == (line = br.readLine()))
				break;
			
			for (int c = 0; c < Constants.LAYER_1_SIZE; c++)
				neu1[c] = 0;
			for (int c = 0; c < Constants.LAYER_1_SIZE; c++)
				neu1e[c] = 0;
			
			line = line.split("##")[0]; //把句子后面的注释去掉,"##"作为注释部分的分隔符
			String[] words = line.split("\\s+");
			int senLength = words.length;
			int senPosition = 0;
			for (; senPosition < senLength; senPosition++) {
				if (cbow) {
				} else {
					skipGram(words, senPosition);
				}
			}
		}
		
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	private void skipGram(String[] sentences, int senPosition) {
		int senLength = sentences.length;
		VocabWord word = vocab.get(sentences[senPosition]);
		if (null == word)
			return ;

		wordCountActual.incrementAndGet();
		nextRandom = Math.abs(nextRandom * (long) 25214903917l + 11);
		int b = (int) (nextRandom % (long) Constants.WINDOW);
		for (int a = b; a < Constants.WINDOW * 2 + 1 - b; a++) {
			if (a == Constants.WINDOW)
				continue;

			int c = senPosition - Constants.WINDOW + a;
			if (c < 0 || c >= senLength)
				continue;
			
			VocabWord lastWord = vocab.get(sentences[c]);
			if (null == lastWord)
				continue;
			
			int l1 = lastWord.getId() * Constants.LAYER_1_SIZE;
			for (c = 0; c < Constants.LAYER_1_SIZE; c++)
				neu1e[c] = (float) 0.0;
			
			if (hs) for (int d = 0; d < word.getCodeLen(); d++) {
				float f = (float) 0.0;
				int l2 = word.getPoint()[d] * Constants.LAYER_1_SIZE;
				for (c = 0; c < Constants.LAYER_1_SIZE; c++)
					f += syn0[c + l1] * syn1[c + l2];
				if (f <= -Constants.MAX_EXP)
					continue;
				else if (f >= Constants.MAX_EXP)
					continue;
				else 
					f = expTable[(int) ((f + (float) Constants.MAX_EXP) * (Constants.EXP_TABLE_SIZE /
							Constants.MAX_EXP / 2))];
				//坑死了，code是char，必须先减去'0'得到int类型的数据
				//float g = ((word.getCode()[d] - '0') - f) * f * (1 - f) * (float) alpha;
				float g = (1 - (word.getCode()[d] - '0') - f) * (float) alpha;
				//System.out.println("g=" + g + ", code=" + word.getCode()[d] + ", f=" + f);
				for (c = 0; c < Constants.LAYER_1_SIZE; c++)
					neu1e[c] += g * syn1[c + l2];
				for (c = 0; c < Constants.LAYER_1_SIZE; c++)
					syn1[c + l2] += g * syn0[c + l1];
			}
			
			//使用负例采样
			if (Constants.NEGATIVE > 0) {
				int targetWordId = 0;
				float label = (float) 0.0;
				for (int d = 0; d < Constants.NEGATIVE + 1; d++) {
					if (0 == d) {
						targetWordId = word.getId();
						label = (float) 1.0;
					} else {
						nextRandom = Math.abs(nextRandom * (long) 25214903917l + 11);
						int index = (int) (Math.abs(nextRandom >> 16) % table.length);
						if (0 == vocabSorted.get(table[index]).getWord().compareTo("</s>"))
							continue;
						targetWordId = vocabSorted.get(table[index]).getId();
						if (targetWordId == word.getId())
							continue;
						label = (float) 0.0;
					}
					
					int l2 = targetWordId * Constants.LAYER_1_SIZE;
					float f = 0, g = 0;
					for (c = 0; c < Constants.LAYER_1_SIZE; c++) 
						f += (syn0[c + l1] * syn1neg[c + l2]);
					if (f > Constants.MAX_EXP)
						g = (label - (float) 1) * alpha;
					else if (f < -Constants.MAX_EXP)
						g = (label - (float) 0) * alpha;
					else 
						g = (label - expTable[(int) ((f + Constants.MAX_EXP) * (Constants.EXP_TABLE_SIZE /
								Constants.MAX_EXP / 2))]) * alpha;
					for (c = 0; c < Constants.LAYER_1_SIZE; c++)
						neu1e[c] += (g * syn1neg[c + l2]);
					for (c = 0; c < Constants.LAYER_1_SIZE; c++)
						syn1neg[c + l2] += (g * syn0[c + l1]);
				}
			}

			for (c = 0; c < Constants.LAYER_1_SIZE; c++)
				syn0[c + l1] += neu1e[c];
		}
	}

	public int getThreadNo() {
		return threadNo;
	}
	public void setThreadNo(int threadNo) {
		this.threadNo = threadNo;
	}
	public int getLines() {
		return lines;
	}
	public void setLines(int lines) {
		this.lines = lines;
	}
}

