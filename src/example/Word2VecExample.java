package example;

import java.io.IOException;

import word2vec4j.Word2Vec;

/**
 * 由于paragraph2vec的代码是在word2vec的基础上写的,引用了很多word2vec的训练结果,
 * 所以代码没有完全拆分开,trainParagraphModel放在了word2vec里面
 * 具体看下代码就明白了,本身pragraph2vec和word2vec就是非常相似的
 * @author J.
 *
 */
public class Word2VecExample {
	private String trainFile = "D:/projectdata/libself/phrases.txt";
	private String wmodelFile = "D:/projectdata/libself/word2vec.model";
	private String pmodelFile = "D:/projectdata/libself/paragrahvec.model";
	public void run() throws IOException, InterruptedException {
		Word2Vec word2vec = new Word2Vec(trainFile, wmodelFile);
		word2vec.init();
		word2vec.trainModel();
		System.out.println("word2vec train done ................................");
		//训练句子向量,结果在内存中,还未写到文件里
		word2vec.trainParagraphModel(trainFile, pmodelFile);
		System.out.println("paragraph2vec train done ................................");
	}

	public static void main(String[] args) throws IOException, InterruptedException {
		Word2VecExample example = new Word2VecExample();
		example.run();
	}
}
