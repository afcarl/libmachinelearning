package word2vec4j;

public class Vector {

	//词
	private String word;
	//向量
	private float[] vec;
	//向量相似度得分
	private Float score;
	
	public Vector(String word, float[] vec, Float score) {
		this.word = word;
		this.vec = vec;
		this.score = score;
	}
	public String getWord() {
		return word;
	}
	public void setWord(String word) {
		this.word = word;
	}
	public float[] getVec() {
		return vec;
	}
	public void setVec(float[] vec) {
		this.vec = vec;
	}
	public Float getScore() {
		return score;
	}
	public void setScore(Float score) {
		this.score = score;
	}
}

