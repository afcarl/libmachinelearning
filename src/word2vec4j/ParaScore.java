package word2vec4j;

public class ParaScore {

	//切分后的句子
	private String paragraph;
	//Rank得分
	private Float score;

	public ParaScore(String paragString, float score) {
		this.paragraph = paragString;
		this.score = score;
	}

	public String getParagraph() {
		return paragraph;
	}
	public void setParagraph(String paragraph) {
		this.paragraph = paragraph;
	}
	public Float getScore() {
		return score;
	}
	public void setScore(Float score) {
		this.score = score;
	}
	
}
