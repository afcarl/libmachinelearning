package word2vec4j;

public class VocabWord {

	//词
	private String word;
	//词的出现频率计数
	private long cn;
	//词的huffman编码,调整成数组？
	private char[] code;
	//huffman编码的长度
	private int codeLen;
	//huffman树内部所有节点的编号
	private int[] point;
	//词id，每个词有一个唯一的id号
	private int id;
	
	public VocabWord(String word, long cn, int codeLen, int id) {
		this.word = word;
		this.cn = cn;
		this.codeLen = codeLen;
		this.id = id;
		this.code = new char[Constants.MAX_CODE_LENGTH];
		this.point = new int[Constants.MAX_CODE_LENGTH];
	}
	
	public VocabWord(String word, long cn, int codeLen, int id, char[] code, int[] point) {
		this.word = word;
		this.cn = cn;
		this.codeLen = codeLen;
		this.id = id;
		this.code = code;
		this.point = point;
	}
	
	public String toString() {
		String desc = "";
		desc += "word[";
		desc += word;
		desc += "] cn[";
		desc += cn;
		desc += "]";
		return desc;
	}

	public String getWord() {
		return word;
	}
	public void setWord(String word) {
		this.word = word;
	}
	public long getCn() {
		return cn;
	}
	public void setCn(long cn) {
		this.cn = cn;
	}
	public int getCodeLen() {
		return codeLen;
	}
	public void setCodeLen(int codeLen) {
		this.codeLen = codeLen;
	}
	public int getId() {
		return id;
	}
	public void setId(int id) {
		this.id = id;
	}
	public int[] getPoint() {
		return point;
	}
	public void setPoint(int[] point) {
		this.point = point;
	}
	public char[] getCode() {
		return code;
	}
	public void setCode(char[] code) {
		this.code = code;
	}
	
}

