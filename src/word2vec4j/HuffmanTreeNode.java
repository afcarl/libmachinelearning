package word2vec4j;

/**
 * huffman树节点结构体
 * @author jiangwen
 *
 */
public class HuffmanTreeNode {

	//huffman树节点编码
	private int id;
	//词(必须是指针才能生效)
	private VocabWord vocabWord;
	//左孩子节点
	private HuffmanTreeNode left;
	//右孩子节点
	private HuffmanTreeNode right;
	//父节点
	private HuffmanTreeNode parent;
	
	public HuffmanTreeNode(VocabWord vocabWord, HuffmanTreeNode left, HuffmanTreeNode right,
			HuffmanTreeNode parent, int id) {
		this.vocabWord = vocabWord;
		this.left = left;
		this.right = right;
		this.parent = parent;
		this.id = id;
	}
	public VocabWord getVocabWord() {
		return vocabWord;
	}
	public void setVocabWord(VocabWord vocabWord) {
		this.vocabWord = vocabWord;
	}
	public HuffmanTreeNode getLeft() {
		return left;
	}
	public void setLeft(HuffmanTreeNode left) {
		this.left = left;
	}
	public HuffmanTreeNode getRight() {
		return right;
	}
	public void setRight(HuffmanTreeNode right) {
		this.right = right;
	}
	public HuffmanTreeNode getParent() {
		return parent;
	}
	public void setParent(HuffmanTreeNode parent) {
		this.parent = parent;
	}
	public int getId() {
		return id;
	}
	public void setId(int id) {
		this.id = id;
	}
}

