package cs224n.deep;
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;
import org.ejml.simple.*;


public class FeatureFactory {


	private FeatureFactory() {

	}

	 
	static List<Datum> trainData;
	/** Do not modify this method **/
	public static List<Datum> readTrainData(String filename) throws IOException {
        if (trainData==null) trainData= read(filename);
        return trainData;
	}
	
	static List<Datum> testData;
	/** Do not modify this method **/
	public static List<Datum> readTestData(String filename) throws IOException {
        if (testData==null) testData= read(filename);
        return testData;
	}
	
	private static List<Datum> read(String filename)
			throws FileNotFoundException, IOException {
	    // TODO: you'd want to handle sentence boundaries
		List<Datum> data = new ArrayList<Datum>();
		BufferedReader in = new BufferedReader(new FileReader(filename));
		for (String line = in.readLine(); line != null; line = in.readLine()) {
			//handle sentence boundaries;
			String word;
			String label;
			if (line.trim().length() == 0) {
			// blank line
				data.add(new Datum("</s>", "O"));
				data.add(new Datum("<s>", "O"));
			} else {
				String[] bits = line.split("\\s+");
				word = bits[0];
				label = bits[1];
			
				Datum datum = new Datum(word, label);
				data.add(datum);
		    } 
		}
		in.close();
		return data;
	}
 
    static List<String> typeList = new ArrayList<String>();
    static HashMap<String, Integer> typeToNum = new HashMap<String, Integer>();
    static int numType;
    public static void initType() {
    	typeList.add("O");
    	typeList.add("LOC");
    	typeList.add("MISC");
    	typeList.add("ORG");
    	typeList.add("PER");

    	typeToNum.put("O", 0);
    	typeToNum.put("LOC", 1);
    	typeToNum.put("MISC", 2);
    	typeToNum.put("ORG", 3);
    	typeToNum.put("PER", 4);
    	
    	numType = 5;
    }


	// Look up table matrix with all word vectors as defined in lecture with dimensionality n x |V|
	static SimpleMatrix allVecs; //access it directly in WindowModel
	public static SimpleMatrix readWordVectors(String vecFilename) throws IOException {
		if (allVecs!=null) return allVecs;
		//return null;
		//TODO implement this
		//set allVecs from filename		
		
		//read data from file
		BufferedReader in = new BufferedReader(new FileReader(vecFilename));
		ArrayList<String[]> wordVecsArray = new ArrayList<String[]>();
		
		for (String line = in.readLine(); line != null; line = in.readLine()){
			String[] bits = line.split("\\s+");
			wordVecsArray.add(bits);			
		}
		
		//setup allVecs
		int numRows = wordVecsArray.get(0).length;
		int numCols = wordVecsArray.size(); 
		allVecs = new SimpleMatrix(numRows, numCols);
		
		for(int col=0; col<numCols; col++){
			String[] vec = wordVecsArray.get(col);
			for(int row=0; row<numRows; row++){
				allVecs.set(row,col,new Double(vec[row]));
			}
		}
		in.close();

		//finish read
		System.out.println("readWordVector into " + numRows + "*" + numCols + " vector matrix.");
		return allVecs;
	}

    // Unknown word
	public static final String UNKNOWN = "UUUNKKK";
	// might be useful for word to number lookups, just access them directly in WindowModel
	public static HashMap<String, Integer> wordToNum = new HashMap<String, Integer>(); 
	public static HashMap<Integer, String> numToWord = new HashMap<Integer, String>();

	public static HashMap<String, Integer> initializeVocab(String vocabFilename) throws IOException {
		//TODO: create this
		BufferedReader in = new BufferedReader(new FileReader(vocabFilename));
		int indLine = 0;
		for (String line = in.readLine(); line!=null; line=in.readLine()){
			wordToNum.put(line,indLine);
			numToWord.put(indLine,line);
			indLine++;
		}
		in.close();
		return wordToNum;
	}
 








}
