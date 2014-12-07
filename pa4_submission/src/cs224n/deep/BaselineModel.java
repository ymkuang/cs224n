package cs224n.deep;
import java.lang.*;
import java.util.*;

import org.ejml.data.*;
import org.ejml.simple.*;


import java.text.*;
import java.io.PrintWriter;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.lang.Math.*;

public class BaselineModel {

    private HashMap<String, String> wordMap;

	public BaselineModel(){
		wordMap = new HashMap<String, String>();
	}

	/**
	 * Simplest baseline training 
	 */
	public void train(List<Datum> trainData ) {
		int numTrain = trainData.size();
		for (int i=0;i<numTrain;i++) {
			Datum sample = trainData.get(i);
			String word = sample.word;
			String label = sample.label;
			if (!wordMap.containsKey(word))
				wordMap.put(word,label);
		}
	}

	
	public void test(List<Datum> testData)
	    throws FileNotFoundException, IOException {
		int numTest = testData.size();
		PrintWriter output = new PrintWriter("../baseline");
		// output according to example.out
		for (int i = 0; i < numTest; i ++) {
			Datum sample = testData.get(i);
			//find sample in train
            		String predLabel = "O";
			if (wordMap.containsKey(sample.word))
				predLabel=wordMap.get(sample.word);
            		output.printf("%s\t%s\t%s\n", sample.word, predLabel, sample.label);
		}
        	output.close();
	}

}
