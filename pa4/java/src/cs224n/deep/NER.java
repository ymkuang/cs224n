package cs224n.deep;

import java.util.*;
import java.io.*;

import org.ejml.simple.SimpleMatrix;


public class NER {
    
    public static void main(String[] args) throws IOException {
	if (args.length < 2) {
	    System.out.println("USAGE: java -cp classes NER ../data/train ../data/dev [windowSize=5] [hiddenlayerSize=100] [learningRate=0.01] [regularization=0.0001]");
	    return;
	}	    

	//default parameter
	int windowSize = 5;
	int hiddenSize = 100;
	double learningRate = 0.01;
	double regularization = 0.0001;
	//parse parameter
	if (args.length >= 3) 
		windowSize = Integer.parseInt(args[2]);
	if (args.length >= 4)
		hiddenSize = Integer.parseInt(args[3]);
	if (args.length >= 5)
		learningRate = Double.parseDouble(args[4]);
	if (args.length >= 6)
		regularization = Double.parseDouble(args[5]);

	// this reads in the train and test datasets
	List<Datum> trainData = FeatureFactory.readTrainData(args[0]);
	List<Datum> testData = FeatureFactory.readTestData(args[1]);	
	
	//	read the train and test data
	//TODO: Implement this function (just reads in vocab and word vectors)
	FeatureFactory.initializeVocab("../data/vocab.txt");
	SimpleMatrix allVecs= FeatureFactory.readWordVectors("../data/wordVectors.txt");

	// initialize model 
	WindowModel model = new WindowModel(windowSize, hiddenSize, learningRate, regularization);
	System.out.println("Current super-parameters used: Window size: " + windowSize + ", Hidden layer size: " + hiddenSize + ", Learning Rate: " + learningRate + " and regularization: " + regularization);
	model.initWeights();
	
	//train and test
	System.out.println("Train:");
	model.train(trainDate);
 	System.out.println("Test:");
	model.test(testData);
    }
}
