package cs224n.deep;

import java.util.*;
import java.io.*;

import org.ejml.simple.SimpleMatrix;


public class NER {
    
    public static void main(String[] args) throws IOException {
	if (args.length < 2) {
	    System.out.println("USAGE: java -Xmx4g -Xprof -cp \"classes:extlib/*\" cs224n.deep.NER ../data/train ../data/dev [windowSize=5] [hiddenlayerSize=100] [maxIteration=20] [learningRate=0.01] [regularization=0.0001]");
	    return;
	}	    

	//default parameter
	int windowSize = 5;
	int hiddenSize = 100;
	int maxIter = 20;
	double learningRate = 0.01;
	double regularization = 0.0001;
	//parse parameter
	if (args.length >= 3) 
		windowSize = Integer.parseInt(args[2]);
	if (args.length >= 4)
		hiddenSize = Integer.parseInt(args[3]);
	if (args.length >= 5)
		maxIter = Integer.parseInt(args[4]);
	if (args.length >= 6)
		learningRate = Double.parseDouble(args[5]);
	if (args.length >= 7)
		regularization = Double.parseDouble(args[6]);

	// this reads in the train and test datasets
	List<Datum> trainData = FeatureFactory.readTrainData(args[0]);
	List<Datum> testData = FeatureFactory.readTestData(args[1]);	
	
	//	read the train and test data
	//TODO: Implement this function (just reads in vocab and word vectors)
	FeatureFactory.initializeVocab("../data/vocab.txt");
	SimpleMatrix allVecs= FeatureFactory.readWordVectors("../data/wordVectors.txt");
	FeatureFactory.initType();

	//baseline
    /*    System.out.println("Baseline:");
        BaselineModel baseline = new BaselineModel();
        baseline.train(trainData);
        baseline.test(testData);
    */

    // One layer NN
	// initialize model 
	/*WindowModel model = new WindowModel(windowSize, hiddenSize, maxIter, learningRate, regularization);
	System.out.println("Current super-parameters used: Window size: " + windowSize + ", Hidden layer size: " + hiddenSize + ", Max Iteration: " + maxIter + ", Learning Rate: " + learningRate + " and regularization: " + regularization);
	model.initWeights();

	//train and test
	System.out.println("Train Network:");
	model.train(trainData);
 	System.out.println("Test Network:");
	model.test(testData);
    */

    // Two layer NN
    // initialize model
    TwoLayerModel twoLayerModel = new TwoLayerModel(5, 150, 100, 20, 0.01, 1e-4);
	twoLayerModel.initWeights();

	//train and test
	System.out.println("Train Network:");
	twoLayerModel.train(trainData);
 	System.out.println("Test Network:");
	twoLayerModel.test(testData);
    }
}
