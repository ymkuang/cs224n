package cs224n.deep;
import java.lang.*;
import java.util.*;

import org.ejml.data.*;
import org.ejml.simple.*;
import org.apache.commons.math.util.FastMath;


import java.text.*;
import java.io.PrintWriter;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.lang.Math.*;

public class WindowModel {

	protected SimpleMatrix U, W, b_1, b_2, h, p, Wout;

	protected List<SimpleMatrix> L;
	
    private SimpleMatrix dU, db_2, dW, db_1;

    private HashMap<Integer, SimpleMatrix> dL;

    private final double lambda, lr, tol;
	//
	public int windowSize, wordSize, hiddenSize, maxIter;

	//more para
	public WindowModel(int _windowSize, int _hiddenSize, int maxIter, double _lr, double _reg){
        //TODO
        this.lr = _lr;
        this.lambda = _reg;
        this.tol = 1e-4;
        this.windowSize = _windowSize;
        this.hiddenSize = _hiddenSize;
		this.wordSize = 50;
        this.maxIter = maxIter;
    }

	/**
	 * Initializes the weights randomly. 
	 */
	public void initWeights(){
		System.out.print("Init Weights... \n");
		//TODO
		// initialize with bias inside as the last column
		// W = SimpleMatrix...
		// U for the score
		// U = SimpleMatrix..
		double eps_W = Math.sqrt(6) / Math.sqrt(wordSize * windowSize + hiddenSize);
		double eps_U = Math.sqrt(6) / Math.sqrt(hiddenSize + FeatureFactory.numType);

		W = SimpleMatrix.random(hiddenSize, wordSize * windowSize, -eps_W, eps_W, new Random(11));
		b_1 = new SimpleMatrix(hiddenSize, 1);
		U = SimpleMatrix.random(FeatureFactory.numType, hiddenSize, -eps_U, eps_U, new Random(13));
		b_2 = new SimpleMatrix(FeatureFactory.numType, 1);
		
		// Init L
		L = new ArrayList<SimpleMatrix>();
		for (int i = 0; i < FeatureFactory.allVecs.numCols(); i++) {
			L.add(FeatureFactory.allVecs.extractVector(false, i));
		}
	}

    private List<List<Integer>> getWindows(List<Datum> data, List<Integer> sampleNums) {
	    List<List<Integer>> windowSampleList = new ArrayList<List<Integer>>();
	    for (int i = 0; i < sampleNums.size(); i++) {
		    windowSampleList.add(getWindow(data, sampleNums.get(i)));
	    }
	    return windowSampleList;
    }

    // getWindow
    private List<Integer> getWindow(List<Datum> data, int sampleIndex) {
    	int windowRadius = windowSize / 2;
        Datum instance = data.get(sampleIndex);
		LinkedList<String> window = new LinkedList<String>();
		window.add(instance.word);
		
		boolean sentenceLeft = instance.word.equals("<s>");
		boolean sentenceRight = instance.word.equals("</s>");
        int i = sampleIndex;
		
		//filling left
		for (int j = 0; j < windowRadius; j++) {
			if (sentenceLeft || i-j-1 < 0) window.addFirst("<s>");
			else {
				window.addFirst(data.get(i-j-1).word);
				sentenceLeft = data.get(i-j-1).word.equals("<s>");
	
			}
		}
		
		//filling right
		int m = data.size();
        for (int j = 0; j < windowRadius; j++) {
            if (sentenceRight || i+j+1 >= m) window.add("</s>");
            else {
                window.add(data.get(i+j+1).word);
                sentenceRight = data.get(i+j+1).word.equals("</s>");                       

            }
        }

		//string to num
		ArrayList<Integer> windowSample = new ArrayList<Integer>();
		for (String word: window) {
			word = word.toLowerCase();
			if (FeatureFactory.wordToNum.containsKey(word)) {
			    windowSample.add(FeatureFactory.wordToNum.get(word));
			} else if (isNumeric(word)) {
                windowSample.add(FeatureFactory.wordToNum.get(FeatureFactory.NUMBER));
			} else {
				windowSample.add(FeatureFactory.wordToNum.get(FeatureFactory.UNKNOWN));
			}
		}
		return windowSample;
    }

    private SimpleMatrix getLVector(List<Integer> window){
		SimpleMatrix vec = new SimpleMatrix(wordSize * windowSize, 1);
		for (int i = 0; i < window.size(); ++i) {
			vec.insertIntoThis(i * wordSize, 0, L.get(window.get(i)).extractVector(false, 0));
		}
		return vec;
	}

	//somthing to get feature related to capticalization
	private int featureCap(String word){
		boolean firstCapital = false;
		int numCap = 0;
		for(int i=0; i<word.length(); i++){
			if(Character.isUpperCase(word.charAt(i))){
				numCap++;
				if(i==0)
					firstCapital = true;
			} else {
				break;
			}
		}
			
		if (numCap==0)
			return 0;
		
		if(numCap==1 && firstCapital)
			return 1;
	        
		if(numCap==word.length())
                        return 2;	
		//other
		return 3;
	}

    private List<String> getLabels(List<Datum> data, List<Integer> samples) {
    	List<String> labels = new ArrayList<String>();
    	for (int i = 0 ; i < samples.size(); i ++) {
    		labels.add(data.get(samples.get(i)).label);
    	}
    	return labels;
    }

	/**
	 * Simplest SGD training 
	 */
	public void train(List<Datum> _trainData )
        throws FileNotFoundException, IOException {
		System.out.print("Start Training: \n");
		List<Integer> samples;
		List<List<Integer>> windows;
		List<String> labels;

        System.out.print("Gradient checking... \n");
		// Check gradient
        // TO DO : get 10 windows and corresponding labels
        samples = getRandomSamples(_trainData, 10);
        windows = getWindows(_trainData, samples);
        labels = getLabels(_trainData, samples);
        initGradient();
        gradient(windows, labels);
        if (!checkGradient(windows, labels)) 
        	System.out.print("Fail in gradient check.\n");
        else
        	System.out.print("Success!\n");

		// SGD
        samples = new ArrayList<Integer>();
		initGradient();
		int pos = _trainData.size() - 1;
		int iter = 0;
		double oldCost = 1, cost = 0, diff = 0;
		boolean stop = false;
		while (true) {
			do {
                pos ++;
                if (pos == _trainData.size()) {
                	cost = allCostFunction(_trainData);
                	System.out.printf("\t Cost: %f \n", cost);
                    iter ++;
                	if (Math.abs(oldCost - cost) < tol || iter > maxIter) stop = true;
                	oldCost = cost;
                	diff = 0;
                	pos = 0;
                	System.out.printf("Iteration: %d ", iter);
                }	
			} while (_trainData.get(pos).word.equals("<s>") || _trainData.get(pos).word.equals("</s>"));
			
			if (stop) break;

            samples.clear();
            samples.add(pos);
            windows = getWindows(_trainData, samples);
            labels = getLabels(_trainData, samples);
          
            // compute gradient
            gradient(windows, labels);
            diff = Math.max(diff, updateParameter());
		}
        String fileName = "../train_result" + windowSize + hiddenSize + maxIter + lr + lambda; 
        fitResult(_trainData, fileName);
	}

	
	public void test(List<Datum> testData) 
        throws FileNotFoundException, IOException {
		// TODO
        String fileName = "../test_result" + windowSize + hiddenSize + maxIter + lr + lambda;
        fitResult(testData, fileName);
		
	}

    private void fitResult(List<Datum> data, String fileName) 
        throws FileNotFoundException, IOException {
        int numTest = data.size();
        PrintWriter output = new PrintWriter(fileName);
        // output according to example.out
        for (int i = 0; i < numTest; i ++) {
            Datum sample = data.get(i);
            if (sample.word.equals("<s>") || sample.word.equals("</s>")) continue;
            SimpleMatrix currentL = getLVector(getWindow(data, i));
            feedForward(currentL);
            String predLabel = FeatureFactory.typeList.get(findMax());
            output.printf("%s\t%s\t%s\n", sample.word, predLabel, sample.label);
        }
        output.close();

    }

    // compute h and p
    private void feedForward(SimpleMatrix currentL) {
    	h = matTanh(W.mult(currentL).plus(b_1));
    	SimpleMatrix v = U.mult(h).plus(b_2);
        p = maxEnt(v);
    } 
    // compute cost function given windows and true labels
    private double costFunction(List<List<Integer>> windows, List<String> labels) {
    	int m = windows.size();
    	double cost = 0;
    	for (int i = 0; i < m; i ++) {
    		SimpleMatrix currentL = getLVector(windows.get(i));
    		feedForward(currentL);
            cost += singleCost(labels.get(i));
    	}
    	cost /= m;
    	cost += lambda / 2 / m * (Math.pow(W.normF(), 2) + Math.pow(U.normF(), 2));
    	return cost;
    }

    private double allCostFunction(List<Datum> data) {
    	int m = 0;
    	double cost = 0;
    	for (int i = 0; i < data.size(); i ++) {
    		Datum sample = data.get(i);
    		if (sample.word.equals("<s>") || sample.word.equals("</s>"))
    			continue;
    		m ++;
    		SimpleMatrix currentL = getLVector(getWindow(data, i));
    		feedForward(currentL);
    		cost += singleCost(sample.label);
    	}
    	cost /= m;
    	cost += lambda / 2 / m * (Math.pow(W.normF(), 2) + Math.pow(U.normF(), 2));
    	return cost;
    }

    private double singleCost(String label) {
        int trueType = FeatureFactory.typeToNum.get(label);
        return -Math.log(p.get(trueType, 0));
    }

    private int findMax() {
    	double maxP = -1;
    	int maxType = -1;
    	for (int i = 0; i < p.numRows(); i ++) {
    		if (p.get(i, 0) > maxP) {
    			maxP = p.get(i, 0);
    			maxType = i;
    		}
    	}
    	return maxType;
    }

    private SimpleMatrix matTanh(SimpleMatrix mat) {
    	int numRows = mat.numRows();
    	int numCols = mat.numCols();
    	SimpleMatrix res = new SimpleMatrix(numRows, numCols);

    	for (int i = 0; i < numRows; i ++)
    		for (int j = 0; j < numCols; j ++) {
    			res.set(i, j, FastMath.tanh(mat.get(i,j)));
    		}
    	return res;
    }

	private SimpleMatrix maxEnt(SimpleMatrix v) {
		SimpleMatrix prob = new SimpleMatrix(v.numRows(), v.numCols());
		double sum = 0;
		for (int i = 0; i < v.numRows(); i ++) {
			sum += Math.exp(v.get(i, 0));
		}
		for (int i = 0; i < prob.numRows(); i ++) {
			prob.set(i, 0, Math.exp(v.get(i, 0)) / sum);
		}
		return prob;
	}
	
    private void gradient(List<List<Integer>> windows, List<String> labels) {
    	int m = windows.size();
    	zeroGradient();
    	for (int i = 0; i < windows.size(); i ++) {
    	  // Set y
    	  SimpleMatrix y = new SimpleMatrix(FeatureFactory.numType, 1);
    	  y.set(FeatureFactory.typeToNum.get(labels.get(i)), 0, 1);

          // Get window
          List<Integer> window = windows.get(i);
          SimpleMatrix currentL = getLVector(window);

          feedForward(currentL); //Get h, p
          
          // Second Layer
          SimpleMatrix delta_2 = p.minus(y);
          dU = dU.plus(delta_2.mult(h.transpose()));
          db_2 = db_2.plus(delta_2);

          // First Layer
          SimpleMatrix ones = new SimpleMatrix(hiddenSize, 1);
          ones.set(1);
          SimpleMatrix delta_1 = U.transpose().mult(delta_2).elementMult(ones.minus(h.elementMult(h)));
          dW = dW.plus(delta_1.mult(currentL.transpose()));
          db_1 = db_1.plus(delta_1);

          // Word Vector
          SimpleMatrix currentdL = W.transpose().mult(delta_1);
          for (int j = 0; j < window.size(); j ++) {
          	SimpleMatrix subdL = currentdL.extractMatrix(j * wordSize, (j + 1) * wordSize, 0, 1);
          	int idx = window.get(j);
            if (dL.containsKey(idx)) {
            	dL.put(idx, dL.get(idx).plus(subdL));
            } else {
            	dL.put(idx, subdL);
            }
          }
    
    	}

    	// Add regularization term
        dU = dU.divide(m).plus(U.divide(m / lambda));
        db_2 = db_2.divide(m);
        dW = dW.divide(m).plus(W.divide(m / lambda));
        db_1 = db_1.divide(m);
        
        
        for (Integer idx : dL.keySet()) {
        	dL.put(idx, dL.get(idx).divide(m));
        }
    }

    private double updateParameter() {
    	double diff = 0;
    	U = U.minus(dU.scale(lr));
        //diff = Math.max(diff, dU.elementMaxAbs() * lr);

    	b_2 = b_2.minus(db_2.scale(lr));
        //diff = Math.max(diff, db_2.elementMaxAbs() * lr);

    	W = W.minus(dW.scale(lr));
        //diff = Math.max(diff, dW.elementMaxAbs() * lr);

    	b_1 = b_1.minus(db_1.scale(lr));
        //diff = Math.max(diff, db_1.elementMaxAbs() * lr);

        for (Integer idx : dL.keySet()) {
        	L.get(idx).set(L.get(idx).minus(dL.get(idx).scale(lr)));
        	//diff = Math.max(diff, dL.get(idx).elementMaxAbs() * lr);
        }
        return diff;
    }

    private List<Integer> getRandomSamples(List<Datum> data, int size) {
    	List<Integer> samples = new ArrayList<Integer>();
    	Random rand = new Random(3);
    	for (int i = 0; i < size; i ++) {
            int sample = rand.nextInt(data.size());
            while (data.get(sample).word.equals("<s>") || data.get(sample).word.equals("</s>")) {
		         sample = rand.nextInt(data.size());
            }
            samples.add(sample);

    	}
    	return samples;
    }

    private boolean checkGradient(List<List<Integer>> windows, List<String> labels) {
    	double epsilon = 1e-4;
    	double maxDiff = 1e-7; 

        // check dU
        System.out.print("Checking dU...\n");
        for (int i = 0; i < U.numRows(); i ++) 
        	for (int j = 0; j < U.numCols(); j ++) {
        		double tmp = U.get(i, j);
        		U.set(i, j, tmp + epsilon);
        		double cost_plus = costFunction(windows, labels);
        		U.set(i, j, tmp - epsilon);
        		double cost_minus = costFunction(windows, labels);
        		U.set(i, j, tmp);
        		if (Math.abs((cost_plus - cost_minus) / (2 * epsilon) - dU.get(i, j)) > maxDiff) 
        			return false;
        	}
        // check db_2
        System.out.print("Checking db_2...\n");
        for (int i = 0; i < b_2.numRows(); i ++) {
        	double tmp = b_2.get(i, 0);
        	b_2.set(i, 0, tmp + epsilon);
        	double cost_plus = costFunction(windows, labels);
        	b_2.set(i, 0, tmp - epsilon);
        	double cost_minus = costFunction(windows, labels);
        	b_2.set(i, 0, tmp);
        	if (Math.abs((cost_plus - cost_minus) / (2 * epsilon) - db_2.get(i, 0)) > maxDiff)
        		return false;
        }

        // check dW
        System.out.print("Checking dW...\n");
        for (int i = 0; i < W.numRows(); i ++) 
        	for (int j = 0; j < W.numCols(); j ++) {
        		double tmp = W.get(i, j);
        		W.set(i, j, tmp + epsilon);
        		double cost_plus = costFunction(windows, labels);
        		W.set(i, j, tmp - epsilon);
        		double cost_minus = costFunction(windows, labels);
        		W.set(i, j, tmp);
        		if (Math.abs((cost_plus - cost_minus) / (2 * epsilon) - dW.get(i, j)) > maxDiff) 
        			return false;
        	}

        // check db_1
        System.out.print("Checking db_1...\n");
        for (int i = 0; i < b_1.numRows(); i ++) {
        	double tmp = b_1.get(i, 0);
        	b_1.set(i, 0, tmp + epsilon);
        	double cost_plus = costFunction(windows, labels);
        	b_1.set(i, 0, tmp - epsilon);
        	double cost_minus = costFunction(windows, labels);
        	b_1.set(i, 0, tmp);
        	if (Math.abs((cost_plus - cost_minus) / (2 * epsilon) - db_1.get(i, 0)) > maxDiff)
        		return false;
        }

        // check dL
        System.out.print("Checking dL...\n");
        for (Integer idx : dL.keySet()) {
            for (int i = 0; i < L.get(idx).numRows(); i ++) {
        	    double tmp = L.get(idx).get(i, 0);
        	    L.get(idx).set(i, 0, tmp + epsilon);
        	    double cost_plus = costFunction(windows, labels);
        	    L.get(idx).set(i, 0, tmp - epsilon);
        	    double cost_minus = costFunction(windows, labels);
        	    L.get(idx).set(i, 0, tmp);
        	    if (Math.abs((cost_plus - cost_minus) / (2 * epsilon) - dL.get(idx).get(i, 0)) > maxDiff) {
        	    	System.out.printf("%d %d\n", idx, i);
        	    	System.out.printf("%f %f %f %f\n", cost_plus, cost_minus, (cost_plus - cost_minus) / (2 * epsilon), dL.get(idx).get(i, 0));
        	    	return false;
        	    }
        		    
            }
        }

        return true;
    }

    private void initGradient() {
    	dU = new SimpleMatrix(FeatureFactory.numType, hiddenSize);
    	db_2 = new SimpleMatrix(FeatureFactory.numType, 1);
    	dW = new SimpleMatrix(hiddenSize, windowSize * wordSize);
    	db_1 = new SimpleMatrix(hiddenSize, 1);
    	
    	dL = new HashMap<Integer, SimpleMatrix>(); 
    }

    private void zeroGradient() {
    	dU.zero();
    	db_2.zero();
    	dW.zero();
    	db_1.zero();
    	
    	dL.clear();
    }

    private boolean isNumeric(String str) {
    	return str.matches("-?\\d+(\\.\\d+)?");
    }
}
