package cs224n.deep;
import java.lang.*;
import java.util.*;

import org.ejml.data.*;
import org.ejml.simple.*;


import java.text.*;
import java.lang.Math.*;

public class WindowModel {

	protected SimpleMatrix U, W, b_1, b_2, h, p, Wout;

	protected List<SimpleMatrix> L;
	
    private SimpleMatrix dU, db_2, dW, db_1;

    private List<SimpleMatrix> dL;
    private HashMap<Integer, Integer> idxMap;

    private final double lambda, lr, tol, reg;
	//
	public int windowSize, wordSize, hiddenSize;

	public WindowModel(int _windowSize, int _hiddenSize, double _lr){
		//TODO
		this.lr = _lr;
		this.lambda = 0.01;
		this.tol = 1e-4;
		this.windowSize=_windowSize;
		this.hiddenSize=_hiddenSize;
		this.wordSize = 50;
	}
	//more para
	public WindowModel(int _windowSize, int _hiddenSize, double _lr, double _reg){
                //TODO
                this.lr = _lr;
                this.lambda = _reg;
                this.tol = 1e-4;
		this.reg = _reg;
                this.windowSize=_windowSize;
                this.hiddenSize=_hiddenSize;
		this.wordSize = 50;
        }

	/**
	 * Initializes the weights randomly. 
	 */
	public void initWeights(){
		//TODO
		// initialize with bias inside as the last column
		// W = SimpleMatrix...
		// U for the score
		// U = SimpleMatrix..
		double eps_W = Math.sqrt(6)/ Math.sqrt(wordSize*windowSize+hiddenSize);
		double eps_U = Math.sqrt(6)/ Math.sqrt(hiddenSize+1);

		W = SimpleMatrix.random(hiddenSize,wordSize*windowSize, -eps_W,eps_W, new Random());
		b_1 = new SimpleMatrix(hiddenSize,1);
		U = SimpleMatrix.random(1,hiddenSize, -eps_U, eps_U, new Random());
		b_2 = new SimpleMatrix(1,1);
		
	}

	//added get windowed input
	private SimpleMatrix getWindowedSample(List<Datum> data, int sampleNum) {
		int m = data.size();
		SimpleMatrix windowSample = new SimpleMatrix(windowSize*wordSize,1);
		int range_window=(windowSize-1)/2;
		String sample;
		for (int i = -range_window;i<=range_window;i++) {
			int cap = 0;
			if (sampleNum+i<0) {
				sample = "<s>";
			} else {
				if (sampleNum+i>=m) {
					sample = "</s>";
				} else {
					sample = data.get(sampleNum+i).word;
					cap = featureCap(sample);
					sample = sample.toLowerCase();
				}
			}
			Integer sampleNum = FeatureFactory.wordToNum.get(sample);
			if (sampleNum == null)
				sampleNum = 0;
			
			SimpleMatrix wordVec = FeatureFactory.allVecs.extractVector(false,sampleNum);
			windowSample.insertIntoThis((i*range_window)*(wordSize),0,wordVec);
		}
		return windowSample;
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

	/**
	 * Simplest SGD training 
	 */
	public void train(List<Datum> _trainData ){
		// Check gradient
        // TO DO : get 10 windows and corresponding labels
        gradient(windows, labels);
        if (!checkGradient(windows, labels)) 
        	System.out.print("Fail in gradient check.\n");

		// SGD
		initGradient();
		int pos = 0;
		while (true) {
            // TO DO: Get next window and label and update pos
          
            // compute gradient
            gradient(windows, labels);
            double diff = updateParameter();
            if (diff < tol) break;
		}
	}

	
	public void test(List<Datum> testData){
		// TODO
		int numTest = testData.size();
		double numCorrect=0;
		double numReturned=0;
		double  numGold=0;
		for (int indTest=0; indTest<numTest;indTest++) {
			SimpleMatrix mat_input = getWindowedSample(testData,indTest);
			SimpleMatrix mat_hiddenin = tanh((W.mult(mat_input)).plus(b1));
			double h = matSigmoid(U.mult(mat_hiddenin).plus(b_2)).get(0,0);
			Integer vecNum = FeatureFactory.wordToNum.get(testData.get(indTest).word.toLowerCase());
		        if (testData.get(indTest).label.equals("O")==false) {
                        	numGold++;
				if (h>0.5)
					numCorrect++;
			}	
			if (h>0.5)
				numReturned++;
		}
		double precision = numCorrect/numReturned;
		double recall = numCorrect/numGold;
		double F1=2*(precision*recall)/(precision+recall);

		System.out.println("Testing with data, F1 score: " + F1 + " with precision: " + precision + " recall: " + recall);
	}

	//sigmoid function of matrix (elementwise)
	private SimpleMatrix matSigmoid(SimpleMatrix mat) {
		int numRows = mat.numRows();
		int numCols = mat.numCols();
		SimpleMatrix sig = new SimpleMatrix(numRows, numCols);

		for (int row = 0; row < numRows; row++) {
			for (int col = 0; col < numCols; col++) {
				double sigVal = mat.get(row, col);
			}
		}

		return sig;
	}
	
    private void gradient(List<List<Integer>> windows, List<String> labels) {
    	int m = windows.size();
    	zeroGradient();
    	for (int i = 0; i < windows.size(); i ++) {
    	  // Set y
    	  SimpleMatrix y = new SimpleMatrix(FeatureFactory.numType, 1);
    	  y.set(FeatureFactory.typeToNum.get(labels.get(i)), 1, 1);

          // Get window
          List<Integer> window = windows.get(i);
          SimpleMatrix currentL = getLVector(window);

          feedForward(currentL); //Get h, p
          
          // Second Layer
          SimpleMatrix delta_2 = p.minus(y);
          dU = dU.plus(delta_2.mult(h.transpose()));
          db_2 = db_2.plus(delta_2);

          // First Layer
          SimpleMatrix ones = SimpleMatrix(hiddenSize, 1);
          ones.set(1);
          SimpleMatrix delta_1 = U.transpose().mult(delta_2).elementMult(ones.minus(h.elementMult(h)));
          dW = dW.plus(delta_1.mult(currentL));
          db_1 = db_1.plus(delta_1);

          // Word Vector
          SimpleMatrix currentdL = W.transpose().mult(delta_1);
          for (int j = 0; j < window.size(); j ++) {
          	SimpleMatrix subdL = currentdL.extractMatrix(j * wordSize, (j + 1) * wordSize, 0, 1);
          	int idx = widnow.get(j);
            if (idxMap.containsKey(idx)) {
            	dL.get(idxMap.get(idx)).plus(subdL);
            } else {
            	dL.add(subdL);
            	idxMap.put(idx, dL.size() - 1);
            }
          }

    	}

    	// Add regularization term
        dU = dU.divide(m).plus(U.divide(m / lambda));
        db_2 = db_2.divide(m);
        dW = dW.divide(m).plus(W.divide(m / lambda));
        db_1 = db_1.divide(m);
        
        for (int i = 0; i < dL.size(); i ++) {
        	dL.get(i).divide(m);
        }
    }

    private double updateParameter() {
    	double diff = 0;
    	U = U.minus(dU.scale(lr));
        diff = Math.max(diff, dU.extractMaxAbs() * lr);

    	b_2 = b_2.minus(db_2.scale(lr));
        diff = Math.max(diff, db_2.extractMaxAbs() * lr);

    	W = W.minus(dW.scale(lr));
        diff = Math.max(diff, dW.extractMaxAbs() * lr);

    	b_1 = b_1.minus(db_1.scale(lr));
        diff = Math.max(diff, db_1.extractMaxAbs() * lr);

        for (int idx : idxMap.keySet()) {
        	L.get(idx).minus(dL.get(idxMap.get(idx)).scale(lr));
        	diff = Math.max(diff, dL.get(idxMap.get(idx)).extractMaxAbs() * lr);
        }
        return diff;
    }

    private boolean checkGradient(List<List<Integer>> windows, List<String> labels) {
    	double epsilon = 1e-4;
    	double maxDiff = 1e-8; 

        // check dU
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
        for (int i = 0; i < b_2.numRows(); i ++) {
        	double tmp = b_2.get(i, 1);
        	b_2.set(i, 1, tmp + epsilon);
        	double cost_plus = costFunction(windows, labels);
        	b_2.set(i, 1, tmp - epsilon);
        	double cost_minus = costFunction(windows, labels);
        	b_2.set(i, 1, tmp);
        	if (Math.abs((cost_plus - cost_minus) / (2 * epsilon) - db_2.get(i, 1)) > maxDiff)
        		return false;
        }

        // check dW
        for (int i = 0; i < W.numRows(); i ++) 
        	for (int j = 0; j < W.numCols(); j ++) {
        		double tmp = W.get(i, j);
        		W.set(i, j, tmp + epsilon);
        		double cost_plus = costFunction(windows, labels);
        		W.set(i, j, tmp - epsilon);
        		double cost_minus = costFunction(windoes, labels);
        		W.set(i, j, tmp);
        		if (Math.abs((cost_plus - cost_minus) / (2 * epsilon) - dW.get(i, j)) > maxDiff) 
        			return false;
        	}

        // check db_1
        for (int i = 0; i < b_1.numRows(); i ++) {
        	double tmp = b_1.get(i, 1);
        	b_1.set(i, 1, tmp + epsilon);
        	double cost_plus = costFunction(windows, labels);
        	b_1.set(i, 1, tmp - epsilon);
        	double cost_minus = costFunction(windows, labels);
        	b_1.set(i, 1, tmp);
        	if (Math.abs((cost_plus - cost_minus) / (2 * epsilon) - db_1.get(i, 1)) > maxDiff)
        		return false;
        }

        // check dL
        for (int idx : idxMap.keySet()) {
            SimpleMatrix currentL = L.get(idx);
            for (int i = 0; i < currentL.numRows(); i ++) {
        	    double tmp = currentL.get(i, 1);
        	    currentL.set(i, 1, tmp + epsilon);
        	    double cost_plus = costFunction(windows, labels);
        	    currentL.set(i, 1, tmp - epsilon);
        	    double cost_minus = costFunction(windows, labels);
        	    currentL.set(i, 1, tmp);
        	    if (Math.abs((cost_plus - cost_minus) / (2 * epsilon) - dL.get(idxMap.get(idx)).get(i, 1)) > maxDiff)
        		    return false;
            }
        }

        return true;
    }

    private void initGradient() {
    	dU = new SimpleMatrix(FeatureFactory.numType, hiddenSize);
    	db_2 = new SimpleMatrix(FeatureFactory.numType, 1);
    	dW = new SimpleMatrix(hiddenSize, windowSize * wordSize);
    	db_1 = new SimpleMatrix(hiddenSize, 1);
    	
    	dL = new ArrayList<SimpleMatrix>();
    	idxMap = new HashMap<Integer, Integer>(); 
    }

    private void zeroGradient() {
    	dU.zero();
    	db_2.zero();
    	dW.zero();
    	db_1.zero();
    	
    	dL.clear();
    	idxMap.clear();
    }
}
