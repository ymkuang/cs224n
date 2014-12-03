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
		double eps_W = Math.sqrt(6) / Math.sqrt(wordSize * windowSize + hiddenSize);
		double eps_U = Math.sqrt(6) / Math.sqrt(hiddenSize + FeatureFactory.numType);

		W = SimpleMatrix.random(hiddenSize, wordSize * windowSize, -eps_W, eps_W, new Random());
		b_1 = new SimpleMatrix(hiddenSize, 1);
		U = SimpleMatrix.random(FeatureFactory.numType, hiddenSize, -eps_U, eps_U, new Random());
		b_2 = new SimpleMatrix(FeatureFactory.numType, 1);
		
		// Init L
		for (int i = 0; i < FeatureFactory.allVecs.numCols(); i++) {
			L.add(FeatureFactory.allVecs.extractVector(false, i));
		}
	}

    private List<List<Integer>> getWindows(List<Datum> data, List<Integer> sampleNums) {
	int windowRadius = windowSize/2;
	List<List<Integer>> windowSampleList = new ArrayList<List<Integer>>();
	for (int i = 0; i < sampleNums.size(); i++) {
		int sampleIndex=sampleNums.get(i);
		Datum instance = data.get(sampleIndex);
		LinkedList<String> window = new LinkedList<String>();
		window.add(instance.word);
		
		boolean sentenceLeft=instance.word.equals("<s>");
		boolean sentenceRight=instance.word.equals("</s>");

		//filling left
		for (int j=0;j<windowRadius;j++) {
			if (sentenceLeft || i-j-1<0) window.addFirst("<s>");
			else {
				window.addFirst(data.get(i-j-1).word);
				sentenceLeft= data.get(i-j-1).word.equals("<s>");
	
			}
		}
		
		//filling right
		int m =data.size();
                for (int j=0;j<windowRadius;j++) {
                        if (sentenceRight || i+j+1>=m) window.addFirst("</s>");
                        else {
                                window.addFirst(data.get(i+j+1).word);
                                sentenceRight= data.get(i+j+1).word.equals("</s>");                       

                        }
                }

		//string to num
		ArrayList<Integer> windowSample = new ArrayList<Integer>();
		for (String word: window) {
			windowSample.add(FeatureFactory.wordToNum.get(word));
		}
		windowSampleList.add(windowSample);
	}
	return windowSampleList;
    }

    // getWindow
    private SimpleMatrix getLVector(List<Integer> window){
		SimpleMatrix vec = new SimpleMatrix(wordSize * windowSize, 1);
		for (int w = 0; w < window.size(); ++w) {
			vec.insertIntoThis(w * wordSize, 0, L.get(w).extractVector(false, 0));
		return vec;
	}
	//added get windowed input

	private SimpleMatrix getWindowedSample(List<Datum> data, int sampleNum) {
		int m = data.size();
		SimpleMatrix windowSample = new SimpleMatrix(windowSize*wordSize,1);
		int range_window=(windowSize-1)/2;
		String sample;
		for (int i = -range_window; i <= range_window; i++) {
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
			Integer sampleOut = FeatureFactory.wordToNum.get(sample);
			if (sampleOut == null)
				sampleOut = 0;
			
			SimpleMatrix wordVec = FeatureFactory.allVecs.extractVector(false,sampleOut);
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
	public void train(List<Datum> _trainData ){
		List<Integer> samples;
		List<List<Integer>> windows;
		List<String> labels;

		// Check gradient
        // TO DO : get 10 windows and corresponding labels
        samples = getRandomSamples(_trainData, 10);
        windows = getWindows(_trainData, samples);
        labels = getLabels(_trainData, samples);
        gradient(windows, labels);
        if (!checkGradient(windows, labels)) 
        	System.out.print("Fail in gradient check.\n");

		// SGD
		initGradient();
		int pos = 0;
		while (true) {
			do {
                pos ++;
                if (pos == _trainData.size()) pos = 0;
			} while (_trainData.get(pos).word.equals("<s>") || _trainData.get(pos).word.equals("</s>"));
			
            samples.clear();
            samples.add(pos);
            windows = getWindows(_trainData, samples);
            labels = getLabels(_trainData, samples);
          
            // compute gradient
            gradient(windows, labels);
            double diff = updateParameter();
            if (diff < tol) break;
		}
	}

	
	public void test(List<Datum> testData){
		// TODO
		int numTest = testData.size();
		PrintWriter output = new PrintWriter("../result");
		// output according to example.out
		for (int i = 0; i < numTest; i ++) {
			Datum sample = testData.get(i);
			if (sample.word.equals("<s>") || sample.word.equals("</s>")) continue;
            SimpleMatrix currentL = getLVector(getWindow(testData, i));
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
    	cost += lambda / 2 / m * (normF(W) ^ 2 + normF(U) ^ 2);
    	return cost;
    }

    private double singleCost(String label) {
        int trueType = FeatureFactory.typeToNum.get(label);
        return -Math.log(p.get(trueType, 1));
    }

    private int findMax() {
    	double maxP = -1;
    	int maxType = -1;
    	for (int i = 0; i < p.numRows(); i ++) {
    		if (p.get(i, 1) > maxP) {
    			maxP = p.get(i, 1);
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
    			res.set(i, j, Math.tanh(mat.get(i,j)));
    		}
    	return res;
    }

	private SimpleMatrix maxEnt(SimpleMatrix v) {
		SimpleMatrix prob = new SimpleMatrix(v.numRows(), v.numCols());
		double sum = 0;
		for (int i = 0; i < v.numRows(); i ++) {
			sum += Math.exp(v.get(i, 1));
		}
		for (int i = 0; i < prob.numRows(); i ++) {
			prob.set(i, 1, Math.exp(v.get(i, 1)) / sum);
		}
		return prob;
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
			}

        for (int idx : idxMap.keySet()) {
        	L.get(idx).minus(dL.get(idxMap.get(idx)).scale(lr));
        	diff = Math.max(diff, dL.get(idxMap.get(idx)).extractMaxAbs() * lr);
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
