package cs224n.deep;
import java.lang.*;
import java.util.*;

import org.ejml.data.*;
import org.ejml.simple.*;


import java.text.*;

public class WindowModel {

	protected SimpleMatrix L, U, W, b_1, b_2, h, p, Wout;
	
    private SimpleMatrix dU, db_2, dW, db_1, dL;
    private final double lambda, lr;
	//
	public int windowSize, wordSize, hiddenSize;

	public WindowModel(int _windowSize, int _hiddenSize, double _lr){
		//TODO
		lr = _lr;
		lambda = 0.01;
	}

	/**
	 * Initializes the weights randomly. 
	 */
	public void initWeights(){
		//TODO
		// initialize with bias inside as the last column
		// W = SimpleMatrix...
		// U for the score
		// U = SimpleMatrix...
	}


	/**
	 * Simplest SGD training 
	 */
	public void train(List<Datum> _trainData ){
		//	TODO
	}

	
	public void test(List<Datum> testData){
		// TODO
		}
	
    private void gradient(List<Datum> data) {
    	int m = data.size();
    	zeroGradient();
    	for (Datum datum : data) {
    	  // Set y
    	  SimpleMatrix y = new SimpleMatrix(FeatureFactory.numType, 1);
    	  y.set(FeatureFactory.typeToNum.get(datum.label), 1, 1);

          feedForward(datum); //Get L, h, p
          SimpleMatrix delta_2 = p.minus(y);
          dU = dU.plus(delta_2.mult(h.transpose()));
          db_2 = db_2.plus(delta_2);

          SimpleMatrix ones = SimpleMatrix(hiddenSize, 1);
          ones.set(1);
          SimpleMatrix delta_1 = U.transpose().mult(delta_2).elementMult(ones.minus(h.elementMult(h)));
          dW = dW.plus(delta_1.mult(L));
          db_1 = db_1.plus(delta_1);
          dL = dL.plus(W.transpose().mult(delta_1));

    	}
        dU = dU.divide(m).plus(U.divide(m / lambda));
        db_2 = db_2.divide(m);
        dW = dW.divide(m).plus(W.divide(m / lambda));
        db_1 = db_1.divide(m);
        dL = dL.divide(m);
    }

    private void initGradient() {
    	dU = new SimpleMatrix(FeatureFactory.numType, hiddenSize);
    	db_2 = new SimpleMatrix(FeatureFactory.numType, 1);
    	dW = new SimpleMatrix(hiddenSize, windowSize * wordSize);
    	db_1 = new SimpleMatrix(hiddenSize, 1);
    	dL = new SimpleMatrix(windowSize * wordSize, 1);
    }

    private void zeroGradient() {
    	dU.zero();
    	db_2.zero();
    	dW.zero();
    	db_1.zero();
    	dL.zero();
    }
}
