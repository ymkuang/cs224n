package cs224n.wordaligner;  

import cs224n.util.*;
import java.util.List;
import java.util.Set;
import java.util.Collection;
import java.util.HashSet;
import java.util.Random;
import java.lang.Math;


/**
 * IBM Model 2
 * 
 * @author Yuming Kuang
 */
public class IBMModel2 implements WordAligner {
  
  // Class for 3-integer group (i, n, m)
  // Used as key in q(j|(i, n, m)) (alignProb)
  private class TriNumber {
    protected int i, n, m;
    
    // Constructor
    public TriNumber(int newI, int newN, int newM) {
      i = newI;
      n = newN;
      m = newM;
    }
    
    // Get functions
    public int getFirst() {
      return i;
    }

    public int getSecond() {
      return n;
    }

    public int getThird() {
      return m;
    }

    // Equal function
    public boolean equals(Object o) {
      if (o == null) return false;
      if (!(o instanceof TriNumber)) return false;

      TriNumber other = (TriNumber) o;
      if (this.i != other.i) return false;
      if (this.n != other.n) return false;
      if (this.m != other.m) return false;

      return true;
    }

    // HashCode function
    public int hashCode() {
      return this.i * 100 + this.n * 10 + this.m;
    }
  }


  private static final long serialVersionUID = 1315751943476440515L;
  
  // Store t(e|f)
  private CounterMap<String, String> lexicalProb;
  // Store q(j|(i, n, m))
  private CounterMap<TriNumber, Integer> alignProb;
  // Random number generator
  private Random generator = new Random(17);
  // Convergence threshold
  private double tol = 1e-2;
  // Max number of iterations
  private int maxNumberIter = 1000;
  // Avoid denominator being too small
  private static final double EPS = 1e-10;
 
  public Alignment align(SentencePair sentencePair) {
    // Initialization
    Alignment alignment = new Alignment();
    int m = sentencePair.getSourceWords().size();
    int n = sentencePair.getTargetWords().size();

    // Alignment for each target word
    for (int tgtIndex = 0; tgtIndex < n; tgtIndex++) {
      // Compute source word with largest t(e|f) * q(j|(i, n, m))
      int alignIndex = -1;
      double maxProb = -1.0;
      for (int srcIndex = -1; srcIndex < m; srcIndex++) {
        // -1 indicates NULL word
        String sourceWord = srcIndex == -1 ? "" : sentencePair.getSourceWords().get(srcIndex);
        String targetWord = sentencePair.getTargetWords().get(tgtIndex);
        TriNumber position = new TriNumber(srcIndex, n, m);
        // Compute probability
        double prob = lexicalProb.getCount(sourceWord, targetWord) * alignProb.getCount(position, tgtIndex);
        if (prob > maxProb) {
           alignIndex = srcIndex;
           maxProb = prob;
        }
      } 

      // Add alignment
      if (alignIndex > -1) {
        alignment.addPredictedAlignment(tgtIndex, alignIndex);
      }
    }
    return alignment;
  }

  // Randomly initialize q(j|(i, n, m)), i.e alignProb
  private void initAlignProb(List<SentencePair> trainingPairs) {
    for (SentencePair pair : trainingPairs) {
      int n = pair.getTargetWords().size();
      int m = pair.getSourceWords().size();
      for (int tgtIndex = 0; tgtIndex < n; tgtIndex ++ ) {
        for (int srcIndex = -1; srcIndex < m; srcIndex ++) {
          TriNumber position = new TriNumber(srcIndex, n, m);
          if (alignProb.getCount(position, tgtIndex) == 0.0)
            // Random assign prob
            alignProb.setCount(position, tgtIndex, generator.nextDouble());
        }
      }
    }
    // Normalize
    alignProb = Counters.conditionalNormalize(alignProb);
  }

  public void train(List<SentencePair> trainingPairs) {
    // Initialize t(e|f) using IBM Model 1
    IBMModel1 ibmModel1 = new IBMModel1();
    ibmModel1.train(trainingPairs);
    lexicalProb = ibmModel1.getLexicalProb();

    // Random Initialize q(j|(i, n, m))
    alignProb = new CounterMap<TriNumber, Integer>();
    initAlignProb(trainingPairs);
    
    CounterMap<String, String> oldLexicalProb;
    CounterMap<TriNumber, Integer> oldAlignProb;

    double diffLexical = 1, diffAlign = 1;
    int iter = 0;
    // EM Iteration
    // Stop if both t(e|f) and q(j|(i, n, m)) converge 
    // or reach max iteration number
    while ((diffLexical > tol || diffAlign > tol) && iter < maxNumberIter) {
      // Initialize
      oldLexicalProb = lexicalProb;
      oldAlignProb = alignProb;
      lexicalProb = new CounterMap<String, String>();
      alignProb = new CounterMap<TriNumber, Integer>();
      iter ++;

      // Update for each sentence pair
      for(SentencePair pair : trainingPairs){
        List<String> targetWords = pair.getTargetWords();
        List<String> sourceWords = pair.getSourceWords();
        int n = targetWords.size();
        int m = sourceWords.size();
        
        for (int tgtIndex = 0; tgtIndex < n; tgtIndex ++) {
          String target = targetWords.get(tgtIndex);

          // Compute sum(t(e|f) * q(j|(i, n, m))) for fixed target on all possible source words
          double totalProb = 0;
          for (int srcIndex = -1; srcIndex < m; srcIndex ++) {
            TriNumber position = new TriNumber(srcIndex, n, m);
            // -1 indicates NULL word
            String source = (srcIndex == -1 ? "" : sourceWords.get(srcIndex));
            
            totalProb += oldLexicalProb.getCount(source, target) * oldAlignProb.getCount(position, tgtIndex);
          }

          // Avoid denominator being too small
          if (Math.abs(totalProb) < EPS) continue;          

          // Compute expectation of alignment and update the new estimate of t(e|f) and q(j|(i, n, m)) 
          for (int srcIndex = -1; srcIndex < m; srcIndex ++) {
            TriNumber position = new TriNumber(srcIndex, n, m);
            // -1 indicates NULL word
            String source = (srcIndex == -1 ? "" : sourceWords.get(srcIndex));
            // Compute expectation
            // all needed alignProb elements should have been initialized 
            double increment = oldLexicalProb.getCount(source, target) * 
                   oldAlignProb.getCount(position, tgtIndex) / totalProb; 
            
            // Deal with NaN case
            increment = Double.isNaN(increment) ? 0.0 : increment;
            // Increment in lexicalProb
            if (lexicalProb.getCount(source, target) == 0.0) {
              lexicalProb.setCount(source, target, increment);
            } else {
              lexicalProb.incrementCount(source, target, increment);
            }

            // Increment in alignProb
            if (alignProb.getCount(position, tgtIndex) == 0.0) {
              alignProb.setCount(position, tgtIndex, increment);
            } else {
              alignProb.incrementCount(position, tgtIndex, increment);
            }
          }   
        }
      }
      // Normalization
      lexicalProb = Counters.conditionalNormalize(lexicalProb);
      alignProb = Counters.conditionalNormalize(alignProb);
      
      // Compute difference
      // Use the mean change of non-zero elements in t and q
      if (iter == 1) diffLexical = diffAlign = 1; 
      else {
        diffLexical = computeDiff(lexicalProb, oldLexicalProb);
        diffAlign = computeDiff(alignProb, oldAlignProb); 
      }
      System.out.printf("Iteration: %d, diffs: %f  %f\n", iter, diffLexical, diffAlign);
    } 
  }

  // Compute the mean change of non-zero elements between two CounterMaps
  private <K, V> double computeDiff(CounterMap<K, V> newCnt, CounterMap<K, V> oldCnt) {
    double diff = 0.0;
    int cnt = 0;    
  
    // Get all keys
    Set<K> allKey = new HashSet<K>(newCnt.keySet());
    allKey.addAll(oldCnt.keySet());  

    for (K key : allKey) {
      // Get all values
      Set<V> allValue = new HashSet<V>(newCnt.getCounter(key).keySet());
      allValue.addAll(oldCnt.getCounter(key).keySet());

      for (V value : allValue) {
        diff += Math.abs(newCnt.getCount(key, value) - oldCnt.getCount(key, value));
        cnt ++;
      }
    }

    return diff / cnt; 
  }
}
