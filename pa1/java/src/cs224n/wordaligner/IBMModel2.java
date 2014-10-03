package cs224n.wordaligner;  

import cs224n.util.*;
import java.util.List;
import java.util.Set;
import java.util.Collection;
import java.util.HashSet;
import java.lang.Math;


/**
 * Simple word alignment baseline model that maps source positions to target 
 * positions along the diagonal of the alignment grid.
 * 
 * IMPORTANT: Make sure that you read the comments in the
 * cs224n.wordaligner.WordAligner interface.
 * 
 * @author Yuming Kuang
 */
public class IBMModel2 implements WordAligner {
  
  private class TriNumber {
    protected int i, n, m;

    public TriNumber(int newI, int newN, int newM) {
      i = newI;
      n = newN;
      m = newM;
    }

    public int getFirst() {
      return i;
    }

    public int getSecond() {
      return n;
    }

    public int getThird() {
      return m;
    }

    public boolean equals(Object o) {
      if (o == null) return false;
      if (!(o instanceof TriNumber)) return false;

      TriNumber other = (TriNumber) o;
      if (this.i != other.i) return false;
      if (this.n != other.n) return false;
      if (this.m != other.m) return false;

      return true;
    }

    public int hashCode() {
      return this.i * 100 + this.n * 100 + this.m;
    }
  }


  private static final long serialVersionUID = 1315751943476440515L;
  
  // TODO: Use arrays or Counters for collecting sufficient statistics
  // from the training data.
  private CounterMap<String, String> lexicalProb;
  private CounterMap<TriNumber, Integer> alignProb;

  public Alignment align(SentencePair sentencePair) {
    // Placeholder code below. 
    // TODO Implement an inference algorithm for Eq.1 in the assignment
    // handout to predict alignments based on the counts you collected with train().
    Alignment alignment = new Alignment();
    int m = sentencePair.getSourceWords().size();
    int n = sentencePair.getTargetWords().size();

    // Alignment for each target word
    for (int tgtIndex = 0; tgtIndex < n; tgtIndex++) {
      // Compute source word with largest t(e|f)
      int alignIndex = -1;
      double maxProb = -1.0;
      for (int srcIndex = -1; srcIndex < m; srcIndex++) {
        String sourceWord = srcIndex == -1 ? "" : sentencePair.getSourceWords().get(srcIndex);
        String targetWord = sentencePair.getTargetWords().get(tgtIndex);
        TriNumber position = new TriNumber(srcIndex, n, m);
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

  public void train(List<SentencePair> trainingPairs) {
    IBMModel1 ibmModel1 = new IBMModel1();
    ibmModel1.train(trainingPairs);
    // Initialize
    lexicalProb = ibmModel1.getLexicalProb();
    CounterMap<String, String> oldLexicalProb;
    alignProb = new CounterMap<TriNumber, Integer>();
    CounterMap<TriNumber, Integer> oldAlignProb;

    double delta = 1;
    int iter = 0;
    // Iteration
    while (delta > 1e-4 && iter < 50) {
      oldLexicalProb = lexicalProb;
      oldAlignProb = alignProb;
      lexicalProb = new CounterMap<String, String>();
      alignProb = new CounterMap<TriNumber, Integer>();
      iter ++;

      // E step
      for(SentencePair pair : trainingPairs){
        List<String> targetWords = pair.getTargetWords();
        List<String> sourceWords = pair.getSourceWords();
        int n = targetWords.size();
        int m = sourceWords.size();
        
        for (int tgtIndex = 0; tgtIndex < n; tgtIndex ++) {
          // Compute sum(t(e_i|f_j))
          String target = targetWords.get(tgtIndex);
          double totalProb = 0;
          for (int srcIndex = -1; srcIndex < m; srcIndex ++) {
            TriNumber position = new TriNumber(srcIndex, n, m);
            String source = (srcIndex == -1 ? "" : sourceWords.get(srcIndex));
            totalProb += oldLexicalProb.getCount(source, target) * (iter == 1 ? 1.0 / (m + 1) : oldAlignProb.getCount(position, tgtIndex));
          }
          // Update M step
          for (int srcIndex = -1; srcIndex < m; srcIndex ++) {
            TriNumber position = new TriNumber(srcIndex, n, m);
            String source = (srcIndex == -1 ? "" : sourceWords.get(srcIndex));
            double increment = oldLexicalProb.getCount(source, target) * (iter == 1 ? 1.0 / (m + 1) : oldAlignProb.getCount(position, tgtIndex)) / totalProb; 
            if (lexicalProb.getCount(source, target) == 0.0) {
              lexicalProb.setCount(source, target, increment);
            } else {
              lexicalProb.incrementCount(source, target, increment);
            }

            if (alignProb.getCount(position, tgtIndex) == 0.0) {
              alignProb.setCount(position, tgtIndex, increment);
            } else {
              alignProb.incrementCount(position, tgtIndex, increment);
            }
          }   
        }
      }
      // M step normalize
      lexicalProb = Counters.conditionalNormalize(lexicalProb);
      alignProb = Counters.conditionalNormalize(alignProb);
      
      // Compute difference
      if (iter == 1) delta = 1; 
      else {
        delta = computeDiff(lexicalProb, oldLexicalProb);
        delta += computeDiff(alignProb, oldAlignProb); 
      }
      System.out.printf("%f\n", delta);
    } 
  }

  private <K, V> double computeDiff(CounterMap<K, V> newCnt, CounterMap<K, V> oldCnt) {
    double delta = 0;
    int cnt = 0;    

    // All keys in two CounterMap
    Set<K> allKey = new HashSet<K>(newCnt.keySet());
    allKey.addAll(oldCnt.keySet());
  
    for (K key : allKey) {
      // All values for a key in two CounterMap
      Set<V> allValue = new HashSet<V>(newCnt.getCounter(key).keySet());
      allValue.addAll(oldCnt.getCounter(key).keySet());

      // Compute difference for each (key, value) pair
      for (V value : allValue) {
        delta += Math.abs(newCnt.getCount(key, value) - oldCnt.getCount(key, value));
        cnt ++;
      }
    }

    return delta / cnt; 
  }
}
