package cs224n.wordaligner;  

import cs224n.util.*;
import java.util.List;
import java.util.Set;
import java.util.Collection;
import java.util.HashSet;
import java.lang.Math;

/**
 * IBM Model 1 
 * 
 * @author Yuming Kuang
 */
public class IBMModel1 implements WordAligner {

  private static final long serialVersionUID = 1315751943476440515L;
  
  // Store t(e|f)
  private CounterMap<String,String> lexicalProb;
  // Iteration stop criterion threshold for convergence
  private double tol = 1e-2;
  // Max number iterations
  private int maxNumberIter = 1000;

  public CounterMap<String, String> getLexicalProb() {
    return lexicalProb;
  }

  public Alignment align(SentencePair sentencePair) {
    // Initialization
    Alignment alignment = new Alignment();
    int numSourceWords = sentencePair.getSourceWords().size();
    int numTargetWords = sentencePair.getTargetWords().size();

    // Alignment for each target word
    for (int tgtIndex = 0; tgtIndex < numTargetWords; tgtIndex++) {
      // Compute source word with largest t(e|f)
      int alignIndex = -1;
      double maxProb = -1.0;
      for (int srcIndex = -1; srcIndex < numSourceWords; srcIndex++) {
        String sourceWord = srcIndex == -1 ? "" : sentencePair.getSourceWords().get(srcIndex);
        String targetWord = sentencePair.getTargetWords().get(tgtIndex);
        double prob = lexicalProb.getCount(sourceWord, targetWord);
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
    // Initialize
    lexicalProb = new CounterMap<String, String>();
    CounterMap<String, String> oldLexicalProb;
    double diff = 1;
    int iter = 0;
    // EM Iteration 
    // Stop if converge or reach max iteration number
    while (diff > tol && iter < maxNumberIter) {
      // Initialization
      oldLexicalProb = lexicalProb;
      lexicalProb = new CounterMap<String, String>();
      iter ++;

      // Train on each sentence
      for(SentencePair pair : trainingPairs){
        List<String> targetWords = pair.getTargetWords();
        List<String> sourceWords = pair.getSourceWords();
      
        for (String target : targetWords) {
          // Compute sum of t(e|f) for fixed target word on all source words
          double totalProb = 0;
          for (int srcIndex = -1; srcIndex < sourceWords.size(); srcIndex ++) {
            // -1 indicates NULL word
            String source = (srcIndex == -1 ? "" : sourceWords.get(srcIndex));
            // For 1st iteration, initialize with uniform prob
            totalProb += (iter == 1 ? 1 : oldLexicalProb.getCount(source, target));
          }
          // Compute the expectation of alignment and count in new t(e|f) estimate
          for (int srcIndex = -1; srcIndex < sourceWords.size(); srcIndex ++) {
            // -1 for NULL word
            String source = (srcIndex == -1 ? "" : sourceWords.get(srcIndex));
            // Initialize with uniform prob for 1st iter
            double increment = (iter == 1 ? 1 : oldLexicalProb.getCount(source, target)) / totalProb; 
            // Do the increment
            if (lexicalProb.getCount(source, target) == 0.0) {
              lexicalProb.setCount(source, target, increment);
            } else {
              lexicalProb.incrementCount(source, target, increment);
            }
          }   
        }
      }
      // normalize t(e|f) count to probability
      lexicalProb = Counters.conditionalNormalize(lexicalProb);
      
      // Compute difference
      // delta store the mean change of non-zero element in t(e|f)
      diff = iter == 1 ? 1 : computeDiff(lexicalProb, oldLexicalProb);
      System.out.printf("Iteration: %d, diff: %f\n", iter, diff);
    } 
  }

  // Compute the mean change of non-zero elements of two CounterMap 
  private <K, V> double computeDiff(CounterMap<K, V> newCnt, CounterMap<K, V> oldCnt) {
    double diff = 0;   
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
        diff += Math.abs(newCnt.getCount(key, value) - oldCnt.getCount(key, value));
        cnt++;
      }
    }

    return diff / cnt;
  }
}
