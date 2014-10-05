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
public class IBMModel1 implements WordAligner {

  private static final long serialVersionUID = 1315751943476440515L;
  
  // TODO: Use arrays or Counters for collecting sufficient statistics
  // from the training data.
  private CounterMap<String,String> lexicalProb;
  private double tol = 1e-2;
  private int maxNumberIter = 30;

  public CounterMap<String, String> getLexicalProb() {
    return lexicalProb;
  }

  public Alignment align(SentencePair sentencePair) {
    // Placeholder code below. 
    // TODO Implement an inference algorithm for Eq.1 in the assignment
    // handout to predict alignments based on the counts you collected with train().
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
    double delta = 1;
    int iter = 0;
    // Iteration
    while (delta > tol && iter < maxNumberIter) {
      oldLexicalProb = lexicalProb;
      lexicalProb = new CounterMap<String, String>();
      iter ++;

      // E step
      for(SentencePair pair : trainingPairs){
        List<String> targetWords = pair.getTargetWords();
        List<String> sourceWords = pair.getSourceWords();
      
        for (String target : targetWords) {
          // Compute sum(t(e_i|f_j))
          double totalProb = 0;
          for (int srcIndex = -1; srcIndex < sourceWords.size(); srcIndex ++) {
            String source = (srcIndex == -1 ? "" : sourceWords.get(srcIndex));
            totalProb += (iter == 1 ? 1 : oldLexicalProb.getCount(source, target));
          }
          // Update M step
          for (int srcIndex = -1; srcIndex < sourceWords.size(); srcIndex ++) {
            String source = (srcIndex == -1 ? "" : sourceWords.get(srcIndex));
            double increment = (iter == 1 ? 1 : oldLexicalProb.getCount(source, target)) / totalProb; 
            if (lexicalProb.getCount(source, target) == 0.0) {
              lexicalProb.setCount(source, target, increment);
            } else {
              lexicalProb.incrementCount(source, target, increment);
            }
          }   
        }
      }
      // M step normalize
      lexicalProb = Counters.conditionalNormalize(lexicalProb);
      
      // Compute difference
      delta = iter == 1 ? 1 : computeDiff(lexicalProb, oldLexicalProb);
      System.out.printf("%f\n", delta);
    } 
  }

  private <K> double computeDiff(CounterMap<K, K> newCnt, CounterMap<K, K> oldCnt) {
    double delta = 0;
    int cnt = 0;    

    // All keys in two CounterMap
    Set<K> allKey = new HashSet<K>(newCnt.keySet());
    allKey.addAll(oldCnt.keySet());
  
    for (K key : allKey) {
      // All values for a key in two CounterMap
      Set<K> allValue = new HashSet<K>(newCnt.getCounter(key).keySet());
      allValue.addAll(oldCnt.getCounter(key).keySet());

      // Compute difference for each (key, value) pair
      for (K value : allValue) {
        delta += Math.abs(newCnt.getCount(key, value) - oldCnt.getCount(key, value));
        cnt ++;
      }
    }

    return delta / cnt; 
  }
}
