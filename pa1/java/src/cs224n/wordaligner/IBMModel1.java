package cs224n.wordaligner;  

import cs224n.util.*;
import java.util.List;
import java.lang.math;

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
 
    // Iteration
    while (delta > 1e-4) {
      oldLexicalProb = lexicalProb;
      lexicalProb = new CounterMap<String, String>();

      // E step
      for(SentencePair pair : trainingPairs){
        List<String> targetWords = pair.getTargetWords();
        List<String> sourceWords = pair.getSourceWords();
        sourceWords.add(0, "");
      
        for (String target : targetWords) {
          // Compute sum(t(e_i|f_j))
          double totalProb = 0;
          for (String source : sourceWords) {
            totalProb += oldLexicalProb.getCount(source, target);
          }
          // Update M step
          for (String source : sourceWords) {
            lexicalProb.incrementCount(source, target, oldLexicalProb.getCount(source, target) / total)
          }   
        }
      }
      // M step normalize
      lexicalProb = Counters.conditionNormalize(lexicalProb);
      // Compute difference
      delta = computeDiff(lexicalProb, oldLexicalProb);
    
    } 
  }

  private double computeDiff(CounterMap<K, K> newCnt, CounterMap<K, K> oldCnt) {
    double delta = 0;
    
    // All keys in two CounterMap
    Set<K> allKey = newCnt.keySet();
    allKey.add(oldCnt.keySet());

    for (K key : allKey) {
      // All values for a key in two CounterMap
      Set<K> allValue = newCnt.getCounter(key).keySet();
      allValue.add(oldCnt.getCounter(key).keySet());

      // Compute difference for each (key, value) pair
      for (K value : allValue) {
        delta += abs(newCnt.getCount(key, value) - oldCnt.getCount(key, value));
      }
    }

    return delta; 
  }
}
