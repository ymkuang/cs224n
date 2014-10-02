package cs224n.wordaligner;  

import cs224n.util.*;
import java.util.List;

/**
 * Simple word alignment baseline model that maps source positions to target 
 * positions along the diagonal of the alignment grid.
 * 
 * IMPORTANT: Make sure that you read the comments in the
 * cs224n.wordaligner.WordAligner interface.
 * 
 * @author Yuming Kuang
 */
public class PMIModel implements WordAligner {

  private static final long serialVersionUID = 1315751943476440515L;
  
  // TODO: Use arrays or Counters for collecting sufficient statistics
  // from the training data.
  private CounterMap<String,String> sourceTargetCounts;

  public Alignment align(SentencePair sentencePair) {
    // Placeholder code below. 
    // TODO Implement an inference algorithm for Eq.1 in the assignment
    // handout to predict alignments based on the counts you collected with train().
    Alignment alignment = new Alignment();
    int numSourceWords = sentencePair.getSourceWords().size();
    int numTargetWords = sentencePair.getTargetWords().size();

    
    for (int tgtIndex = 0; tgtIndex < numTargetWords; tgtIndex++) {
      int alignIndex = -1;
      double maxPMI = -1.0;
      for (int srcIndex = -1; srcIndex < numSourceWords; srcIndex++) {
        String sourceWord = srcIndex == -1 ? "" : sentencePair.getSourceWords().get(srcIndex);
        String targetWord = sentencePair.getTargetWords().get(tgtIndex);
        double PMI = sourceTargetCounts.getCount(sourceWord, targetWord);
        if (PMI > maxPMI) {
           alignIndex = srcIndex;
           maxPMI = PMI;
        }
      } 
      if (alignIndex > -1) {
        alignment.addPredictedAlignment(tgtIndex, alignIndex);
      }
    }
    return alignment;
  }

  public void train(List<SentencePair> trainingPairs) {
    sourceTargetCounts = new CounterMap<String, String>();
    
    for(SentencePair pair : trainingPairs){
      List<String> targetWords = pair.getTargetWords();
      List<String> sourceWords = pair.getSourceWords();
      sourceWords.add("");
      
      for(String source : sourceWords){
        for(String target : targetWords){
          // TODO: Warm-up. Your code here for collecting sufficient statistics.
          if (sourceTargetCounts.getCount(source, target) == 0.0) {
            sourceTargetCounts.setCount(source, target, 1.0);
          } else {
            sourceTargetCounts.incrementCount(source, target, 1.0);
          }
        }
      }
    }
    sourceTargetCounts = Counters.conditionalNormalize(sourceTargetCounts);
   
  }
 
}
