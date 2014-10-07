package cs224n.wordaligner;  

import cs224n.util.*;
import java.util.List;

/**
 * PMI model for language alignment
 * 
 * @author Yuming Kuang
 */
public class PMIModel implements WordAligner {

  private static final long serialVersionUID = 1315751943476440515L;
  
  // Store p(f | e) = p(f, e) / p(f)
  private CounterMap<String, String> sourceTargetCounts;

  public Alignment align(SentencePair sentencePair) {
    // Initialization
    Alignment alignment = new Alignment();
    int numSourceWords = sentencePair.getSourceWords().size();
    int numTargetWords = sentencePair.getTargetWords().size();

    // Go through each target word and do alignment
    for (int tgtIndex = 0; tgtIndex < numTargetWords; tgtIndex++) {
      int alignIndex = -1;
      double maxPMI = -1.0;
      // Go through each source word, start with -1 indicating NULL
      for (int srcIndex = -1; srcIndex < numSourceWords; srcIndex++) {
        // Compute PMI
        String sourceWord = srcIndex == -1 ? "" : sentencePair.getSourceWords().get(srcIndex);
        String targetWord = sentencePair.getTargetWords().get(tgtIndex);
        double PMI = sourceTargetCounts.getCount(sourceWord, targetWord);
        // Do alignment
        if (PMI > maxPMI) {
           alignIndex = srcIndex;
           maxPMI = PMI;
        }
      }
      // Reocrd the alignment 
      if (alignIndex > -1) {
        alignment.addPredictedAlignment(tgtIndex, alignIndex);
      }
    }
    return alignment;
  }

  public void train(List<SentencePair> trainingPairs) {
    // Initialize
    sourceTargetCounts = new CounterMap<String, String>();
    
    // Go through each pair of sentences
    for(SentencePair pair : trainingPairs){
      List<String> targetWords = pair.getTargetWords();
      List<String> sourceWords = pair.getSourceWords();
      
      // Go through all the pairs of source and target words
      for(int srcIndex = -1; srcIndex < sourceWords.size(); srcIndex ++){
        for(int tgtIndex = 0; tgtIndex < targetWords.size(); tgtIndex ++){
          String source = (srcIndex == -1 ? "" : sourceWords.get(srcIndex));
          String target = targetWords.get(tgtIndex);
          // Count
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
