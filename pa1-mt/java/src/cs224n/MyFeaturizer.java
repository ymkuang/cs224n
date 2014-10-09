package edu.stanford.nlp.mt.decoder.feat;

import java.util.List;

import edu.stanford.nlp.mt.util.FeatureValue;
import edu.stanford.nlp.mt.util.Featurizable;
import edu.stanford.nlp.mt.util.IString;
import edu.stanford.nlp.mt.decoder.feat.RuleFeaturizer;
import edu.stanford.nlp.util.Generics;

/**
 * A rule featurizer.
 */
public class MyFeaturizer implements RuleFeaturizer<IString, String> {
  
  @Override
  public void initialize() {
    // Do any setup here.
  }

  @Override
  public List<FeatureValue<String>> ruleFeaturize(
      Featurizable<IString, String> f) {

    // TODO: Return a list of features for the rule. Replace these lines
    // with your own feature.
    
	List<FeatureValue<String>> features = Generics.newLinkedList();
    
//	features.add(new FeatureValue<String>("MyFeature", 1.0));
	
	features.add(new FeatureValue<String>(
		String.format("%s:%d","TGTD",(f.targetPhrase.size())),1.0));
  
	features.add(new FeatureValue<String>(
		String.format("%s:%d","SOCD",(f.sourcePhrase.size())),1.0));
	
	features.add(new FeatureValue<String>(
                String.format("%s:%d","RULD",(f.sourcePhrase.size()+f.targetPhrase.size())),1.0));  
	
	features.add(new FeatureValue<String>(
		String.format("%s","UTST"), f.numUntranslatedSourceTokens));

/*        if (f.targetPhrase.size()>0)
	features.add(new FeatureValue<String>(
                String.format("%s","TGTCap"), (Character.isUpperCase(f.targetPhrase.get(0).toString().charAt(0)) ? 1 : 0)));

        if (f.sourcePhrase.size()>0)
	features.add(new FeatureValue<String>(
                String.format("%s","SOCCap"), (Character.isUpperCase(f.sourcePhrase.get(0).toString().charAt(0)) ? 1 : 0)));
*/
	if (f.targetPhrase.size()>0 && f.sourcePhrase.size()>0)
	features.add(new FeatureValue<String>(
		String.format("%s","SSL"),(f.sourcePhrase.get(0).toString().charAt(0)==f.targetPhrase.get(0).toString().charAt(0)) ? 1 : 0 ));

	
	
	return features;
  }

  @Override
  public boolean isolationScoreOnly() {
    return false;
  }
}
