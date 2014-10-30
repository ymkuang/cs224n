package cs224n.corefsystems;

import java.util.Collection;
import java.util.List;
import java.util.ArrayList;

import cs224n.coref.ClusteredMention;
import cs224n.coref.Document;
import cs224n.coref.Entity;
import cs224n.coref.Mention;
import cs224n.util.Pair;

public class OneCluster implements CoreferenceSystem {

	@Override
	public void train(Collection<Pair<Document, List<Entity>>> trainingData) {
		// TODO Auto-generated method stub

	}

	@Override
	public List<ClusteredMention> runCoreference(Document doc) {
		List<ClusteredMention> mentions = new ArrayList<ClusteredMention>();

        for (Mention m : doc.getMentions()) {
        	if (mentions.isEmpty()) {
        		ClusteredMention newCluster = m.markSingleton();
                mentions.add(newCluster);
        	} else {
        		mentions.add(m.markCoreferent(mentions.get(0)));
        	}
        }

		return mentions;
	}

}
