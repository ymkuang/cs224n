package cs224n.corefsystems;

import java.util.Collection;
import java.util.List;
import java.util.ArrayList;
import java.util.*;


import cs224n.coref.ClusteredMention;
import cs224n.coref.Document;
import cs224n.coref.Mention;
import cs224n.coref.Entity;
import cs224n.coref.*;
import cs224n.util.Pair;

public class BetterBaseline implements CoreferenceSystem {
    private Map<String, Set<String>> headCluster;

	@Override
	public void train(Collection<Pair<Document, List<Entity>>> trainingData) {
		headCluster = new HashMap<String, Set<String>>();
        for (Pair<Document, List<Entity>> pair : trainingData) {
            List<Entity> clusters = pair.getSecond();
            for (Entity e : clusters) {
                for (Pair<Mention, Mention> mentionPair : e.orderedMentionPairs()) {
                    String firstHead = mentionPair.getFirst().headWord();
                    String secondHead = mentionPair.getSecond().headWord();
                    Set<String> setFirst = headCluster.get(firstHead);
                    Set<String> setSecond = headCluster.get(secondHead);
                    Set<String> newHeadSet;
                    if (setFirst == null && setSecond == null) {
                       newHeadSet = new HashSet<String>();
                       newHeadSet.add(firstHead);
                       newHeadSet.add(secondHead);
                    } else if (setFirst == null) {
                        newHeadSet = setSecond;
                        newHeadSet.add(firstHead);
                    } else if (setSecond == null) {
                        newHeadSet = setFirst;
                        newHeadSet.add(secondHead);
                    } else {
                        if (setFirst.equals(setSecond)) continue;
                        newHeadSet = setFirst;
                        newHeadSet.addAll(setSecond);
                    }
                    for (String s : newHeadSet) {
                        headCluster.put(s, newHeadSet);
                    }    
                }
            }
        }
	}

	@Override
	public List<ClusteredMention> runCoreference(Document doc) {
		List<ClusteredMention> mentions = new ArrayList<ClusteredMention>();
		Map<Set<String>, Entity> clusters = new HashMap<Set<String>, Entity>();
		for (Mention m : doc.getMentions()) {
			String head = m.headWord();
            Set<String> setHead = headCluster.get(head);
            if (setHead == null) {
                setHead = new HashSet<String>();
                setHead.add(head);
            }
            Entity e = clusters.get(setHead);
            if (e != null) mentions.add(m.markCoreferent(e));
            else {
            	ClusteredMention newCluster = m.markSingleton();
            	clusters.put(setHead, newCluster.entity);
            	mentions.add(newCluster);
            }
        }
        return mentions;
    }

}
