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
		headCluster = new HashMap<String, int>();
		for(Entity e : clusters) {
            for(Pair<Mention, Mention> mentionPair : e.orderedMentionPairs()) {
                Set<String> setFirst = headCluster.get(mentionPair.getFirst());
                Set<String> setSecond = headCluster.get(mentionPair.getSecond());
                if (setFirst == null && setSecond == null) {
                	Set<String> newHeadSet = new Set<String>();
                	newHeadSet.add(mentionPair.getFirst());
                	newHeadSet.add(mentionPair.getSecond());
                	headCluster.put(mentionPair.getFirst(), newHeadSet);
                	headCluster.put(mentionPair.getSecond(), newHeadSet);
                } else if (setFirst == null) {
                	setSecond.add(mentionPair.getFirst());
                	headCluster.put(mentionPair.getFirst(), setSecond);
                } else if (setSecond == null) {
                	setFirst.add(mentionPair.getSecond());
                	headCluster.put(mentionPair.getSecond(), setFirst);
                } else {
                	if (setFirst == setSecond) continue;
                	setFirst.addAll(setSecond);
                	for (String s : setSecond) {
                		headCluster.put(s, setFirst);
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
            if (setHead == null) setHead = new Set<String>(head);
            Entity e = clusters.get(setHead);
            if (e != null) mentions.add(m.makeCoreferent(e));
            else {
            	ClusteredMention newCluster = m.markSingleton();
            	clusters.put(setHead, newCluster.entity);
            	mentions.add(newCluster);
            }
        }
        return mentions;
    }

}
