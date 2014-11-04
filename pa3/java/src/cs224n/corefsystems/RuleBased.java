package cs224n.corefsystems;

import java.util.Collection;
import java.util.List;
import java.util.*;

import cs224n.coref.Pronoun;
import cs224n.coref.ClusteredMention;
import cs224n.coref.Document;
import cs224n.coref.Mention;
import cs224n.coref.Entity;
import cs224n.coref.*;
import cs224n.util.Pair;
import cs224n.ling.Tree;

public class RuleBased implements CoreferenceSystem {
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
		Map<Mention, Set<Mention>> clusterMap = new HashMap<Mention, Set<Mention>>();
		List<Mention> pronouns = new ArrayList<Mention>();
		Map<Tree<String>, Tree<String>> parentMap = constructParentMap(doc);
		Map<Tree<String>, Mention> parseMap = constructParseMap(doc);

		// First Pass: Exact match
		Map<String, Set<Mention>> exactMap = new HashMap<String, Set<Mention>>();
		for (Mention m : doc.getMentions()) {
			String word = m.gloss();
			if (Pronoun.isSomePronoun(word)) {
				pronouns.add(m);
			} else if (exactMap.containsKey(word)) {
				exactMap.get(word).add(m);
				clusterMap.put(m, exactMap.get(word));
			} else {
				Set<Mention> cluster = new HashSet<Mention>();
				cluster.add(m);
				clusterMap.put(m, cluster);
			}
		}

		// Second Pass: Appositive
		for (Mention m1 : clusterMap.keySet()) {
			for (Mention m2 : clusterMap.keySet()) {
				if (m1.equals(m2)) continue;
                if (clusterMap.get(m1).equals(clusterMap.get(m2))) continue;
				if (isAppositive(m1, m2, parentMap) && isAgree(m1, m2)) {
					System.out.printf("Appositive: %s  AND  %s\n", m1.gloss(), m2.gloss());
					mergeSet(m1, m2, clusterMap);
				}
			}
		}

		// Third Pass: Head Match
		for (Mention m1 : clusterMap.keySet()) {
			for (Mention m2: clusterMap.keySet()) {
				if (m1.equals(m2)) continue;
                if (clusterMap.get(m1).equals(clusterMap.get(m2))) continue;
				if (isHeadMatch(m1, m2) && isAgree(m1, m2)) {
					System.out.printf("Head Match: %s  AND  %s\n", m1.gloss(), m2.gloss());
					mergeSet(m1, m2, clusterMap);
				}
			}
		}

		// Fourth Pass: Pronouns
		for (Mention m : pronouns) {
			Mention mention = null;
			if (m.headToken().isQuoted()) mention = quotedPronoun(m, doc);
            if (mention == null) mention = hobbs(doc, m, parentMap, parseMap);
            if (mention != null )
            	System.out.printf("Pronoun Match: %s  AND  %s\n", m.gloss(), mention.gloss());
            
            Set<Mention> mCluster = clusterMap.get(m);
            Set<Mention> mentionCluster = (mention == null ? null : clusterMap.get(mention));
            Set<Mention> newCluster;
            if (mCluster == null && mentionCluster == null) {
            	newCluster = new HashSet<Mention>();
            	newCluster.add(m);
            	if (mention != null) newCluster.add(mention);
            } else if (mCluster == null) {
            	newCluster = mentionCluster;
            	newCluster.add(m);
            } else if (mentionCluster == null) {
            	if (mention == null) continue;
            	newCluster = mCluster;
            	newCluster.add(mention);
            } else {
            	if (mCluster.equals(mentionCluster)) continue;
            	newCluster = mCluster;
            	newCluster.addAll(mentionCluster);
            }
            for (Mention n : newCluster) clusterMap.put(n, newCluster);
		}
        
        // Result
	    List<ClusteredMention> mentions = new ArrayList<ClusteredMention>();
        for (Set<Mention> cluster : getClusters(clusterMap)) {
        	ClusteredMention newCoref = null;
        	for (Mention m : cluster) {
        		if (newCoref != null) mentions.add(m.markCoreferent(newCoref));
        		else {
        			newCoref = m.markSingleton();
        			mentions.add(newCoref);
        		}
        	}
        }

        return mentions;
	}
	
	private boolean isAppositive(Mention m1, Mention m2, Map<Tree<String>, Tree<String>> parentMap) {
		if (m1 == null || m2 == null) return false;
		Sentence s1 = m1.sentence, s2 = m2.sentence;
		if (!s1.equals(s2)) return false;
		if (!m1.parse.getLabel().equals("NP") || !m2.parse.getLabel().equals("NP")) return false;
		Tree<String> parent1 = parentMap.get(m1.parse), parent2 = parentMap.get(m2.parse);
		if (parent1 == null || parent2 == null) return false;
		if (parent1.equals(parent2)) return false;
		Tree<String> parent = parent1;
		List<Tree<String>> children = parent.getChildren();
		if (parent.getLabel().equals("NP") && children.size() == 3 
			&& children.get(0).getLabel().equals(m1.parse) 
			&& children.get(2).getLabel().equals(m2.parse)
			&& children.get(1).getLabel().equals(","))
			return true;
	    else return false;
	}

    private boolean isAgree(Mention m1, Mention m2) {
    	if (m1 == null || m2 == null) return false;
    	Pair<Boolean, Boolean> genderAgreeInfo = Util.haveGenderAndAreSameGender(m1, m2);
        boolean genderAgree = (genderAgreeInfo.getFirst() && genderAgreeInfo.getSecond()) || !genderAgreeInfo.getFirst();
        Pair<Boolean, Boolean> numberAgreeInfo = Util.haveNumberAndAreSameNumber(m1, m2);
        boolean numberAgree = (numberAgreeInfo.getFirst() && numberAgreeInfo.getSecond()) || !numberAgreeInfo.getFirst();
        boolean nerAgree = isNerAgree(m1, m2);
        boolean speakerAgree = isSpeakerAgree(m1, m2);
        return genderAgree && numberAgree && nerAgree && speakerAgree;
    }

    private boolean isSpeakerAgree(Mention m1, Mention m2) {
    	Pronoun pronoun1 = Pronoun.valueOrNull(m1.gloss());
    	Pronoun pronoun2 = Pronoun.valueOrNull(m2.gloss());
    	if (pronoun1 == null || pronoun2 == null) return true;
    	return pronoun1.speaker == pronoun2.speaker;
    }

    private boolean isNerAgree(Mention m1, Mention m2) {
    	if (m1.headToken().equals("O") || m1.headToken().equals("0")) return true;
    	if (m2.headToken().equals("O") || m2.headToken().equals("0")) return true;
    	return m1.headToken().nerTag().equals(m2.headToken().nerTag());
    }

    private boolean isHeadMatch(Mention m1, Mention m2) {
    	if (m1 == null || m2 == null) return false;
    	String head1 = m1.headWord();
    	String head2 = m2.headWord();
    	if (head1.equals(head2)) return true;
    	Set<String> headSet1 = headCluster.get(head1);
    	Set<String> headSet2 = headCluster.get(head2);
    	if (headSet1 == null || headSet2 == null) return false;
    	if (headSet1.contains(head2) || headSet2.contains(head1)) return true;
    	return false;
    }

    private Mention quotedPronoun(Mention m, Document doc) {
    	Pronoun pronoun = Pronoun.valueOrNull(m.gloss());
    	if (pronoun == null) return null;
    	if (pronoun.speaker != Pronoun.Speaker.FIRST_PERSON) return null;
    	String speaker = m.headToken().speaker();
    	for (Mention mention : doc.getMentions()) {
    		if (speaker.equals(mention.gloss()) && isAgree(m, mention)) return mention;
    	}
    	return null;
    }

    private void mergeSet(Mention m1, Mention m2, Map<Mention, Set<Mention>> clusterMap) {
        if (m1 == null || m2 == null) return;
        Set<Mention> newCluster = clusterMap.get(m1);
        newCluster.addAll(clusterMap.get(m2));
        for (Mention m : newCluster)
        	clusterMap.put(m, newCluster);
    }

    private Set<Set<Mention>> getClusters(Map<Mention, Set<Mention>> clusterMap) {
    	Set<Set<Mention>> clusterSet = new HashSet<Set<Mention>>();
    	for (Set<Mention> s : clusterMap.values()) {
    		clusterSet.add(s);
    	}
    	return clusterSet;
    }

    private Map<Tree<String>, Tree<String>> constructParentMap(Document doc) {
        Map<Tree<String>, Tree<String>> parentMap = new HashMap<Tree<String>, Tree<String>>();
        for (Sentence s : doc.sentences) {
        	constructParentMapHelper(s.parse, parentMap);
        }
        return parentMap;
    }

    private void constructParentMapHelper(Tree<String> node, Map<Tree<String>, Tree<String>> parentMap) {
        if (node == null) return;
        for (Tree<String> child : node.getChildren()) {
        	constructParentMapHelper(child, parentMap);
        	parentMap.put(child, node);
        }
    }

    private Map<Tree<String>, Mention> constructParseMap(Document doc) {
    	Map<Tree<String>, Mention> parseMap = new HashMap<Tree<String>, Mention>();
    	for (Mention m : doc.getMentions()) {
    		parseMap.put(m.parse, m);
    	}
    	return parseMap;
    }

    private Tree<String> traceUp(Tree<String> cur, int labelType, 
    	Map<Tree<String>, Tree<String>> parentMap, Set<Tree<String>> path) {
    	if (cur == null) return null;
    	if (path != null) path.clear();
    	while (true) {
    		if (path != null) path.add(cur);

            System.out.printf("%s ->", cur.getLabel());
    		cur = parentMap.get(cur);
    		if (cur == null) break;
    		System.out.printf("%s\n", cur.getLabel());

    		if (cur.getLabel().equals("NP") || (labelType == 1 && cur.getLabel().equals("S")))
    			return cur;
    	}
    	return null;
    }

    private void traverse(Tree<String> start, int traverseType, 
    	Set<Tree<String>> path, LinkedList<Tree<String>> candidates) {
    	if (start == null) return;
    	LinkedList<Tree<String>> queue = new LinkedList<Tree<String>>();
    	
    	boolean beforePath = true;
    	for (Tree<String> node : start.getChildren()) {
            if (traverseType != 0 && path.contains(node)) {
            	beforePath = false;
            	continue;
            }
            if (traverseType == 0 || (traverseType == 1 && beforePath) 
            	|| (traverseType == 2 && !beforePath)) {
            	queue.add(node);
            }
    	} 

    	while (!(queue.size() == 0)) {
    		Tree<String> node = queue.remove();
            if (node.getLabel().equals("NP")) candidates.add(node);
            if (traverseType != 2 || !node.getLabel().equals("NP")) {
            	for (Tree<String> child : node.getChildren()) {
            		queue.add(child);
            	}
            }
    	}

    }

    private Mention findCandidate(LinkedList<Tree<String>> candidates, Mention m, 
    	Map<Tree<String>, Mention> parseMap) {
        
        while (!(candidates.size() == 0)) {
        	Tree<String> node = candidates.remove();
            Mention candidate = parseMap.get(node);   
            if (candidate != null) System.out.printf("%s \n", candidate.gloss()); 
            if (candidate != null && isAgree(candidate, m)) return candidate;
        }
        return null;
    }

    private boolean isNonHeadPhrase(Tree<String> cur, Set<Tree<String>> path) {
        if (path == null) return false;
        for (Tree<String> node : path) {
        	if (!node.getLabel().equals("NP") && !node.getLabel().equals("NNS") 
        		 && !node.getLabel().equals("NN") && !node.getLabel().equals("NNP") 
        		 && !node.getLabel().equals("NNPS"))
        		return true;
        }
        return false;
    }

    private Mention hobbs(Document doc, Mention m, 
    	Map<Tree<String>, Tree<String>> parentMap, Map<Tree<String>, Mention> parseMap) {
    	System.out.printf("Pronoun: %s\n", m.gloss());
    	System.out.printf("%s\n", m.sentence.parse.toString());
    	Tree<String> cur = m.parse;
    	if (cur != null) System.out.printf("%s\n", cur.getLabel());
    	Sentence s = m.sentence; 
        Set<Tree<String>> path = new HashSet<Tree<String>>();
        LinkedList<Tree<String>> candidates = new LinkedList<Tree<String>>();
        Mention candidate;

        // Step 1
        System.out.printf("Step 1\n");
        if (!cur.getLabel().equals("NP")) cur = traceUp(cur, 0, parentMap, null);
        if (cur != null) System.out.printf("%s\n", cur.getLabel());
        if (cur == null) return null;
        
        // Step 2
        System.out.printf("Step 2\n");
        cur = traceUp(cur, 1, parentMap, path);
        if (cur != null) System.out.printf("%s\n", cur.getLabel());
        if (cur == null) return null;

        // Step 3
        System.out.printf("Step 3\n");
        traverse(cur, 1, path, candidates);
        candidate = findCandidate(candidates, m, parseMap);
        if (candidate != null) return candidate;

        while (true) {
            // Step 4
            if (cur.equals(s.parse)) {
            	System.out.printf("Step 4\n");
            	int sentenceIndex = doc.indexOfSentence(s);
            	if (sentenceIndex == 0) break;
            	s = doc.sentences.get(sentenceIndex - 1);
                cur = s.parse;
                traverse(cur, 0, null, candidates);
                candidate = findCandidate(candidates, m, parseMap);
                if (candidate != null) return candidate;
            } else {
            	// Step 5
            	System.out.printf("Step 5\n");
            	cur = traceUp(cur, 1, parentMap, path);
            	if (cur == null) {
            		cur = s.parse;
            		continue;
            	}
            	
            	// Step 6
            	System.out.printf("Step 6\n");
            	if (isNonHeadPhrase(cur, path)) candidates.add(cur);
            	candidate = findCandidate(candidates, m, parseMap);
                if (candidate != null) return candidate;

                // Step 7
                System.out.printf("Step 7\n");
                traverse(cur, 1, path, candidates);
                candidate = findCandidate(candidates, m, parseMap);
                if (candidate != null) return candidate;

                // Step 8
                System.out.printf("Step 8\n");
                if (cur.getLabel().equals("S")) traverse(cur, 2, path, candidates);
                candidate = findCandidate(candidates, m, parseMap);
                if (candidate != null) return candidate;
            }
        }
        return null;
    }
    
}
