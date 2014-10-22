package cs224n.assignment;

import cs224n.ling.Tree;
import cs224n.util.Pair;
import cs224n.util.Triplet;
import java.util.*;

/**
 * The CKY PCFG Parser you will implement.
 */
public class PCFGParser implements Parser {
    private Grammar grammar;
    private Lexicon lexicon;
    private BaselineParser baselineParser;

    public void train(List<Tree<String>> trainTrees) {
        // TODO: before you generate your grammar, the training trees
        // need to be binarized so that rules are at most binary
        List<Tree<String>> binaryTrainTrees = new ArrayList<Tree<String>>();
        for (Tree<String> trainTree : trainTrees)
            binaryTrainTrees.add(TreeAnnotations.annotateTree(trainTree,0));

        lexicon = new Lexicon(binaryTrainTrees);
        grammar = new Grammar(binaryTrainTrees);

        baselineParser = new BaselineParser();
        baselineParser.train(trainTrees);
    }

    public void train(List<Tree<String>> trainTrees,int mode) {
        // TODO: before you generate your grammar, the training trees
        // need to be binarized so that rules are at most binary
        //mode:
//	System.out.print("Specified Annotation Mode : "+mode+"\n");

	List<Tree<String>> binaryTrainTrees = new ArrayList<Tree<String>>();
        for (Tree<String> trainTree : trainTrees)
            binaryTrainTrees.add(TreeAnnotations.annotateTree(trainTree,mode));

        lexicon = new Lexicon(binaryTrainTrees);
        grammar = new Grammar(binaryTrainTrees);

        baselineParser = new BaselineParser();
        baselineParser.train(trainTrees);
    }

    public Tree<String> getBestParse(List<String> sentence) {
        int numWord = sentence.size();
        // Initialize score and back
        ArrayList< ArrayList< Map<String, Double> > > score 
            = new ArrayList< ArrayList< Map<String, Double> > >();
        ArrayList< ArrayList< Map<String, Triplet<Integer, String, String> > > > back 
            = new ArrayList< ArrayList< Map<String, Triplet<Integer, String, String> > > >();
        for (int i = 0; i <= numWord; i ++) {
            ArrayList< Map<String, Double> > partScore 
                = new ArrayList< Map<String, Double> >();
            ArrayList< Map<String, Triplet<Integer, String, String> > > partBack
                = new ArrayList< Map<String, Triplet<Integer, String, String> > >();
            for (int j = 0; j <= numWord; j ++) {
              partScore.add(new HashMap<String, Double>());
              partBack.add(new HashMap<String, Triplet<Integer, String, String> >());
            }
            score.add(partScore);
            back.add(partBack);
        }

        // Start with words
        for (int i = 0; i < numWord; i ++) {
            String word = sentence.get(i);
            for (String tag : lexicon.getAllTags()) {
                double newScore = lexicon.scoreTagging(word, tag);
                score.get(i).get(i + 1).put(tag, newScore);
            } 
            boolean added = true;
            while (added) {
              added = false;
              for (String tagChild : score.get(i).get(i + 1).keySet()) {
                for (Grammar.UnaryRule unaryRule : grammar.getUnaryRulesByChild(tagChild)) {
                    double prob = unaryRule.getScore() * score.get(i).get(i + 1).get(tagChild);
                    String tagParent = unaryRule.getParent();
                    Double oldProb = score.get(i).get(i + 1).get(tagParent);
                    if (oldProb == null || prob > oldProb) {
                        score.get(i).get(i + 1).put(tagParent, prob);
                        Triplet<Integer, String, String> newBack 
                            = new Triplet<Integer, String, String>(-1, tagChild, "");
                        back.get(i).get(i + 1).put(tagParent, newBack);
                        added = true;
                    }
                }
                if (added) break;
              }
            }
        }

        // Go on with grammars
        for (int span = 2; span <= numWord; span ++ ) {
            for (int begin = 0; begin <= numWord - span; begin ++) {
                int end = begin + span;
                for (int split = begin + 1; split <= end - 1; split ++) 
                    for (String tagLeftChild : score.get(begin).get(split).keySet()) {
                        Set<String> tagRightChildren = score.get(split).get(end).keySet();
                        for (Grammar.BinaryRule binaryRule : grammar.getBinaryRulesByLeftChild(tagLeftChild)) {
                            if (!tagRightChildren.contains(binaryRule.getRightChild())) continue;
                            String tagRightChild = binaryRule.getRightChild();
                            double prob = score.get(begin).get(split).get(tagLeftChild) *
                                    score.get(split).get(end).get(tagRightChild) *
                                    binaryRule.getScore();
                            String tagParent = binaryRule.getParent();
                            Double oldProb = score.get(begin).get(end).get(tagParent);             
                            if (oldProb == null || prob > oldProb) {
                                score.get(begin).get(end).put(tagParent, prob);
                                Triplet<Integer, String, String> newBack 
                                    = new Triplet<Integer, String, String>(split, tagLeftChild, tagRightChild);
                                back.get(begin).get(end).put(tagParent, newBack);
                            }
                        }
                    }

                boolean added = true;
                while (added) {
                    added = false;
                    for (String tagChild : score.get(begin).get(end).keySet()) {
                        for (Grammar.UnaryRule unaryRule : grammar.getUnaryRulesByChild(tagChild)) {
                            double prob = unaryRule.getScore() * score.get(begin).get(end).get(tagChild);
                            String tagParent = unaryRule.getParent();
                            Double oldProb = score.get(begin).get(end).get(tagParent);             
                            if (oldProb == null || prob > oldProb) {
                               score.get(begin).get(end).put(tagParent, prob);
                               Triplet<Integer, String, String> newBack 
                                   = new Triplet<Integer, String, String>(-1, tagChild, "");
                               back.get(begin).get(end).put(tagParent, newBack);
                               added = true;
                            }
                        }
                        if (added) break;
                    }
                }    
            }
        }

        if (!score.get(0).get(numWord).containsKey("ROOT")) {
            // Can't parse, use BaselineParser 
            return baselineParser.getBestParse(sentence);
        } else 
            return recoverParseTree(sentence, score, back);
    }

    private Tree<String> recoverParseTree(
        List<String> sentence, ArrayList< ArrayList< Map<String, Double> > > score,
        ArrayList< ArrayList< Map<String, Triplet<Integer, String, String> > > > back) {
        Queue< Triplet<Tree<String>, Integer, Integer> > queue 
            = new LinkedList< Triplet<Tree<String>, Integer, Integer> >();
        

        int end = score.size() - 1;
        
        Tree<String> root = new Tree<String>("ROOT");
        queue.offer(new Triplet<Tree<String>, Integer, Integer> (root, 0, end));
        while(queue.size() != 0) {
            List< Tree<String> > children = new ArrayList< Tree<String> >();
            Triplet<Tree<String>, Integer, Integer> parent = queue.poll();
            String paTag = parent.getFirst().getLabel();
            int paBegin = parent.getSecond();
            int paEnd = parent.getThird();
            Triplet<Integer, String, String> next = back.get(paBegin).get(paEnd).get(paTag);
            if (next == null && paBegin == paEnd - 1) {
                Tree<String> node = new Tree<String>(sentence.get(paBegin));
                children.add(node);
                parent.getFirst().setChildren(children);
                continue;
            } 
            if (next == null) continue;

            int split = next.getFirst();
            if (split == -1) {
                String chTag = next.getSecond();
                Tree<String> node = new Tree<String>(chTag);
                children.add(node);
                parent.getFirst().setChildren(children);
                queue.offer(new Triplet<Tree<String>, Integer, Integer>(node, paBegin, paEnd));
            } else {
                String leftChTag = next.getSecond();
                String rightChTag = next.getThird();
                Tree<String> leftNode = new Tree<String>(leftChTag);
                Tree<String> rightNode = new Tree<String>(rightChTag);
                children.add(leftNode);
                children.add(rightNode);
                parent.getFirst().setChildren(children);

                queue.offer(new Triplet<Tree<String>, Integer, Integer>(leftNode, paBegin, split));
                queue.offer(new Triplet<Tree<String>, Integer, Integer>(rightNode, split, paEnd));                
            }
        }
        return TreeAnnotations.unAnnotateTree(root);
    }

}
