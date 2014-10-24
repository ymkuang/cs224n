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
        // Do markovization and binarization
        List<Tree<String>> binaryTrainTrees = new ArrayList<Tree<String>>();
        for (Tree<String> trainTree : trainTrees)
            binaryTrainTrees.add(TreeAnnotations.annotateTree(trainTree, 1));

        lexicon = new Lexicon(binaryTrainTrees);
        grammar = new Grammar(binaryTrainTrees);

        baselineParser = new BaselineParser();
        baselineParser.train(trainTrees);
    }

    public void train(List<Tree<String>> trainTrees, int mode) {
        // Do markovization and binarization
	    List<Tree<String>> binaryTrainTrees = new ArrayList<Tree<String>>();
        for (Tree<String> trainTree : trainTrees)
            binaryTrainTrees.add(TreeAnnotations.annotateTree(trainTree, mode));

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
            addUnaryRule(score.get(i).get(i + 1), back.get(i).get(i + 1));
        }

        // Go on with grammars
        for (int span = 2; span <= numWord; span ++ ) {
            for (int begin = 0; begin <= numWord - span; begin ++) {
                int end = begin + span;
                for (int split = begin + 1; split <= end - 1; split ++) {
                    Set<String> tagRightChildren = score.get(split).get(end).keySet();
                    for (String tagLeftChild : score.get(begin).get(split).keySet()) {
                        double leftChildScore = score.get(begin).get(split).get(tagLeftChild);
                        for (Grammar.BinaryRule binaryRule : grammar.getBinaryRulesByLeftChild(tagLeftChild)) {
                            if (!tagRightChildren.contains(binaryRule.getRightChild())) continue;
                            String tagRightChild = binaryRule.getRightChild();
                            double prob = leftChildScore * score.get(split).get(end).get(tagRightChild) *
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
                }
                addUnaryRule(score.get(begin).get(end), back.get(begin).get(end));
            }
        }

        if (!score.get(0).get(numWord).containsKey("ROOT")) {
            // Can't parse, use BaselineParser 
            return baselineParser.getBestParse(sentence);
        } else 
            return recoverParseTree(sentence, score, back);
    }

    private void addUnaryRule(Map<String, Double> score, Map<String, Triplet<Integer, String, String> > back) {
        // Use queue to add unary rules for a cell
        LinkedList< Triplet<String, String, Double> > addTag = new LinkedList< Triplet<String, String, Double> >();
        for (String tagChild : score.keySet()) {
            double childScore = score.get(tagChild);
            for (Grammar.UnaryRule unaryRule : grammar.getUnaryRulesByChild(tagChild)) {
                double prob = unaryRule.getScore() * childScore;
                String tagParent = unaryRule.getParent();
                addTag.offer(new Triplet(tagParent, tagChild, prob));
            }
        }
        while (addTag.size() != 0) {
            Triplet<String, String, Double> newTag = addTag.poll();
            String tagParent = newTag.getFirst();
            String tagChild = newTag.getSecond();
            double prob = newTag.getThird();
            Double oldProb = score.get(tagParent);
            if (oldProb == null || prob > oldProb) {
                score.put(tagParent, prob);
                Triplet<Integer, String, String> newBack 
                    = new Triplet<Integer, String, String>(-1, tagChild, "");
                back.put(tagParent, newBack);

                tagChild = tagParent;
                double childScore = score.get(tagChild);
                for (Grammar.UnaryRule unaryRule : grammar.getUnaryRulesByChild(tagChild)) {
                    prob = unaryRule.getScore() * childScore;
                    tagParent = unaryRule.getParent();
                    addTag.offer(new Triplet(tagParent, tagChild, prob));
                }
            }
        }
    }

    private Tree<String> recoverParseTree(
        List<String> sentence, ArrayList< ArrayList< Map<String, Double> > > score,
        ArrayList< ArrayList< Map<String, Triplet<Integer, String, String> > > > back) {
        // Use queue (breadth fist search) to recover the parse tree
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
