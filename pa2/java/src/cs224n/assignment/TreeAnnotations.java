package cs224n.assignment;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import cs224n.ling.Tree;
import cs224n.ling.Trees;
import cs224n.ling.Trees.MarkovizationAnnotationStripper;
import cs224n.util.Filter;

/**
 * Class which contains code for annotating and binarizing trees for
 * the parser's use, and debinarizing and unannotating them for
 * scoring.
 */



public class TreeAnnotations {

	public static int mode_run = 0; 
	//0 just log lossless, 1 markov 2nd order, 2 markov 3rd order, 3 markov 3rd order + horizontal 2nd order
	
	public static int setMode(int mode) {
		mode_run = mode;
		return mode;
	}
		
    public static Tree<String> annotateTree(Tree<String> unAnnotatedTree) {
    	//0 just log lossless, 1 markov 2nd order, 2 markov 3rd order, 3 markov 3rd order + horizontal 2nd order
    	if (mode_run == 0) {
        	return binarizeTree(unAnnotatedTree);
    	}
    	if (mode_run == 1) {
        	return binarizeTree(myVert2MarkovTree(unAnnotatedTree, ""));
    	}
    	if (mode_run==2) {
            return binarizeTree(myVert3MarkovTree(unAnnotatedTree, "", ""));
        }
		if (mode_run==3) {
			return binarizeTree(myHor2Vert3MarkovTree(unAnnotatedTree, "", "", ""));
		}
        return binarizeTree(unAnnotatedTree);
    }
	
	public static Tree<String> annotateTree(Tree<String> unAnnotatedTree, int mode_input) {
		mode_run = mode_input;		
		//0 just log lossless, 1 markov 2nd order, 2 markov 3rd order, 3 markov 3rd order + horizontal 2nd order
		if (mode_input == 0) {
			return binarizeTree(unAnnotatedTree);
		}
		if (mode_input == 1) {
			return binarizeTree(myVert2MarkovTree(unAnnotatedTree, ""));
		}
		if (mode_input == 2) {
            return binarizeTree(myVert3MarkovTree(unAnnotatedTree, "", ""));
        }
		if (mode_input == 3) {
			return binarizeTree(myHor2Vert3MarkovTree(unAnnotatedTree, "", "", ""));
		}
		return binarizeTree(unAnnotatedTree);
	}


	private static int myPrintTree(Tree<String> tree) {
        System.out.print("-> "+tree.getLabel()+":");
        String label = tree.getLabel();

        if (tree.isLeaf()) {
            return 0;
        }
        else {
            List<Tree<String>> childrenTrees = tree.getChildren();
            for (Tree<String> childrenTree : childrenTrees)
            myPrintTree(childrenTree);
        }
        System.out.print("\n");
		return 1;
    }

	private static Tree<String> myVert2MarkovTree(Tree<String> tree, String labelParent) {
		// 2nd vertical Markovize
		String label = tree.getLabel();

		if (tree.isLeaf()) {
			return new Tree<String>(label);
		}
		
	    List<Tree<String>> newChildrenTrees = new ArrayList<Tree<String>>();
 		for (Tree<String> childrenTree : tree.getChildren())
            newChildrenTrees.add(myVert2MarkovTree(childrenTree, label));
		
		Tree<String> newNode;
		if (labelParent != ""){
			newNode = new Tree<String>(label + "^" + labelParent, newChildrenTrees);
		} else {
		    newNode = new Tree<String>(label, newChildrenTrees);
		}
		return newNode;
	}
	
	private static Tree<String> myVert3MarkovTree(Tree<String> tree, String labelParent, String labelGrandParent) {
        // 3rd vertical Markovize
        String label = tree.getLabel();
        if (tree.isLeaf()) {
            return new Tree<String>(label);
        }
        
        List<Tree<String>> newChildrenTrees = new ArrayList<Tree<String>>();
        for (Tree<String> childrenTree : tree.getChildren())
            newChildrenTrees.add(myVert3MarkovTree(childrenTree, label, labelParent));
        
        if (labelParent != ""){
            label += "^" + labelParent;
        }
		if (labelGrandParent != "") {
			label += "^" + labelGrandParent;
		}
        return new Tree<String> (label, newChildrenTrees);
    }	

	private static Tree<String> myHor2Vert3MarkovTree(Tree<String> tree, 
		String labelParent, String labelGrandParent, String labelBrother) {
        // 3rd vertical Markovize + 2nd horizental Markovize
        String label = tree.getLabel();
        if (tree.isLeaf()) {
        	return new Tree<String>(label);
		}

		List<Tree<String>> newChildrenTrees = new ArrayList<Tree<String>>();
		String labelChildBrother, labelChild = "";
		for (Tree<String> childrenTree : tree.getChildren()) {
            labelChildBrother = labelChild;
            labelChild = childrenTree.getLabel();
            newChildrenTrees.add(myHor2Vert3MarkovTree(childrenTree, label, labelParent, labelChildBrother));
		}
        if (labelParent != "") label += "^" + labelParent;
        if (labelGrandParent != "") label += "^" + labelGrandParent;
        if (labelBrother != "") label += "->" + labelBrother;

        return new Tree<String>(label, newChildrenTrees);

        }


	private static Tree<String> binarizeTree(Tree<String> tree) {
		String label = tree.getLabel();
		if (tree.isLeaf())
			return new Tree<String>(label);
		if (tree.getChildren().size() == 1) {
			return new Tree<String>
			(label, 
					Collections.singletonList(binarizeTree(tree.getChildren().get(0))));
		}
		// otherwise, it's a binary-or-more local tree, 
		// so decompose it into a sequence of binary and unary trees.
		String intermediateLabel = "@"+label+"->";
		Tree<String> intermediateTree;		 
		intermediateTree =	binarizeTreeHelper(tree, 0, intermediateLabel);
		return new Tree<String>(label, intermediateTree.getChildren());
	}

	private static Tree<String> binarizeHorizontalTreeHelper(Tree<String> tree,
			int numChildrenGenerated, 
			String intermediateLabel) {
		Tree<String> leftTree = tree.getChildren().get(numChildrenGenerated);
		List<Tree<String>> children = new ArrayList<Tree<String>>();
		children.add(binarizeTree(leftTree));
		if (numChildrenGenerated < tree.getChildren().size() - 1) {
			String horizontalLabel = intermediateLabel + "_" + leftTree.getLabel();
			int index1 = horizontalLabel.indexOf("->");
			int index2 = horizontalLabel.lastIndexOf("_");
			int index3 = horizontalLabel.lastIndexOf("_",index2-1);
			String horizontalLabel_sim = horizontalLabel;
			if (index1>0 && index2>0 && index3>0) {
				horizontalLabel_sim = horizontalLabel_sim.substring(0,index1+1)+"..."+horizontalLabel_sim.substring(index3+1);
			}
			Tree<String> rightTree = 
					binarizeTreeHelper(tree, numChildrenGenerated + 1, 
							horizontalLabel_sim);
			children.add(rightTree);
			intermediateLabel=horizontalLabel_sim;
		}
		return new Tree<String>(intermediateLabel, children);
	} 

	private static Tree<String> binarizeTreeHelper(Tree<String> tree,
                        int numChildrenGenerated,
                        String intermediateLabel) {
                Tree<String> leftTree = tree.getChildren().get(numChildrenGenerated);
                List<Tree<String>> children = new ArrayList<Tree<String>>();
                children.add(binarizeTree(leftTree));
                if (numChildrenGenerated < tree.getChildren().size() - 1) {
                        Tree<String> rightTree =
                                        binarizeTreeHelper(tree, numChildrenGenerated + 1,
                                                        intermediateLabel + "_" + leftTree.getLabel());
                        children.add(rightTree);
                }
                return new Tree<String>(intermediateLabel, children);
        }


	public static Tree<String> unAnnotateTree(Tree<String> annotatedTree) {

		// Remove intermediate nodes (labels beginning with "@"
		// Remove all material on node labels which follow their base symbol
		// (cuts at the leftmost - or ^ character)
		// Examples: a node with label @NP->DT_JJ will be spliced out, 
		// and a node with label NP^S will be reduced to NP

		Tree<String> debinarizedTree =
				Trees.spliceNodes(annotatedTree, new Filter<String>() {
					public boolean accept(String s) {
						return s.startsWith("@");
					}
				});
		Tree<String> unAnnotatedTree = 
				(new Trees.FunctionNodeStripper()).transformTree(debinarizedTree);
    Tree<String> unMarkovizedTree =
        (new Trees.MarkovizationAnnotationStripper()).transformTree(unAnnotatedTree);
		return unMarkovizedTree;
	}
}
