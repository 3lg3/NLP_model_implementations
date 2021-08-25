import math
import sys
from collections import defaultdict
import itertools
from grammar import Pcfg


### Use the following two functions to check the format of your data structures in part 3 ###
def check_table_format(table):
    """
    Return true if the backpointer table object is formatted correctly.
    Otherwise return False and print an error.  
    """
    if not isinstance(table, dict):
        sys.stderr.write("Backpointer table is not a dict.\n")
        return False
    for split in table:
        if not isinstance(split, tuple) and len(split) == 2 and \
                isinstance(split[0], int) and isinstance(split[1], int):
            sys.stderr.write("Keys of the backpointer table must be tuples (i,j) representing spans.\n")
            return False
        if not isinstance(table[split], dict):
            sys.stderr.write("Value of backpointer table (for each span) is not a dict.\n")
            return False
        for nt in table[split]:
            if not isinstance(nt, str):
                sys.stderr.write(
                    "Keys of the inner dictionary (for each span) must be strings representing nonterminals.\n")
                return False
            bps = table[split][nt]
            if isinstance(bps, str):  # Leaf nodes may be strings
                continue
            if not isinstance(bps, tuple):
                sys.stderr.write(
                    "Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Incorrect type: {}\n".format(
                        bps))
                return False
            if len(bps) != 2:
                sys.stderr.write(
                    "Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Found more than two backpointers: {}\n".format(
                        bps))
                return False
            for bp in bps:
                if not isinstance(bp, tuple) or len(bp) != 3:
                    sys.stderr.write(
                        "Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Backpointer has length != 3.\n".format(
                            bp))
                    return False
                if not (isinstance(bp[0], str) and isinstance(bp[1], int) and isinstance(bp[2], int)):
                    print(bp)
                    sys.stderr.write(
                        "Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Backpointer has incorrect type.\n".format(
                            bp))
                    return False
    return True


def check_probs_format(table):
    """
    Return true if the probability table object is formatted correctly.
    Otherwise return False and print an error.  
    """
    if not isinstance(table, dict):
        sys.stderr.write("Probability table is not a dict.\n")
        return False
    for split in table:
        if not isinstance(split, tuple) and len(split) == 2 and isinstance(split[0], int) and isinstance(split[1], int):
            sys.stderr.write("Keys of the probability must be tuples (i,j) representing spans.\n")
            return False
        if not isinstance(table[split], dict):
            sys.stderr.write("Value of probability table (for each span) is not a dict.\n")
            return False
        for nt in table[split]:
            if not isinstance(nt, str):
                sys.stderr.write(
                    "Keys of the inner dictionary (for each span) must be strings representing nonterminals.\n")
                return False
            prob = table[split][nt]
            if not isinstance(prob, float):
                sys.stderr.write(
                    "Values of the inner dictionary (for each span and nonterminal) must be a float.{}\n".format(prob))
                return False
            if prob > 0:
                sys.stderr.write("Log probability may not be > 0.  {}\n".format(prob))
                return False
    return True


class CkyParser(object):
    """
    A CKY parser.
    """

    def __init__(self, grammar):
        """
        Initialize a new parser instance from a grammar. 
        """
        self.grammar = grammar

    def is_in_language(self, tokens):
        """
        Membership checking. Parse the input tokens and return True if 
        the sentence is in the language described by the grammar. Otherwise
        return False
        """
        # TODO, part 2
        n = len(tokens)
        # initialize the 2D table
        table = [[set() for i in range(n + 1)] for j in range(n + 1)]
        # populate the diagonal of the CKY table.
        for i in range(n):
            for rhs in self.grammar.rhs_to_rules.keys():
                if rhs == (tokens[i],):
                    for rule in self.grammar.rhs_to_rules[rhs]:
                        table[i][i + 1].add(rule[0])

        # populate the upper diagonal cells of the CKY table.
        for length in range(2, n + 1):
            for i in range(n - length + 1):
                j = i + length
                for k in range(i + 1, j):
                    for rhs in self.grammar.rhs_to_rules.keys():
                        for B in table[i][k]:
                            for C in table[k][j]:
                                if rhs == (B, C):
                                    for rule in grammar.rhs_to_rules[rhs]:
                                        table[i][j].add(rule[0])

        if "TOP" in table[0][n]:
            return True

        return False

    def parse_with_backpointers(self, tokens):
        """
        Parse the input tokens and return a parse table and a probability table.
        """
        # TODO, part 3
        n = len(tokens)
        table = defaultdict(defaultdict)
        probs = defaultdict(defaultdict)

        # populate the diagonal of the CKY table with backpointers.
        for i in range(n):
            for rhs in self.grammar.rhs_to_rules.keys():
                if rhs == (tokens[i],):
                    for rule in self.grammar.rhs_to_rules[rhs]:
                        table[(i, i + 1)][rule[0]] = rule[1][0]
                        probs[(i, i + 1)][rule[0]] = math.log(rule[2])

        # populate the upper diagonal cells of the CKY table with backpointers.
        for length in range(2, n + 1):
            for i in range(n - length + 1):
                j = i + length
                for k in range(i + 1, j):
                    for rhs in self.grammar.rhs_to_rules.keys():
                        for B in table[(i, k)].keys():
                            for C in table[(k, j)].keys():
                                if rhs == (B, C):
                                    for rule in self.grammar.rhs_to_rules[rhs]:
                                        split = ((rule[1][0], i, k), (rule[1][1], k, j))
                                        p = math.log(rule[2]) + probs[(i, k)][rule[1][0]] + probs[(k, j)][rule[1][1]]
                                        if rule[0] not in probs[(i, j)].keys():
                                            table[(i, j)][rule[0]] = split
                                            probs[(i, j)][rule[0]] = p
                                        else:
                                            if p > probs[(i, j)][rule[0]]:
                                                table[(i, j)][rule[0]] = split
                                                probs[(i, j)][rule[0]] = p
        return table, probs


def get_tree(chart, i, j, nt):
    """
    Return the parse-tree rooted in non-terminal nt and covering span i,j.
    """
    # TODO: Part 4
    split = chart[(i, j)][nt]
    if isinstance(split, str):
        return nt, split
    else:
        return nt, get_tree(chart, split[0][1], split[0][2], split[0][0]), \
               get_tree(chart, split[1][1], split[1][2], split[1][0])


if __name__ == "__main__":
    with open('atis3.pcfg', 'r') as grammar_file:
        grammar = Pcfg(grammar_file)
        parser = CkyParser(grammar)
        toks = ['flights', 'from', 'miami', 'to', 'cleveland', '.']
        toks2 = ['miami', 'flights', 'cleveland', 'from', 'to', '.']
        print("*********** Part 2 *************")
        print(parser.is_in_language(toks))
        print(parser.is_in_language(toks2))
        print("*********** Part 3 *************")
        table,probs = parser.parse_with_backpointers(toks)
        print(check_table_format(table))
        print(check_probs_format(probs))
        print("*********** Part 4 *************")
        print(get_tree(table, 0, len(toks), grammar.startsymbol))
