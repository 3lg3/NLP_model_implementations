import sys
from collections import defaultdict
from math import fsum
import math


class Pcfg(object):
    """
    Represent a probabilistic context free grammar. 
    """

    def __init__(self, grammar_file):
        self.rhs_to_rules = defaultdict(list)
        self.lhs_to_rules = defaultdict(list)
        self.startsymbol = None
        self.read_rules(grammar_file)

    def read_rules(self, grammar_file):

        for line in grammar_file:
            line = line.strip()
            if line and not line.startswith("#"):
                if "->" in line:
                    rule = self.parse_rule(line.strip())
                    lhs, rhs, prob = rule
                    self.rhs_to_rules[rhs].append(rule)
                    self.lhs_to_rules[lhs].append(rule)
                else:
                    startsymbol, prob = line.rsplit(";")
                    self.startsymbol = startsymbol.strip()

    def parse_rule(self, rule_s):
        lhs, other = rule_s.split("->")
        lhs = lhs.strip()
        rhs_s, prob_s = other.rsplit(";", 1)
        prob = float(prob_s)
        rhs = tuple(rhs_s.strip().split())
        return lhs, rhs, prob

    def verify_grammar(self):
        """
        Return True if the grammar is a valid PCFG in CNF.
        Otherwise return False. 
        """
        # TODO, Part 1
        for lhs, rules in self.lhs_to_rules.items():
            probs = []
            for rule in rules:
                rhs = rule[1]
                probs.append(rule[2])
                if len(rhs) == 0 or len(rhs) > 2:
                    print("Invalid grammar: invalid length of rhs!")
                    return False
                if len(rhs) == 1:
                    # if not rhs[0].islower():
                    if rhs[0].isupper():
                        print("Invalid grammar: terminal symbol {} is upper "
                              "for the symbol {}!".format(rhs[0], rule[0]))
                        return False
                if len(rhs) == 2:
                    if not rhs[0].isupper():
                        print("Invalid grammar: nonterminal symbol {} is not" 
                              " upper for the symbol {}!".format(rhs[0], rule[0]))
                        return False
                    elif not rhs[1].isupper():
                        print("Invalid grammar: nonterminal symbol {} is not"
                              " upper for the symbol {}!".format(rhs[1], rule[0]))
                        return False

            probs_sum = fsum(probs)
            if not math.isclose(probs_sum, 1):
                print("Invalid grammar: sum of all probabilities for the same lhs is not 1.0")
                return False

        return True


if __name__ == "__main__":
    with open(sys.argv[1], 'r') as grammar_file:
        grammar = Pcfg(grammar_file)

    # for k in grammar.rhs_to_rules.keys():
    #     if k == ('tonystark',):
    #         print("i am ironman!")
