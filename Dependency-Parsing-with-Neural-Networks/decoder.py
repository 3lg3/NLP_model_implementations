from conll_reader import DependencyStructure, DependencyEdge, conll_reader
from collections import defaultdict
import copy
import sys

import numpy as np
from tensorflow import keras

from extract_training_data import FeatureExtractor, State


# Last Modified by Zhaoze Zhang @ 06:10 pm Nov/15

class Parser(object):

    def __init__(self, extractor, modelfile):
        self.model = keras.models.load_model(modelfile)
        self.extractor = extractor

        # The following dictionary from indices to output actions will be useful
        self.output_labels = dict([(index, action) for (action, index) in extractor.output_labels.items()])

    def parse_sentence(self, words, pos):
        state = State(range(1, len(words)))
        state.stack.append(0)

        while state.buffer:
            # TODO: Write the body of this loop for part 4
            features = self.extractor.get_input_representation(words, pos, state).reshape(1, -1)
            probs = self.model.predict(features)[0]

            # zip the actions and problems then sort descendingly
            action_list = list(self.output_labels.values())
            actions = [x for _, x in sorted(zip(probs, action_list), reverse=True)]
            # print(actions)

            for action in actions:
                arc = action[0]
                label = action[1]
                blen = len(state.buffer)
                slen = len(state.stack)
                if arc == 'shift':
                    #  freely shift if there are more than 1 words in buffer
                    #  shift the only word out of buffer is legal only when stack is empty
                    if blen > 1 or (blen == 1 and slen == 0):
                        state.shift()
                        # print("shift success")
                        break
                elif arc == 'left_arc':
                    # left_arc allowed when stack is not empty and root node is not the target of left-arc
                    if slen > 0 and state.stack[-1] != 0:
                        state.left_arc(label)
                        # print("left_arc success")
                        break
                else:  # right_arc
                    # right_arc when stack is not empty
                    if slen > 0:
                        state.right_arc(label)
                        # print("right_arc success")
                        break
        # print("out of loop")
        result = DependencyStructure()
        for p, c, r in state.deps:
            result.add_deprel(DependencyEdge(c, words[c], pos[c], p, r))
        # print("setence parsed!")
        return result


if __name__ == "__main__":

    WORD_VOCAB_FILE = 'data/words.vocab'
    POS_VOCAB_FILE = 'data/pos.vocab'

    try:
        word_vocab_f = open(WORD_VOCAB_FILE, 'r')
        pos_vocab_f = open(POS_VOCAB_FILE, 'r')
    except FileNotFoundError:
        print("Could not find vocabulary files {} and {}".format(WORD_VOCAB_FILE, POS_VOCAB_FILE))
        sys.exit(1)

    extractor = FeatureExtractor(word_vocab_f, pos_vocab_f)
    parser = Parser(extractor, sys.argv[1])

    with open(sys.argv[2], 'r') as in_file:
        for dtree in conll_reader(in_file):
            words = dtree.words()
            pos = dtree.pos()
            deps = parser.parse_sentence(words, pos)
            print(deps.print_conll())
            print()
