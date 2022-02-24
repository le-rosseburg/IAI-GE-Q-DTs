#!/usr/bin/python3
"""
Implementation of the grammatical evolution translator

"""

# translates geno- to phenotype
class Translator:
    def __init__(self):
        """
        Initializes a new instance of the Grammatical Evolution
        :param grammar: A dictionary containing the rules of the grammar and their production
        """
        self.operators = grammar
