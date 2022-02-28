"""
Implementation of the grammatical evolution translator which translates a genotype into a phenotype.

"""

import re


class GETranslator:
    def __init__(self, grammar):
        """
        Initializes a new instance of the Grammatical Evolution
        :param grammar: A dictionary containing the rules of the grammar and their production
        """
        self.operators = grammar

    def _calculate_phenotype_value(self, candidate, gene):
        """
        Calculates a phenotype value based on the production rule derived from a candidate and a gene.
        :param candidate: A candidate of the grammar?
        :param gene: A gene of a genotype

        Returns:
            value: Calculated phenotype value
        """
        key = candidate.replace("<", "").replace(">", "")
        value = self.operators[key][gene % len(self.operators[key])]
        return value

    def genotype_to_str(self, genotype):
        """
        This method translates a genotype into an executable program (phenotype).
        If the individual runs out of genes, it restarts from the beginning, as suggested by Ryan et al 1998.
        :param genotype: A genotype containing genes

        Returns:
            phenotype: A list of values depicting the phenotype
            genes_used: Number of genes that were used
        """
        string = "<bt>"
        candidates = [None]
        ctr = 0
        _max_trials = 1
        genes_used = 0

        while len(candidates) > 0 and ctr <= _max_trials:
            if ctr == _max_trials:
                return "", len(genotype)
            for gene in genotype:
                candidates = re.findall("<[^> ]+>", string)
                if len(candidates) > 0:
                    value = self._calculate_phenotype_value(candidates[0], gene)
                    string = string.replace(candidates[0], value, 1)
                    genes_used += 1
                else:
                    break
            ctr += 1

        phenotype = self._fix_phenotype(string)
        return phenotype, genes_used

    def _fix_phenotype(self, string):
        """
        This method fixes the phenotype string so that it can be used in a DecisionTree.
        :param string: A string containing a phenotype

        Returns:
            fixed_phenotype: The fixed phenotype string
        """
        # If parenthesis are present in the outermost block, remove them
        if string[0] == "{":
            string = string[1:-1]

        # Split in lines
        string = string.replace(";", "\n").replace("{", "{\n").replace("}", "}\n")
        lines = string.split("\n")

        fixed_lines = []
        n_tabs = 0
        # Fix lines
        for line in lines:
            if len(line) > 0:
                fixed_lines.append(
                    " " * 4 * n_tabs + line.replace("{", "").replace("}", "")
                )

                if line[-1] == "{":
                    n_tabs += 1
                while len(line) > 0 and line[-1] == "}":
                    n_tabs -= 1
                    line = line[:-1]
                # Not sure if needed
                # if n_tabs >= 100:
                #    return "None"
        fixed_phenotype = "\n".join(fixed_lines)
        return fixed_phenotype
