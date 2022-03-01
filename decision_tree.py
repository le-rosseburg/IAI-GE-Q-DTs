"""
Implementation of the class DecisionTree and the class Leaf 
incl. q-learning and random initialization

"""
import numpy as np


class DecisionTree:
    """
    Class to represent a decision tree for grammatical evolution.

    :param (string) phenotype: The phenotype of the decision tree
    :param (float) current_reward : The current reward of the decision tree
    :param (dictionary) leaves : A dictionary containing all existing leaves
    :param (int) leaf_count : The total number of existing leaves
    :param (Leaf) last_leaf : The last leaf visited
    :param exec_ : Path to the resulting byte compiled file generated from the phenotype

    :func get_action: Returns the index of the optimal action in the current state
    :func set_reward: Sets the current reward of the decision tree to the transferred value
    :func new_episode: Prepares the decsion tree for a new episode by setting the value of the last visited leaf back to 'None'
    """

    def __init__(self, phenotype, leaf):
        """
        Initializes a new instance of the class DecisonTree.

        :param (string) phenotype: The phenotype representing the decision tree.
        :param (Leaf) leaf: The LeafClass used to build the decision tree.
        """
        self.phenotype = phenotype
        self.current_reward = 0
        self.leaves = {}
        self.leaf_count = 0
        self.last_leaf = None

        while "_leaf" in self.phenotype:
            # create new leaf, add leaf to dictionary and update the leaf counter
            new_leaf = leaf()
            leaf_name = "leaf_{}".format(self.leaf_count)
            self.leaves[leaf_name] = new_leaf
            self.leaf_count += 1

            # replace phenotype entry with newly created leaf
            self.phenotype = self.phenotype.replace(
                "_leaf", "'{}.get_action()'".format(leaf_name), 1
            )
            self.phenotype = self.phenotype.replace("_leaf", "{}".format(leaf_name), 1)

        self.exec_ = compile(self.phenotype, "<string>", "exec", optimize=2)

    def get_action(self, observation):
        """
        Executes the phenotype of the decision tree with the current state variables of the environment,
        applies the q-learning update to the current leaf and returns the index of the selected action for the current state.

        :param observation: The current state variables of the environment.

        Returns:
            (int) action: The index of the selected action in the current state
        """
        if len(self.phenotype) == 0:
            return None

        variables = {}  # {"out": None, "leaf": None}
        for idx, i in enumerate(observation):
            variables["_in_{}".format(idx)] = i
        variables.update(self.leaves)

        # executing phenotype with the current state variables of the environment
        exec(self.exec_, variables)

        current_leaf = self.leaves[variables["leaf"]]
        current_q_value = max(current_leaf.q)
        if self.last_leaf is not None:
            # apply q-learning update to current leaf
            self.last_leaf.update(self.current_reward, current_q_value)

        # update last used leaf
        self.last_leaf = current_leaf
        action = current_leaf.get_action()
        return action

    def set_reward(self, reward):
        """
        Sets the current reward of the decision tree to the transferred value

        :param (float) reward: A value containing the new reward of the decision tree
        """
        self.current_reward = reward

    def new_episode(self):
        """
        Prepares the decsion tree for a new episode
        by setting the value of the last visited leaf back to 'None'
        """
        self.last_leaf = None

    def __str__(self):
        """
        Represents the class object as a string and is called
        when the functions print() or str() are invoked on the class object.

        returns:
            (string) phenotype: The phenotype of the decision tree
        """
        return self.phenotype


class Leaf:
    """
    Class to represent a leaf used to build a decision tree.

    :param (list<int>) parent: A list with the parents of the leaf
    :param (int) last_action: The index of the last used action
    :param (list<int>) used_actions: list with the count of how often each action was used
    :param (list<float>) q: The q-values of the leaf

    :func get_action: Returns the index of the optimal action in the current state
    :func update: Applies the q-learning update to the leaf
    :func count_actions: Increases the count of the used action by one
    """

    def __init__(
        self,
        n_actions,
        learning_rate=0.001,
        discount_factor=0.05,
        epsilon=0.05,
        randInit=False,
        low=-100,
        up=100,
    ):
        """
        Initializes a new instance of the class Leaf.

        :param (string) n_actions: Number of possible actions
        :param (float) learning_rate: The used learning rate for q-learning
        :param (float) discount_factor: The used discount factor for q-learning
        :param (float) epsilon: The used epsilon for the e-greedy strategy
        :param (bool) randInit: Boolean to determine if q-values are randomly initialized or not
        :param (float) low: The minimum permissible q-value when randomly initialized
        :param (float) up: The maximum permissible q-value when randomly initialized
        """
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon

        self.last_action = None
        self.used_actions = [1] * n_actions
        # check if the q-values of the leaf should be random initialized or not
        if randInit:
            self.q = np.random.uniform(low, up, self.n_actions)
        else:
            self.q = np.zeros(self.n_actions, dtype=np.float32)

    def get_action(self):
        """
        Selects the action for the current leaf using the e-greedy strategy.

        Returns:
            (int) action: The index of the selected action
        """
        # apply e-greedy strategy
        if np.random.uniform() < self.epsilon:
            # choose random action
            action = np.random.randint(self.n_actions)
        else:
            # choose randomly between all actions that have the maximum q-value
            max_v = max(self.q)
            indices = [i for i, v in enumerate(self.q) if v == max_v]
            action = np.random.choice(indices)

        # update and count last used action
        self.last_action = action
        self.count_actions()
        return action

    def update(self, reward, q_next):
        """
        Updates the q-values of the leaf by applying the q-learning update.

        """
        if self.last_action is not None:
            """
            Special szenario, used for LunarLander environment:
            "The learning rate has been set to 1/k, where k is the number of visits to the [last] state-action pair."
            "This guarantees that the state-action function converges to the optimum with k → ∞"
            """
            # apply q-learning update
            self.q[self.last_action] += self.learning_rate * (
                reward + self.discount_factor * q_next - self.q[self.last_action]
            )

    def count_actions(self):
        """
        Increases the count of the last used action by one.
        """
        self.used_actions[self.last_action] += 1

    def __repr__(self):
        """
        Represents the class object as a string and is called
        when the function repr() is invoked on the class object.

        returns:
            (string): The q-values of the leaf seperated by comma
        """
        return ", ".join(["{:.2f}".format(k) for k in self.q])

    def __str__(self):
        """
        Represents the class object as a string and is called
        when the functions print() or str() are invoked on the class object.

        returns:
            (string): The string representation of the class objekt defined by the function __repr__
        """
        return repr(self)
