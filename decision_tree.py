"""
Implementation of the class DecisionTree and the class Leaf 
incl. q-learning and random initialization

"""
import numpy as np


class DecisionTree:
    """
    Class to represent a decision tree for grammatical evolution.

    :param (str) phenotype: The phenotype of the decision tree
    :param (float) current_reward : The current reward of the decision tree
    :param (dict) leaves : A dictionary containing all existing leaves
    :param (int) leaf_count : The total number of existing leaves
    :param (Leaf) last_leaf : The last leaf visited
    :param (Code object) exec_ : Path to the resulting byte compiled file generated from the phenotype

    :func get_action: Returns the index of the optimal action in the current state
    :func set_reward: Sets the current reward of the decision tree to the transferred value
    :func new_episode: Prepares the decision tree for a new episode by setting the value of the last visited leaf back to 'None'
    """

    def __init__(self, phenotype, leaf):
        """
        Initializes a new instance of the class DecisionTree.

        :param (str) phenotype: The phenotype representing the decision tree.
        :param (Leaf) leaf: The Leaf class used to build the decision tree.
        """
        assert type(phenotype) == str, "'phenotype' must be of type 'str', got {}".format(type(phenotype))
        self.phenotype = phenotype
        self.current_reward = 0
        self.leaves = {}
        self.leaf_count = 0
        self.last_leaf = None

        while "_leaf" in self.phenotype:
            # Create new leaf, add leaf to dictionary and update the leaf counter
            new_leaf = leaf()
            leaf_name = "leaf_{}".format(self.leaf_count)
            self.leaves[leaf_name] = new_leaf
            self.leaf_count += 1

            # Replace phenotype entry with newly created leaf
            self.phenotype = self.phenotype.replace(
                "_leaf", "'{}.get_action()'".format(leaf_name), 1
            )
            self.phenotype = self.phenotype.replace("_leaf", "{}".format(leaf_name), 1)

        # Generates executable python code object from the phenotype
        self.exec_ = compile(self.phenotype, "<string>", "exec", optimize=2)

    def get_action(self, observation):
        """
        Executes the phenotype of the decision tree with the current state variables of the environment,
        applies the q-learning update to the current leaf and returns the index of the selected action for the current state.

        :param observation: The current state variables of the environment.

        Returns:
            (int) action: The index of the selected action in the current state
        """
        assert len(observation) > 0

        if len(self.phenotype) == 0:
            return None

        # Insert the observation variables and the decision tree leaves into 'variables'
        variables = {}  # {"out": None, "leaf": None}
        for idx, i in enumerate(observation):
            variables["_in_{}".format(idx)] = i
        variables.update(self.leaves)

        # Executing phenotype with the current state variables of the environment
        assert len(variables) == (len(self.leaves) + len(observation))
        exec(self.exec_, variables)

        current_leaf = self.leaves[variables["leaf"]]
        current_q_value = max(current_leaf.q)
        if self.last_leaf is not None:
            # Apply q-learning update to current leaf
            self.last_leaf.update(self.current_reward, current_q_value)

        # Set last leaf to current leaf
        self.last_leaf = current_leaf
        action = current_leaf.get_action()
        return action

    def set_reward(self, reward):
        """
        Sets the current reward of the decision tree to the transferred value

        :param (float) reward: A value containing the new reward of the decision tree
        """
        assert type(reward) == float, "'reward' must be of type 'float', got {}".format(type(reward))
        self.current_reward = reward

    def new_episode(self):
        """
        Prepares the decision tree for a new episode
        by setting the value of the last visited leaf back to 'None'
        """
        self.last_leaf = None

    def __str__(self):
        """
        Represents the class object as a string and is called
        when the functions print() or str() are invoked on the class object.

        returns:
            (str) phenotype: The phenotype of the decision tree
        """
        assert type(self.phenotype) == str, "'phenotype' must be of type 'str', got {}".format(type(reward))
        return self.phenotype


class Leaf:
    """
    Class that represents a leaf of a decision tree.

    :param (int) last_action: The index of the last used action
    :param (list<int>) used_actions: A list with the count of how often each action was used
    :param (list<float>) q: The q-values of the leaf

    :func get_action: Returns the index of the selected action in the current state
    :func update: Applies the q-learning update to the leaf
    :func count_actions: Increases the count of the used action by one
    """

    def __init__(
        self,
        n_actions,
        learning_rate=0.001,
        discount_factor=0.05,
        epsilon=0.05,
        randInit=True,
        low=-1,
        up=1,
    ):
        """
        Initializes a new instance of the class Leaf.

        :param (str) n_actions: The number of possible actions
        :param (float) learning_rate: The used learning rate for q-learning
        :param (float) discount_factor: The used discount factor for q-learning
        :param (float) epsilon: The used epsilon for the e-greedy strategy
        :param (bool) randInit: A boolean to determine if q-values are randomly initialized or not
        :param (float) low: The minimum permissible q-value when randomly initialized
        :param (float) up: The maximum permissible q-value when randomly initialized
        """
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon

        self.last_action = None
        self.used_actions = [1] * n_actions
        # Check if the q-values of the leaf should be random initialized or not
        assert type(randInit) == bool, "'randInit' must be of type 'bool', got {}.".format(type(randInit))
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
        assert len(self.q) > 0

        # Apply e-greedy strategy
        if np.random.uniform() < self.epsilon:
            # Choose random action
            action = np.random.randint(self.n_actions)
        else:
            # Choose randomly between all actions that have the maximum q-value
            max_v = max(self.q)
            indices = [i for i, v in enumerate(self.q) if v == max_v]
            action = np.random.choice(indices)

        # Update and count last used action
        self.last_action = action
        self.used_actions[self.last_action] += 1
        return action

    def update(self, reward, q_next):
        """
        Updates the q-values of the leaf by applying the q-learning update.

        :param (float) reward: The reward of the current observation
        :param (float) q_next: The q-value of the subsequent leaf
        """
        if self.last_action is not None:
            """
            A dynamic learning rate is used for the LunarLander-v2 environment.
            "The learning rate has been set to 1/k, where k is the number of visits to the [last] state-action pair.
            This guarantees that the state-action function converges to the optimum with k → ∞" [CustodeIacca2021]

            [CustodeIacca2021] Leonardo Lucio Custode, Giovanni Iacca, 'Evolutionary learning
            of interpretable decision trees', 2021
            """
            # Compute the dynamic learning_rate if it is set to "auto"
            if self.learning_rate == "auto":
                self.learning_rate = 1 / self.used_actions[self.last_action]
            # Apply the q-learning update
            assert type(self.learning_rate) == float, "'learning_rate' must be of type 'float', got {}.".format(type(randInit))
            self.q[self.last_action] += self.learning_rate * (
                reward + self.discount_factor * q_next - self.q[self.last_action]
            )

    def __repr__(self):
        """
        Represents the class object as a string and is called
        when the function repr() is invoked on the class object.

        Returns:
            (str): The comma seperated q-values of the leaf
        """
        return ", ".join(["{:.2f}".format(k) for k in self.q])

    def __str__(self):
        """
        Represents the class object as a string and is called
        when the functions print() or str() are invoked on the class object.

        Returns:
            (str): The string representation of the class objekt defined by the function __repr__
        """
        return repr(self)
