# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 
import json
import logging
import math
import os
import os.path
import pickle
from typing import Callable, Tuple

import numpy as np

from .Node import Node
from .utils import latin_hypercube, from_unit_cube

logger = logging.getLogger('lamcts')

class MCTS:
    DEFAULT_CP = 1.0
    DEFAULT_LEAF_SIZE = 20
    DEFAULT_KERNAL_TYPE = 'rbf'
    DEFAULT_GAMMA_TYPE = 'auto'
    DEFAULT_SOLVER_TYPE = 'turbo'

    def __init__(self, lb, ub, dims, ninits, func: Callable, Cp: float = DEFAULT_CP, leaf_size: int = DEFAULT_LEAF_SIZE,
                 kernel_type: str = DEFAULT_KERNAL_TYPE, gamma_type: str = DEFAULT_GAMMA_TYPE,
                 solver_type: str = DEFAULT_SOLVER_TYPE):
        self.dims = dims
        self.samples = []
        self.nodes = []
        self.Cp = Cp
        self.lb = lb
        self.ub = ub
        self.ninits = ninits
        self.func = func
        self.curt_best_value = float("-inf")
        self.curt_best_sample = None
        self.best_value_trace = []
        self.sample_counter = 0
        self.visualization = False

        self.leaf_size = leaf_size
        self.kernel_type = kernel_type
        self.gamma_type = gamma_type

        self.solver_type = solver_type  # solver can be 'bo' or 'turbo'

        logger.debug(f"gamma_type: {gamma_type}")

        # we start the most basic form of the tree, 3 nodes and height = 1
        root = Node(parent=None, cp=self.Cp, dims=self.dims, reset_id=True, kernel_type=self.kernel_type,
                    gamma_type=self.gamma_type, solver_type=self.solver_type)
        self.nodes.append(root)

        self.root = root

    def populate_training_data(self):
        # only keep root
        Node.obj_counter = 0
        for node in self.nodes:
            node.clear_data()
        self.nodes.clear()
        self.root = Node(parent=None, cp=self.Cp, dims=self.dims, reset_id=True, kernel_type=self.kernel_type,
                         gamma_type=self.gamma_type, solver_type=self.solver_type)
        self.nodes.append(self.root)
        self.root.update_bag(self.samples)

    def get_leaf_status(self) -> np.ndarray:
        status = [(n.is_leaf() and len(n.bag) > self.leaf_size and n.is_svm_splittable) for n in self.nodes]
        return np.array(status)

    def get_split_idx(self):
        split_by_samples = np.argwhere(self.get_leaf_status()).reshape(-1)
        return split_by_samples

    def is_splitable(self) -> bool:
        return self.get_leaf_status().any()

    def dynamic_treeify(self):
        # we bifurcate a node once it contains over 20 samples
        # the node will bifurcate into a good and a bad kid
        self.populate_training_data()
        assert len(self.root.bag) == len(self.samples)
        assert len(self.nodes) == 1

        while self.is_splitable():
            to_split = self.get_split_idx()
            # print("==>to split:", to_split, " total:", len(self.nodes) )
            for nidx in to_split:
                parent = self.nodes[nidx]  # parent check if the boundary is splittable by svm
                assert len(parent.bag) >= self.leaf_size
                assert parent.is_svm_splittable == True
                # print("spliting node:", parent.get_name(), len(parent.bag))
                good_kid_data, bad_kid_data = parent.train_and_split()
                # creat two kids, assign the data, and push into lists
                # children's lb and ub will be decided by its parent
                assert len(good_kid_data) + len(bad_kid_data) == len(parent.bag)
                assert len(good_kid_data) > 0
                assert len(bad_kid_data) > 0
                good_kid = Node(parent=parent, cp=self.Cp, dims=self.dims, reset_id=False, kernel_type=self.kernel_type,
                                gamma_type=self.gamma_type, solver_type=self.solver_type)
                bad_kid = Node(parent=parent, cp=self.Cp, dims=self.dims, reset_id=False, kernel_type=self.kernel_type,
                               gamma_type=self.gamma_type, solver_type=self.solver_type)
                good_kid.update_bag(good_kid_data)
                bad_kid.update_bag(bad_kid_data)

                parent.update_kids(good_kid=good_kid, bad_kid=bad_kid)

                self.nodes.append(good_kid)
                self.nodes.append(bad_kid)

            # print("continue split:", self.is_splitable())

        # self.print_tree()

    def collect_samples(self, sample, value=None):
        # TODO: to perform some checks here
        if value == None:
            value = self.func(sample) * -1

        if value > self.curt_best_value:
            self.curt_best_value = value
            self.curt_best_sample = sample
            self.best_value_trace.append((value, self.sample_counter))
        self.sample_counter += 1
        if math.isnan(value):
            logger.warning(f"sample {sample} value is NaN")
            value = 0.0
        self.samples.append((sample, value))
        return value

    def init_train(self):
        # here we use latin hyper space to generate init samples in the search space
        init_points = latin_hypercube(self.ninits, self.dims)
        init_points = from_unit_cube(init_points, self.lb, self.ub)

        for point in init_points:
            self.collect_samples(point)

        header = f"{'=' * 10} collect {str(len(self.samples))} points for initializing MCTS {'=' * 10}"
        logger.debug(header)
        logger.debug(f"lb: {self.lb}")
        logger.debug(f"ub: {self.ub}")
        logger.debug(f"Cp: {self.Cp}")
        logger.debug(f"inits: {self.ninits}")
        logger.debug(f"dims: {self.dims}")
        logger.debug(f"{'=' * len(header)}")

    def print_tree(self):
        logger.debug('-' * 100)
        for node in self.nodes:
            logger.debug(node)
        logger.debug('-' * 100)

    def load_agent(self):
        node_path = 'mcts_agent'
        if os.path.isfile(node_path) == True:
            with open(node_path, 'rb') as json_data:
                self = pickle.load(json_data)
                print("=====>loads:", len(self.samples), " samples")

    def dump_agent(self):
        node_path = 'mcts_agent'
        print("dumping the agent.....")
        with open(node_path, "wb") as outfile:
            pickle.dump(self, outfile)

    def dump_samples(self):
        sample_path = 'samples_' + str(self.sample_counter)
        with open(sample_path, "wb") as outfile:
            pickle.dump(self.samples, outfile)

    def dump_trace(self):
        trace_path = 'best_values_trace'
        final_results_str = json.dumps(self.best_value_trace)
        with open(trace_path, "a") as f:
            f.write(final_results_str + '\n')

    def select(self, greedy: bool = False):
        curt_node = self.root
        path = []
        while not curt_node.is_leaf():
            if self.visualization:
                curt_node.plot_samples_and_boundary(self.func)
            ucts = [i.xbar if greedy else i.uct for i in curt_node.kids]
            max_uct = np.amax(ucts)
            choice = np.random.choice(np.argwhere(ucts == max_uct).reshape(-1), 1)[0]
            path.append((curt_node, choice))
            curt_node = curt_node.kids[choice]
        logger.debug(f"{'=>'.join([n.get_name() for n, _ in path])}")
        return curt_node, path

    @staticmethod
    def backpropogate(leaf: Node, acc):
        curt_node = leaf
        while curt_node:
            assert curt_node.n > 0
            curt_node.update_xbar(acc)
            # curt_node._x_bar = (curt_node._x_bar * curt_node.n + acc) / (curt_node.n + 1)
            # curt_node.n += 1
            curt_node = curt_node.parent

    def search(self, iterations) -> Tuple[np.ndarray, float]:
        self.init_train()
        for idx in range(self.sample_counter, iterations):
            logger.debug("")
            logger.debug("=" * 10)
            logger.debug(f"iteration: {idx}")
            logger.debug("=" * 10)
            self.dynamic_treeify()
            leaf, path = self.select()
            samples, values = leaf.propose_samples(path, self.lb, self.ub, self.samples, self.func)
            if values is None or len(values) == 0:
                values = [None for _ in range(len(samples))]
            for sample, value in zip(samples, values):
                value = self.collect_samples(sample, value)
                MCTS.backpropogate(leaf, value)
            logger.debug(f"total samples: {len(self.samples)}")
            logger.debug(f"current best f(x): {np.absolute(self.curt_best_value)}")
            # print("current best x:", np.around(self.curt_best_sample, decimals=1) )
            logger.debug(f"current best x: {self.curt_best_sample}")
        return self.curt_best_sample, self.curt_best_value