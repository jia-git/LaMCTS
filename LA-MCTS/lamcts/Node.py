# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 
import math
import logging

import numpy as np

from .Classifier import BOClassifier, TurBOClassifier

logger = logging.getLogger('lamcts')

class Node:
    obj_counter = 0

    # If a leave holds >= SPLIT_THRESH, we split into two new nodes.

    def __init__(self, parent=None, cp: float = 1.0, dims=0, reset_id=False, kernel_type="rbf", gamma_type="auto",
                 solver_type="turbo"):
        # Note: every node is initialized as a leaf,
        # only internal nodes equip with classifiers to make decisions
        # if not is_root:
        #     assert type( parent ) == type( self )
        self._cp = cp
        self.dims = dims
        self._x_bar = 0
        self.n = 0
        if solver_type == 'turbo':
            self.classifier = TurBOClassifier(self.dims, kernel_type, gamma_type)
        else:
            self.classifier = BOClassifier(self.dims, kernel_type, gamma_type)

        # insert curt into the kids of parent
        self.parent = parent
        self.kids = []  # 0:good, 1:bad

        self.bag = []
        self.is_svm_splittable = False

        if reset_id:
            Node.obj_counter = 0

        self.id = Node.obj_counter

        # data for good and bad kids, respectively
        Node.obj_counter += 1

    def update_kids(self, good_kid, bad_kid):
        assert len(self.kids) == 0
        self.kids.append(good_kid)
        self.kids.append(bad_kid)
        good_mean = self.kids[0].classifier.get_mean()
        bad_mean = self.kids[1].classifier.get_mean()
        if good_mean < bad_mean:
            logger.warning(f"good_mean({good_mean}) < bad_mean({bad_mean})")

    def is_good_kid(self):
        if self.parent is not None:
            if self.parent.kids[0] == self:
                return True
            else:
                return False
        else:
            return False

    def is_leaf(self):
        if len(self.kids) == 0:
            return True
        else:
            return False

    def visit(self):
        self.n += 1

    def update_bag(self, samples):
        assert len(samples) > 0

        self.bag.clear()
        self.bag.extend(samples)
        self.classifier.update_samples(self.bag)
        if len(self.bag) <= 2:
            self.is_svm_splittable = False
        else:
            self.is_svm_splittable = self.classifier.is_splittable_svm()
        self._x_bar = self.classifier.get_mean()
        self.n = len(self.bag)

    def clear_data(self):
        self.bag.clear()

    def get_name(self):
        # state is a list of jsons
        return "node" + str(self.id)

    def pad_str_to_8chars(self, ins, total):
        if len(ins) <= total:
            ins += ' ' * (total - len(ins))
            return ins
        else:
            return ins[0:total]

    def get_rand_sample_from_bag(self):
        if len(self.bag) > 0:
            upeer_boundary = len(list(self.bag))
            rand_idx = np.random.randint(0, upeer_boundary)
            return self.bag[rand_idx][0]
        else:
            return None

    def get_parent_str(self):
        return self.parent.get_name()

    def propose_samples(self, path=None, lb=None, ub=None, samples=None, func=None):
        return self.classifier.propose_samples(path, lb, ub, samples, func)

    def propose_samples_bo(self, num_samples, path, lb, ub, samples):
        proposed_X = self.classifier.propose_samples_bo(num_samples, path, lb, ub, samples)
        return proposed_X

    def propose_samples_turbo(self, num_samples, path, func):
        proposed_X, fX = self.classifier.propose_samples_turbo(num_samples, path, func)
        return proposed_X, fX

    def __str__(self):
        name = self.get_name()
        name = self.pad_str_to_8chars(name, 7)
        name += (self.pad_str_to_8chars('is good:' + str(self.is_good_kid()), 15))
        name += (self.pad_str_to_8chars('is leaf:' + str(self.is_leaf()), 15))

        val = 0
        name += (self.pad_str_to_8chars(' val:{0:.4f}   '.format(round(self.x_bar, 3)), 20))
        name += (self.pad_str_to_8chars(' _uct:{0:.4f}   '.format(round(self.uct, 3)), 20))

        name += self.pad_str_to_8chars('sp/n:' + str(len(self.bag)) + "/" + str(self.n), 15)
        upper_bound = np.around(np.max(self.classifier.X, axis=0), decimals=2)
        lower_bound = np.around(np.min(self.classifier.X, axis=0), decimals=2)
        boundary = ''
        for idx in range(0, self.dims):
            boundary += str(lower_bound[idx]) + '>' + str(upper_bound[idx]) + ' '

        # name  += ( self.pad_str_to_8chars( 'bound:' + boundary, 60 ) )

        parent = '----'
        if self.parent is not None:
            parent = self.parent.get_name()
        parent = self.pad_str_to_8chars(parent, 10)

        name += (' parent:' + parent)

        kids = ''
        kid = ''
        for k in self.kids:
            kid = self.pad_str_to_8chars(k.get_name(), 10)
            kids += kid
        name += (' kids:' + kids)

        return name

    @property
    def uct(self):
        return self._x_bar + 2 * self._cp * math.sqrt(
            2 * np.power(self.parent.n, 0.5) / self.n) if self.parent and self.n else float('inf')

    @property
    def x_bar(self):
        return self._x_bar

    @x_bar.setter
    def x_bar(self, x_bar: float):
        self._x_bar = x_bar

    def update_xbar(self, value: float):
        self.n += 1
        self._x_bar += (value - self._x_bar) / self.n

    def get_n(self):
        return self.n

    def train_and_split(self):
        assert len(self.bag) >= 2
        self.classifier.update_samples(self.bag)
        good_kid_data, bad_kid_data = self.classifier.split_data()
        assert len(good_kid_data) + len(bad_kid_data) == len(self.bag)
        return good_kid_data, bad_kid_data

    def plot_samples_and_boundary(self, func):
        name = self.get_name() + ".pdf"
        self.classifier.plot_samples_and_boundary(func, name)
