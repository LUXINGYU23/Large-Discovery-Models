"""Seed learner: the released SimpleCIL prototype classifier restated for the slot."""
import logging

import numpy as np
import torch
from torch.utils.data import DataLoader

from models.base import BaseLearner
from utils.inc_net import SimpleVitNet


class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = SimpleVitNet(args, True)
        self.args = args

    def after_task(self):
        self._known_classes = self._total_classes

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)
        logging.info("Learning on %s-%s", self._known_classes, self._total_classes)
        classes = np.arange(self._known_classes, self._total_classes)
        prototypes = data_manager.get_dataset(classes, source="train", mode="test")
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test")
        self.test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=8, pin_memory=True)
        loader = DataLoader(prototypes, batch_size=128, shuffle=False, num_workers=8, pin_memory=True)
        self._network.to(self._device)
        self._network.eval()
        features, labels = [], []
        with torch.no_grad():
            for _, inputs, targets in loader:
                features.append(self._network.backbone(inputs.to(self._device)).cpu())
                labels.append(targets)
        features, labels = torch.cat(features), torch.cat(labels)
        for label in classes:
            self._network.fc.weight.data[label] = features[labels == label].mean(0).to(self._device)
