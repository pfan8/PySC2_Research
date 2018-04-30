from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features
from sklearn.cluster import KMeans

import math
import time

# Functions
_NOOP = actions.FUNCTIONS.no_op.id

# Unit IDs
_NEUTRAL_VESPENE_GEYSER = 342

# Features
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index


class CountAgent(base_agent.BaseAgent):
    def step(self, obs):
        super(CountAgent, self).step(obs)

        time.sleep(0.5)
        unit_type = obs.observation['screen'][_UNIT_TYPE]
        vespene_y, vespene_x = (unit_type == _NEUTRAL_VESPENE_GEYSER).nonzero()

        vespene_geyser_count = int(math.ceil(len(vespene_y) / 97))

        units = []
        for i in range(0, len(vespene_y)):
            units.append((vespene_x[i], vespene_y[i]))

        kmeans = KMeans(n_clusters=vespene_geyser_count)
        kmeans.fit(units)
        vespene1_x = int(kmeans.cluster_centers_[0][0])
        vespene1_y = int(kmeans.cluster_centers_[0][1])
        vespene2_x = int(kmeans.cluster_centers_[1][0])
        vespene2_y = int(kmeans.cluster_centers_[1][1])
        kmeans.fit(units)
        print(kmeans.cluster_centers_)
        """
        [[60. 21.]
         [25. 56.]]
        """
        kmeans.fit(units)
        print(kmeans.cluster_centers_)
        """
        [[25. 56.]
         [60. 21.]]
        """
        return actions.FunctionCall(_NOOP, [])