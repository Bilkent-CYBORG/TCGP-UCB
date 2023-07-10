"""
This class represents a base arm's reward/outcome
"""
from Arm import Arm


class Reward:
    def __init__(self, arm: Arm, sup_perf, grp_perf):
        self.grp_perf = grp_perf
        self.sup_perf = sup_perf
        self.context = arm.context
        self.grp_thresh = arm.grp_thresh
        self.arm = arm
