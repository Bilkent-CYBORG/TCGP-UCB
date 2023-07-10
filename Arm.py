"""
This class represents a base arm
"""


class Arm:
    def __init__(self, unique_id, context, sup_outcome, grp_outcome, grp_thresh, group_id=None):
        self.grp_thresh = grp_thresh
        self.grp_outcome = grp_outcome  # Only used by the benchmark
        self.sup_outcome = sup_outcome  # Only used by the benchmark
        self.unique_id = unique_id
        self.context = context
        self.group_id = group_id
