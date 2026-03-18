#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from arguments import ParamGroup

class PlanningDMControlParams(ParamGroup):
    def __init__(self, parser, sentinel=False):
        self.constraint_tolerance = 1e-6
        self.unit_circle_penalty_weight = 10.0
        self.enable_projection = True
        self.enable_penalty = True
        self.flow_magnitude_exponent = 1.0
        super().__init__(parser, "Planning DM Control Parameters", sentinel)
