import numpy as np
from utils import restrict_angle

class CostFunction:

    def __init__(self):
        pass

class LogDetCost(CostFunction):

    def __init__(self, y_dim):
        CostFunction.__init__(self)
        self.y_dim = y_dim

    def getCost(self, Sigma):
        cost = np.log(np.linalg.det(Sigma))
        #print("Log det Sigma = ", cost)
        return cost

class DeltaBearingCost(CostFunction):

    def __init__(self, y_dim):
        CostFunction.__init__(self)
        self.y_dim = y_dim

    def getCost(self, x, y):

        num_targets = int(len(y)/self.y_dim)
        bearing_diffs = []
        for i in range(num_targets):
            for j in range(i + 1, num_targets):
                targ_pos1 = y[i * self.y_dim:i * self.y_dim + 2]
                targ_pos2 = y[j * self.y_dim:j * self.y_dim + 2]
                bearing1 = restrict_angle(np.arctan2(targ_pos1[1] - x[1], targ_pos1[0] - x[0]) - x[2])
                bearing2 = restrict_angle(np.arctan2(targ_pos2[1] - x[1], targ_pos2[0] - x[0]) - x[2])

                # above angles in range [-pi to pi],
                bearing_diffs.append(np.abs(restrict_angle(bearing1 - bearing2)))

        # want to maximize the difference, therefore minimize the negative
        bearing_diffs_neg = [-1 * diff for diff in bearing_diffs]
        return sum(bearing_diffs_neg)

class GateOverlapCost(CostFunction):

    def __init__(self, y_dim, level):
        CostFunction.__init__(self)
        self.y_dim = y_dim  # dimension of target state
        if level == 0.95:
            self.k_alpha = 3.84
        elif level == 0.99:
            self.k_alpha = 6.64
        elif level == 0.999:
            self.k_alpha = 10.83
        else:
            print("not recognized target gate level")

    def getCost(self, x, y, inn_cov_list):
        """
        compute the gate overlap cost using
        :param x: state of the sensor
        :param y: state of the target system (multi-target)
        :param Sigma: covariance of the target system
        :return: cost of the search node
        """

        #print('inn_cov_list size = ', len(inn_cov_list))

        num_targets = int(len(y)/self.y_dim)
        gates = []                          # list of gate volumes for each target
        for i in range(num_targets):
            targ_pos = y[i * self.y_dim:i * self.y_dim + 2]
            bearing = restrict_angle(np.arctan2(targ_pos[1] - x[1], targ_pos[0] - x[0]) - x[2])
            gate_volume = 2 * np.sqrt(self.k_alpha) * np.sqrt(np.linalg.det(inn_cov_list[i]))
            bearing_min = restrict_angle(bearing - gate_volume/2)
            bearing_max = restrict_angle(bearing + gate_volume/2)
            gates.append((bearing_min, bearing_max))

        total_volume = 0
        # with gate intervals, calculate overlapping gate volume
        for i in range(num_targets):
            for j in range(i+1, num_targets):
                # gate 1 = (a,b)
                # gate 2 = (c,d)
                a = gates[i][0]
                b = gates[i][1]
                c = gates[j][0]
                d = gates[j][1]
                # map to [0, 2pi]
                a, b, c, d, = a + np.pi, b + np.pi, c + np.pi, d + np.pi
                overlap = self.get_overlapped_bearing(a, b, c, d)
                total_volume += overlap

        return total_volume

    def get_overlapped_bearing(self, a, b, c, d):
        """
        given two gates (a, b) and (c, d), this function calculates the overlappind interval of the bearing gates
        :param a: min angle of gate 1
        :param b: max angle of gate 1
        :param c: min angle of gate 2
        :param d: max angle of gate 2
        :return:
        """

        if b < a:  # (a, b) is wrapped
            if d < c:  # (c, d) is also wrapped
                return self.get_double_wrapped_overlap((a, b), (c, d))
            else:
                return self.get_single_wrapped_overlap((a, b), (c, d))
        elif d < c:
            return self.get_single_wrapped_overlap((c, d), (a, b))
        else:
            return self.get_no_wrap_overlap((a, b), (c, d))

    def get_no_wrap_overlap(self, g1, g2):
        """
        calculate regular overlap without any wrapped gates
        :param g1:
        :param g2:
        :return:
        """
        a = g1[0]
        b = g1[1]
        c = g2[0]
        d = g2[1]

        if a < c:
            if b > c and b < d:
                return b - c
            elif b > d:
                return d - c
            else:
                return 0
        elif c < a:
            if d > a and d < b:
                return d - a
            elif d > b:
                return b - a
            else:
                return 0

    def get_single_wrapped_overlap(self, g1, g2):
        """
        calculate overlap where g1 is wrapped
        :param g1: wrapped gate (min, max) min will be greater than max due to wrapping and summation with pi
        :param g2: unwrapped gate
        :return:
        """
        a = g1[0]
        b = g1[1]
        c = g2[0]
        d = g2[1]
        if c < b:
            if d > b:
                return b - c
            else:
                return d - c
        elif d > a:
            if c < a:
                return d - a
            else:
                return d - c
        else:
            return 0

    def get_double_wrapped_overlap(self, g1, g2):
        """
        calculate overlap where both g1 ans g2 are wrapped.
        :param g1: wrapped gate
        :param g2: wrapped gate
        :return:
        """
        a = g1[0]
        b = g1[1]
        c = g2[0]
        d = g2[1]
        if b > d:
            if c < a:
                return d + 2 * np.pi - a
            else:
                return d + 2 * np.pi - c
        else:
            if a < c:
                return b + 2 * np.pi - c
            else:
                return b + 2 * np.pi - a
