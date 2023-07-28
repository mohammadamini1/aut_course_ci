# -*- coding: utf-8 -*-

# python imports
from math import degrees
import numpy as np
import skfuzzy as fuzz

# pyfuzzy imports
from fuzzy.storage.fcl.Reader import Reader


class FuzzyController:

    def __init__(self, fcl_path):
        self.system = Reader().load_from_file(fcl_path)
        self.last_force = 0
        self.decide_counter = 0

    def _make_input(self, world):
        return dict(
            cp = world.x,
            cv = world.v,
            pa = degrees(world.theta),
            pv = degrees(world.omega)
        )


    def _make_output(self):
        return dict(
            left_fast = 0.,
            left_slow = 0.,
            stop = 0.,
            right_slow = 0.,
            right_fast = 0.,
        )


    def decide(self, world):
        input = self._make_input(world)
        output = dict(
            force = 0.
        )
        self.system.calculate(input, output)
        print self.decide_counter,
        self.decide_counter += 1
        return output['force']










    ##! fuzzification

    def linear_cal(self, start, end, z, x):
        if x < start or x > end:
            raise('x not in range')

        z = z / (end - start)
        y =  (x - start) * z
        if z < 0.: y = 1. + y
        # print 'linear cal: {} - {} value: {}  z = {} y = {}'.format(start, end, x, z, y)

        return y

    def membership_function(self, name, value):
        ret = {}
        if name == 'pa':
            pa_terms = dict(
                            up_more_right   = [0., 30., 60.],
                            up_right        = [30., 60., 90.],
                            up              = [60., 90., 120.],
                            up_left         = [90., 120., 150.],
                            up_more_left    = [120., 150., 180.],
                            down_more_left  = [180., 210., 240.],
                            down_left       = [210., 240., 270.],
                            down            = [240., 270., 300.],
                            down_right      = [270., 300., 330.],
                            down_more_right = [300., 330., 360.],
                        )
            for term in pa_terms:
                start, end = pa_terms[term][0], pa_terms[term][1]
                if value >= start and value <= end:
                    ret.update({term: self.linear_cal(start, end, 1., value)})

                start, end = pa_terms[term][1], pa_terms[term][2]
                if value >= start and value <= end:
                    ret.update({term: self.linear_cal(start, end, -1., value)})

        elif name == 'pv':
            pv_terms = dict(
                        cw_fast  = [-200, -200, -100],
                        cw_slow  = [-200, -100, 0],
                        stop     = [-100, 0, 100],
                        ccw_slow = [0, 100, 200],
                        ccw_fast = [100, 200, 200],
                    )
            for term in pv_terms:
                start, end = pv_terms[term][0], pv_terms[term][1]
                if value >= start and value <= end and start != end:
                    ret.update({term: self.linear_cal(start, end, 1., value)})

                start, end = pv_terms[term][1], pv_terms[term][2]
                if value >= start and value <= end and start != end:
                    ret.update({term: self.linear_cal(start, end, -1., value)})

        elif name == 'cp':
            cp_terms = dict(
                        left_far   = [-10, -10, -5],
                        left_near  = [-10, -2.5, 0],
                        stop       = [-2.5, 0, 2.5],
                        right_near = [0, 2.5, 10],
                        right_far  = [5, 10, 10],
                    )
            for term in cp_terms:
                start, end = cp_terms[term][0], cp_terms[term][1]
                if value >= start and value <= end and start != end:
                    ret.update({term: self.linear_cal(start, end, 1., value)})

                start, end = cp_terms[term][1], cp_terms[term][2]
                if value >= start and value <= end and start != end:
                    ret.update({term: self.linear_cal(start, end, -1., value)})

        elif name == 'cv':
            cv_terms = dict(
                        left_fast = [-5, -5, -2.5],
                        left_slow = [-5, -1, 0],
                        stop = [-1, 0, 1],
                        right_slow = [0, 1, 5],
                        right_fast = [2.5, 5, 5],
                    )
            for term in cv_terms:
                start, end = cv_terms[term][0], cv_terms[term][1]
                if value >= start and value <= end and start != end:
                    ret.update({term: self.linear_cal(start, end, 1., value)})

                start, end = cv_terms[term][1], cv_terms[term][2]
                if value >= start and value <= end and start != end:
                    ret.update({term: self.linear_cal(start, end, -1., value)})

        else:
            pass

        return ret

    def fuzzification(self, input):
        fuzzed_input = {
            'pa': dict(
                    up_more_right = 0.,
                    up_right = 0.,
                    up = 0.,
                    up_left = 0.,
                    up_more_left = 0.,
                    down_more_left = 0.,
                    down_left = 0.,
                    down = 0.,
                    down_right = 0.,
                    down_more_right = 0.
                )
            ,
            'pv': dict(
                    cw_fast = 0.,
                    cw_slow = 0.,
                    stop = 0.,
                    ccw_slow = 0.,
                    ccw_fast = 0.,
                )
            ,
            'cp': dict(
                    left_far = 0.,
                    left_near = 0.,
                    stop = 0.,
                    right_near = 0.,
                    right_far = 0.,
                )
            ,
            'cv': dict(
                    left_fast = 0.,
                    left_slow = 0.,
                    stop = 0.,
                    right_slow = 0.,
                    right_fast = 0.,
                )
            ,
        }
        for (name, value) in input.items():
            member_ships = self.membership_function(name, value)
            # print name, 'input : ', value, ' -> ', member_ships
            for member_ship in member_ships:
                fuzzed_input[name][member_ship] = member_ships[member_ship]

        # print "fuzzed_input: ", fuzzed_input
        return fuzzed_input



    ##! inference

    def max(self, arr):
        m = arr[0]
        for a in arr:
            if a > m:
                m = a
        return m

    def min(self, arr):
        m = arr[0]
        for a in arr:
            if a < m:
                m = a
        return m

    def compute_rule(self, pa, pv, force, rule_number):
        name, value = None, None
        if rule_number == 0:
            name = 'stop'
            value = self.min([
                self.max([pa['up'], pv['stop']]),
                self.max([pa['up_right'], pv['ccw_slow']]),
                self.max([pa['up_left'], pv['cw_slow']]),
            ])

        elif rule_number == 1:
            name = 'right_fast'
            value = self.min([pa['up_more_right'], pv['ccw_slow']])

        elif rule_number == 2:
            name = 'right_fast'
            value = self.min([pa['up_more_right'], pv['cw_slow']])

        elif rule_number == 3:
            name = 'left_fast'
            value = self.min([pa['up_more_left'], pv['cw_slow']])

        elif rule_number == 4:
            name = 'left_fast'
            value = self.min([pa['up_more_left'], pv['ccw_slow']])

        elif rule_number == 5:
            name = 'left_slow'
            value = self.min([pa['up_more_right'], pv['ccw_fast']])

        elif rule_number == 6:
            name = 'right_fast'
            value = self.min([pa['up_more_right'], pv['cw_fast']])

        elif rule_number == 7:
            name = 'right_slow'
            value = self.min([pa['up_more_left'], pv['cw_fast']])

        elif rule_number == 8:
            name = 'left_fast'
            value = self.min([pa['up_more_left'], pv['ccw_fast']])

        elif rule_number == 9:
            name = 'right_fast'
            value = self.min([pa['down_more_right'], pv['ccw_slow']])

        elif rule_number == 10:
            name = 'stop'
            value = self.min([pa['down_more_right'], pv['cw_slow']])

        elif rule_number == 11:
            name = 'left_fast'
            value = self.min([pa['down_more_left'], pv['cw_slow']])

        elif rule_number == 12:
            name = 'stop'
            value = self.min([pa['down_more_left'], pv['ccw_slow']])

        elif rule_number == 13:
            name = 'stop'
            value = self.min([pa['down_more_right'], pv['ccw_fast']])

        elif rule_number == 14:
            name = 'stop'
            value = self.min([pa['down_more_right'], pv['cw_fast']])

        elif rule_number == 15:
            name = 'stop'
            value = self.min([pa['down_more_left'], pv['cw_fast']])

        elif rule_number == 16:
            name = 'stop'
            value = self.min([pa['down_more_left'], pv['ccw_fast']])

        elif rule_number == 17:
            name = 'right_fast'
            value = self.min([pa['down_right'], pv['ccw_slow']])

        elif rule_number == 18:
            name = 'right_fast'
            value = self.min([pa['down_right'], pv['cw_slow']])

        elif rule_number == 19:
            name = 'left_fast'
            value = self.min([pa['down_left'], pv['cw_slow']])

        elif rule_number == 20:
            name = 'left_fast'
            value = self.min([pa['down_left'], pv['ccw_slow']])

        elif rule_number == 21:
            name = 'stop'
            value = self.min([pa['down_right'], pv['ccw_fast']])

        elif rule_number == 22:
            name = 'right_slow'
            value = self.min([pa['down_right'], pv['cw_fast']])

        elif rule_number == 23:
            name = 'stop'
            value = self.min([pa['down_left'], pv['cw_fast']])

        elif rule_number == 24:
            name = 'left_slow'
            value = self.min([pa['down_left'], pv['ccw_fast']])

        elif rule_number == 25:
            name = 'right_slow'
            value = self.min([pa['up_right'], pv['ccw_slow']])

        elif rule_number == 26:
            name = 'right_fast'
            value = self.min([pa['up_right'], pv['cw_slow']])

        elif rule_number == 27:
            name = 'right_fast'
            value = self.min([pa['up_right'], pv['stop']])

        elif rule_number == 28:
            name = 'left_slow'
            value = self.min([pa['up_left'], pv['cw_slow']])

        elif rule_number == 29:
            name = 'left_fast'
            value = self.min([pa['up_left'], pv['ccw_slow']])

        elif rule_number == 30:
            name = 'left_fast'
            value = self.min([pa['up_left'], pv['stop']])

        elif rule_number == 31:
            name = 'left_fast'
            value = self.min([pa['up_right'], pv['ccw_fast']])

        elif rule_number == 32:
            name = 'right_fast'
            value = self.min([pa['up_right'], pv['cw_fast']])

        elif rule_number == 33:
            name = 'right_fast'
            value = self.min([pa['up_left'], pv['cw_fast']])

        elif rule_number == 34:
            name = 'left_fast'
            value = self.min([pa['up_left'], pv['ccw_fast']])

        elif rule_number == 35:
            name = 'right_fast'
            value = self.min([pa['down'], pv['stop']])

        elif rule_number == 36:
            name = 'stop'
            value = self.min([pa['down'], pv['cw_fast']])

        elif rule_number == 37:
            name = 'stop'
            value = self.min([pa['down'], pv['ccw_fast']])

        elif rule_number == 38:
            name = 'left_slow'
            value = self.min([pa['up'], pv['ccw_slow']])

        elif rule_number == 39:
            name = 'left_fast'
            value = self.min([pa['up'], pv['ccw_fast']])

        elif rule_number == 40:
            name = 'right_slow'
            value = self.min([pa['up'], pv['cw_slow']])

        elif rule_number == 41:
            name = 'right_fast'
            value = self.min([pa['up'], pv['cw_fast']])

        elif rule_number == 42:
            name = 'stop'
            value = self.min([pa['up'], pv['stop']])

        else:
            raise ValueError('rule number not found')

        return name, value

    def inference(self, pa, pv, force):
        for rule_number in range(43):
            try:
                name, value = self.compute_rule(pa, pv, force, rule_number)
            except Exception as e:
                print(str(e))
                continue

            # if value != 0:
            #     print('force {}: {} -> {}'.format(name, force[name], self.max([force[name], value])))
            force[name] = self.max([force[name], value])

        return force



    ##! defuzzification

    def defuzzification(self, fuzzed):
        x_force = np.arange(-100, 100, 1)

        y_left_fast  = fuzz.trimf(x_force, [-100, -80, -60])
        y_left_slow  = fuzz.trimf(x_force, [-80, -60, 0])
        y_stop       = fuzz.trimf(x_force, [-60, 0, 60])
        y_right_slow = fuzz.trimf(x_force, [0, 60, 80])
        y_right_fast = fuzz.trimf(x_force, [60, 80, 100])

        left_fast  = fuzzed['left_fast']
        left_slow  = fuzzed['left_slow']
        stop       = fuzzed['stop']
        right_slow = fuzzed['right_slow']
        right_fast = fuzzed['right_fast']

        activ_left_fast  = np.fmin(left_fast, y_left_fast)
        activ_left_slow  = np.fmin(left_slow, y_left_slow)
        activ_stop       = np.fmin(stop, y_stop)
        activ_right_slow = np.fmin(right_slow, y_right_slow)
        activ_right_fast = np.fmin(right_fast, y_right_fast)

        aggregated = np.fmax(activ_left_fast,
                        np.fmax(activ_left_slow,
                        np.fmax(activ_stop,
                        np.fmax(activ_right_slow,
                                activ_right_fast))))

        force = self.last_force
        if left_fast + left_slow + stop + right_slow + right_fast != 0:
            force = fuzz.defuzz(x_force, aggregated, 'centroid')
            self.last_force = force

        return force



    def decide(self, world):
        input = self._make_input(world)
        output = self._make_output()

        ## fuzzification
        input = self.fuzzification(input)

        ## inference
        output = self.inference(
            input['pa'],
            input['pv'],
            output,
        )

        ## defuzzification
        force = self.defuzzification(output)

        print self.decide_counter,
        self.decide_counter += 1

        return force






