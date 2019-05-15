# -*- coding: utf-8 -*-


import numpy as np


def solve(rule_groups, context, candidates):
    """Search-based Heuristic Solver.
    Arguments:
        rule_groups(list of list of Rule): rules that apply to each component
        context(list of AoTNode): a list of context AoTs in a row-major order;
            should be of length 8
        candidates(list of AoTNode): a list of candidate answer AoTs;
            should be of length 8
    Returns:
        ans(int): index of the correct answer in the candidates
    """
    satisfied = [0] * len(candidates)
    for i in range(len(candidates)):
        candidate = candidates[i]
        # note that rule.component_idx should be the same as j
        for j in range(len(rule_groups)):
            rule_group = rule_groups[j]
            rule_num_pos = rule_group[0]
            satisfied[i] += check_num_pos(rule_num_pos, context, candidate)
            regenerate = False
            if rule_num_pos.attr == "Number" or rule_num_pos.name == "Arithmetic":
                regenerate = True
            rule_type = rule_group[1]
            satisfied[i] += check_entity(rule_type, context, candidate, "Type", regenerate)
            rule_size = rule_group[2]
            satisfied[i] += check_entity(rule_size, context, candidate, "Size", regenerate)
            rule_color = rule_group[3]
            satisfied[i] += check_entity(rule_color, context, candidate, "Color", regenerate)
    satisfied = np.array(satisfied)
    answer_set = np.where(satisfied == max(satisfied))[0]
    return np.random.choice(answer_set)


def check_num_pos(rule_num_pos, context, candidate):
    """Check whether Rule on layout attribute is satisfied.
    Arguments:
        rule_num_pos(Rule): the rule to check
        context(list of AoTNode): the 8 context figures 
        candidate(AoTNode): the candidate AoT
    Returns:
        ret(int): 0 if failure, 1 if success
    """
    ret = 0
    component_idx = rule_num_pos.component_idx
    row_3_1_layout = context[6].children[0].children[component_idx].children[0]
    row_3_2_layout = context[7].children[0].children[component_idx].children[0]
    candidate_layout = candidate.children[0].children[component_idx].children[0]
    if rule_num_pos.name == "Constant":
        set_row_3_1_pos = set(row_3_1_layout.position.get_value_idx())
        set_row_3_2_pos = set(row_3_2_layout.position.get_value_idx())
        set_candidate_pos = set(candidate_layout.position.get_value_idx())
        # note that set equal only when len(Number) equal and content equal
        if set_candidate_pos == set_row_3_1_pos and set_candidate_pos == set_row_3_2_pos:
            ret = 1
    elif rule_num_pos.name == "Progression":
        if rule_num_pos.attr == "Number":
            row_3_1_num = row_3_1_layout.number.get_value_level()
            row_3_2_num = row_3_2_layout.number.get_value_level()
            candidate_num = candidate_layout.number.get_value_level()
            if row_3_2_num * 2 == row_3_1_num + candidate_num:
                ret = 1
        else:
            row_3_1_pos = row_3_1_layout.position.get_value_idx()
            row_3_2_pos = row_3_2_layout.position.get_value_idx()
            candidate_pos = candidate_layout.position.get_value_idx()
            most_num = len(candidate_layout.position.values)
            diff = rule_num_pos.value
            if (set((row_3_1_pos + diff) % most_num) == set(row_3_2_pos)) and \
               (set((row_3_2_pos + diff) % most_num) == set(candidate_pos)):
               ret = 1
    elif rule_num_pos.name == "Arithmetic":
        mode = rule_num_pos.value
        if rule_num_pos.attr == "Number":
            row_3_1_num = row_3_1_layout.number.get_value()
            row_3_2_num = row_3_2_layout.number.get_value()
            candidate_num = candidate_layout.number.get_value()
            if mode > 0 and (candidate_num == row_3_1_num + row_3_2_num):
                ret = 1
            if mode < 0 and (candidate_num == row_3_1_num - row_3_2_num):
                ret = 1
        else:
            row_3_1_pos = row_3_1_layout.position.get_value_idx()
            row_3_2_pos = row_3_2_layout.position.get_value_idx()
            candidate_pos = candidate_layout.position.get_value_idx()
            if mode > 0 and (set(candidate_pos) == set(row_3_1_pos) | set(row_3_2_pos)):
                ret = 1
            if mode < 0 and (set(candidate_pos) == set(row_3_1_pos) - set(row_3_2_pos)):
                ret = 1
    else:
        three_values = rule_num_pos.value_levels[2]
        if rule_num_pos.attr == "Number":
            row_3_1_num = row_3_1_layout.number.get_value_level()
            row_3_2_num = row_3_2_layout.number.get_value_level()
            candidate_num = candidate_layout.number.get_value_level()
            if row_3_1_num == three_values[0] and \
               row_3_2_num == three_values[1] and \
               candidate_num == three_values[2]:
                ret = 1
        else:
            row_3_1_pos = row_3_1_layout.position.get_value_idx()
            row_3_2_pos = row_3_2_layout.position.get_value_idx()
            candidate_pos = candidate_layout.position.get_value_idx()
            if set(row_3_1_pos) == set(three_values[0]) and \
               set(row_3_2_pos) == set(three_values[1]) and \
               set(candidate_pos) == set(three_values[2]):
                ret = 1
    return ret


def check_consistency(candidate, attr, component_idx):
    candidate_layout = candidate.children[0].children[component_idx].children[0]
    entity_0 = candidate_layout.children[0]
    attr_name = attr.lower()
    entity_0_value = getattr(entity_0, attr_name).get_value_level()
    for i in range(1, len(candidate_layout.children)):
        entity_i = candidate_layout.children[i]
        entity_i_value = getattr(entity_i, attr_name).get_value_level()
        if entity_i_value != entity_0_value:
            return False
    return True


def check_entity(rule, context, candidate, attr, regenerate):
    """Check whether Rule on entity attribute is satisfied.
    Arguments:
        rule(Rule): the rule to check
        context(list of AoTNode): the 8 context figures 
        candidate(AoTNode): the candidate AoT
        attr(str): attribute name
    Returns:
        ret(int): 0 if failure, 1 if success
    """
    ret = 0
    component_idx = rule.component_idx
    row_3_1_layout = context[6].children[0].children[component_idx].children[0]
    row_3_2_layout = context[7].children[0].children[component_idx].children[0]
    candidate_layout = candidate.children[0].children[component_idx].children[0]
    uni = candidate_layout.uniformity.get_value()
    attr_name = attr.lower()
    if rule.name == "Constant":
        if uni:
            if check_consistency(candidate, attr, component_idx):
                if getattr(candidate_layout.children[0], attr_name).get_value_level() == \
                   getattr(row_3_2_layout.children[0], attr_name).get_value_level():
                    ret = 1
        else:
            row_3_1_num = row_3_1_layout.number.get_value_level()
            row_3_2_num = row_3_2_layout.number.get_value_level()
            candidate_num = candidate_layout.number.get_value_level()
            if (row_3_1_num == row_3_2_num) and (row_3_2_num == candidate_num):
                if regenerate:
                    ret = 1
                else:
                    flag = True
                    for i in range(len(candidate_layout.children)):
                        if not (getattr(candidate_layout.children[i], attr_name).get_value_level() == 
                                getattr(row_3_2_layout.children[i], attr_name).get_value_level()):
                            flag = False
                            break
                    if flag:
                        ret = 1
            else:
                ret = 1
    elif rule.name == "Progression":
        if check_consistency(candidate, attr, component_idx):
            row_3_1_value = getattr(row_3_1_layout.children[0], attr_name).get_value_level()
            row_3_2_value = getattr(row_3_2_layout.children[0], attr_name).get_value_level()
            candidate_value = getattr(candidate_layout.children[0], attr_name).get_value_level()
            if row_3_2_value * 2 == row_3_1_value + candidate_value:
                ret = 1
    elif rule.name == "Arithmetic":
        if check_consistency(candidate, attr, component_idx):
            row_3_1_value = getattr(row_3_1_layout.children[0], attr_name).get_value_level()
            row_3_2_value = getattr(row_3_2_layout.children[0], attr_name).get_value_level()
            candidate_value = getattr(candidate_layout.children[0], attr_name).get_value_level()
            if rule.value > 0:
                if attr == "Color":
                    if candidate_value == row_3_1_value + row_3_2_value:
                        ret = 1
                else:
                    if candidate_value == row_3_1_value + row_3_2_value + 1:
                        ret = 1
            if rule.value < 0:
                if attr == "Color":
                    if candidate_value == row_3_1_value - row_3_2_value:
                        ret = 1
                else:
                    if candidate_value == row_3_1_value - row_3_2_value - 1:
                        ret = 1                
    else:
        if check_consistency(candidate, attr, component_idx):
            row_3_1_value = getattr(row_3_1_layout.children[0], attr_name).get_value_level()
            row_3_2_value = getattr(row_3_2_layout.children[0], attr_name).get_value_level()
            candidate_value = getattr(candidate_layout.children[0], attr_name).get_value_level()
            three_values = rule.value_levels[2]
            if row_3_1_value == three_values[0] and \
               row_3_2_value == three_values[1] and \
               candidate_value == three_values[2]:
               ret = 1
    return ret
