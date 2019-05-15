# -*- coding: utf-8 -*-


from AoT import Component, Layout, Root, Structure
from constraints import (gen_entity_constraint, gen_layout_constraint,
                         rule_constraint)


def build_center_single():
    # Build AoT here
    root = Root("Scene")

    # Singleton struct
    struct = Structure("Singleton")

    # Singleton comp
    comp = Component("Grid")

    # Center_Single layout
    entity_constraint = gen_entity_constraint(type_min=1)
    layout_constraint = gen_layout_constraint("planar", 
                                              [(0.5, 0.5, 1, 1)], 
                                              num_min=0, 
                                              num_max=0)
    layout = Layout("Center_Single", layout_constraint, entity_constraint)
    comp.insert(layout)

    struct.insert(comp)
    root.insert(struct)

    return root


def build_distribute_four():
    # Build AoT here
    root = Root("Scene")

    # Singleton struct
    struct = Structure("Singleton")

    # Singleton comp
    comp = Component("Grid")

    # Distribute_Four
    entity_constraint = gen_entity_constraint(type_min=1)
    layout_constraint = gen_layout_constraint("planar",
                                              [(0.25, 0.25, 0.5, 0.5),
                                               (0.25, 0.75, 0.5, 0.5),
                                               (0.75, 0.25, 0.5, 0.5),
                                               (0.75, 0.75, 0.5, 0.5)],
                                              num_min=0,
                                              num_max=3)
    layout = Layout("Distribute_Four", layout_constraint, entity_constraint)
    comp.insert(layout)

    struct.insert(comp)
    root.insert(struct)

    return root


def build_distribute_nine():
    # Build AoT here
    root = Root("Scene")

    # Singleton struct
    struct = Structure("Singleton")

    # Singleton comp
    comp = Component("Grid")

    # Distribute_Nine
    entity_constraint = gen_entity_constraint(type_min=1)
    layout_constraint = gen_layout_constraint("planar", 
                                              [(0.16, 0.16, 0.33, 0.33),
                                               (0.16, 0.5, 0.33, 0.33),
                                               (0.16, 0.83, 0.33, 0.33),
                                               (0.5, 0.16, 0.33, 0.33),
                                               (0.5, 0.5, 0.33, 0.33),
                                               (0.5, 0.83, 0.33, 0.33),
                                               (0.83, 0.16, 0.33, 0.33),
                                               (0.83, 0.5, 0.33, 0.33),
                                               (0.83, 0.83, 0.33, 0.33)],
                                              num_min=0,
                                              num_max=8)
    layout = Layout("Distribute_Nine", layout_constraint, entity_constraint)
    comp.insert(layout)

    struct.insert(comp)
    root.insert(struct)
    
    return root


def build_left_center_single_right_center_single():
    # Build AoT here
    root = Root("Scene")

    # Left-Right Structure
    struct = Structure("Left_Right")

    # Left Component
    comp_left = Component("Left")

    # Left_Center_Single
    entity_constraint = gen_entity_constraint(type_min=1)
    layout_constraint = gen_layout_constraint("planar", 
                                              [(0.5, 0.25, 0.5, 0.5)], 
                                              num_min=0, 
                                              num_max=0)
    layout = Layout("Left_Center_Single", layout_constraint, entity_constraint)
    comp_left.insert(layout)

    # Right Component
    comp_right = Component("Right")

    # Right_Center_Single
    entity_constraint = gen_entity_constraint(type_min=1)
    layout_constraint = gen_layout_constraint("planar", 
                                              [(0.5, 0.75, 0.5, 0.5)], 
                                              num_min=0, 
                                              num_max=0)
    layout = Layout("Right_Center_Single", layout_constraint, entity_constraint)
    comp_right.insert(layout)

    struct.insert(comp_left)
    struct.insert(comp_right)
    root.insert(struct)
    
    return root


def build_up_center_single_down_center_single():
    # Build AoT here
    root = Root("Scene")

    # Up-Down Structure
    struct = Structure("Up_Down")

    # Left Component
    comp_up = Component("Up")

    # Up_Center_Single
    entity_constraint = gen_entity_constraint(type_min=1)
    layout_constraint = gen_layout_constraint("planar", 
                                              [(0.25, 0.5, 0.5, 0.5)], 
                                              num_min=0, 
                                              num_max=0)
    layout = Layout("Up_Center_Single", layout_constraint, entity_constraint)
    comp_up.insert(layout)

    # Down Component
    comp_down = Component("Down")

    # Down_Center_Single
    entity_constraint = gen_entity_constraint(type_min=1)
    layout_constraint = gen_layout_constraint("planar", 
                                              [(0.75, 0.5, 0.5, 0.5)], 
                                              num_min=0, 
                                              num_max=0)
    layout = Layout("Down_Center_Single", layout_constraint, entity_constraint)
    comp_down.insert(layout)

    struct.insert(comp_up)
    struct.insert(comp_down)
    root.insert(struct)
    
    return root


def build_in_center_single_out_center_single():
    # Build AoT here
    root = Root("Scene")

    # In-Out Structure
    struct = Structure("Out_In")

    # Out Component 
    comp_out = Component("Out")

    # Out_One
    entity_constraint = gen_entity_constraint(type_min=1, 
                                              size_min=3,
                                              color_max=0)
    layout_constraint = gen_layout_constraint("planar",
                                              [(0.5, 0.5, 1, 1)],
                                              num_min=0,
                                              num_max=0)
    layout = Layout("Out_Center_Single", layout_constraint, entity_constraint)
    comp_out.insert(layout)

    # In Component
    comp_in = Component("In")

    # In_Center_Single
    entity_constraint = gen_entity_constraint(type_min=1)
    layout_constraint = gen_layout_constraint("planar",
                                              [(0.5, 0.5, 0.33, 0.33)],
                                              num_min=0,
                                              num_max=0)
    layout = Layout("In_Center_Single", layout_constraint, entity_constraint)
    comp_in.insert(layout)

    struct.insert(comp_out)
    struct.insert(comp_in)
    root.insert(struct)

    return root


def build_in_distribute_four_out_center_single():
    # Build AoT here
    root = Root("Scene")

    # In-Out Structure
    struct = Structure("Out_In")

    # Out Component 
    comp_out = Component("Out")

    # Out_One
    entity_constraint = gen_entity_constraint(type_min=1, 
                                              size_min=3,
                                              color_max=0)
    layout_constraint = gen_layout_constraint("planar",
                                              [(0.5, 0.5, 1, 1)],
                                              num_min=0,
                                              num_max=0)
    layout = Layout("Out_Center_Single", layout_constraint, entity_constraint)
    comp_out.insert(layout)

    # In Component
    comp_in = Component("In")

    # In_Four
    entity_constraint = gen_entity_constraint(type_min=1, size_min=2)
    layout_constraint = gen_layout_constraint("planar",
                                              [(0.42, 0.42, 0.15, 0.15),
                                               (0.42, 0.58, 0.15, 0.15),
                                               (0.58, 0.42, 0.15, 0.15),
                                               (0.58, 0.58, 0.15, 0.15)],
                                              num_min=0,
                                              num_max=3)
    layout = Layout("In_Distribute_Four", layout_constraint, entity_constraint)
    comp_in.insert(layout)

    struct.insert(comp_out)
    struct.insert(comp_in)
    root.insert(struct)

    return root