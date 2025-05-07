# -*- coding: utf-8 -*-


import argparse
import copy
import os
import random
import sys

import numpy as np
from tqdm import trange

from build_tree import (build_center_single, build_distribute_four,
                        build_distribute_nine,
                        build_in_center_single_out_center_single,
                        build_in_distribute_four_out_center_single,
                        build_left_center_single_right_center_single,
                        build_up_center_single_down_center_single)
from const import IMAGE_SIZE, RULE_ATTR
from rendering import (generate_matrix, generate_matrix_answer, imsave, imshow,
                       render_panel)
from Rule import Rule_Wrapper
from sampling import sample_attr, sample_attr_avail, sample_rules
from serialize import dom_problem, serialize_aot, serialize_rules
from solver import solve


def merge_component(dst_aot, src_aot, component_idx):
    src_component = src_aot.children[0].children[component_idx]
    dst_aot.children[0].children[component_idx] = src_component


def fuse(args, all_configs):
    random.seed(args.seed)
    np.random.seed(args.seed)

    acc = 0
    for k in trange(args.num_samples * len(all_configs)):
        if k < args.num_samples * (1 - args.val - args.test):
            set_name = "train"
        elif k < args.num_samples * (1 - args.test):
            set_name = "val"
        else:
            set_name = "test"
        
        tree_name = random.choice(all_configs.keys())
        root = all_configs[tree_name]
        while True:
            rule_groups = sample_rules()
            new_root = root.prune(rule_groups)    
            if new_root is not None:
                break
        
        start_node = new_root.sample()

        row_1_1 = copy.deepcopy(start_node)
        for l in range(len(rule_groups)):
            rule_group = rule_groups[l]
            rule_num_pos = rule_group[0]
            row_1_2 = rule_num_pos.apply_rule(row_1_1)
            row_1_3 = rule_num_pos.apply_rule(row_1_2)
            for i in range(1, len(rule_group)):
                rule = rule_group[i]
                row_1_2 = rule.apply_rule(row_1_1, row_1_2)
            for i in range(1, len(rule_group)):
                rule = rule_group[i]
                row_1_3 = rule.apply_rule(row_1_2, row_1_3)
            if l == 0:
                to_merge = [row_1_1, row_1_2, row_1_3]
            else:
                merge_component(to_merge[1], row_1_2, l)
                merge_component(to_merge[2], row_1_3, l)
        row_1_1, row_1_2, row_1_3 = to_merge

        row_2_1 = copy.deepcopy(start_node)
        row_2_1.resample(True)
        for l in range(len(rule_groups)):
            rule_group = rule_groups[l]
            rule_num_pos = rule_group[0]
            row_2_2 = rule_num_pos.apply_rule(row_2_1)
            row_2_3 = rule_num_pos.apply_rule(row_2_2)
            for i in range(1, len(rule_group)):
                rule = rule_group[i]
                row_2_2 = rule.apply_rule(row_2_1, row_2_2)
            for i in range(1, len(rule_group)):
                rule = rule_group[i]
                row_2_3 = rule.apply_rule(row_2_2, row_2_3)
            if l == 0:
                to_merge = [row_2_1, row_2_2, row_2_3]
            else:
                merge_component(to_merge[1], row_2_2, l)
                merge_component(to_merge[2], row_2_3, l)
        row_2_1, row_2_2, row_2_3 = to_merge

        row_3_1 = copy.deepcopy(start_node)
        row_3_1.resample(True)
        for l in range(len(rule_groups)):
            rule_group = rule_groups[l]
            rule_num_pos = rule_group[0]
            row_3_2 = rule_num_pos.apply_rule(row_3_1)
            row_3_3 = rule_num_pos.apply_rule(row_3_2)
            for i in range(1, len(rule_group)):
                rule = rule_group[i]
                row_3_2 = rule.apply_rule(row_3_1, row_3_2)
            for i in range(1, len(rule_group)):
                rule = rule_group[i]
                row_3_3 = rule.apply_rule(row_3_2, row_3_3)
            if l == 0:
                to_merge = [row_3_1, row_3_2, row_3_3]
            else:
                merge_component(to_merge[1], row_3_2, l)
                merge_component(to_merge[2], row_3_3, l)
        row_3_1, row_3_2, row_3_3 = to_merge

        imgs = [render_panel(row_1_1),
                render_panel(row_1_2),
                render_panel(row_1_3),
                render_panel(row_2_1),
                render_panel(row_2_2),
                render_panel(row_2_3),
                render_panel(row_3_1),
                render_panel(row_3_2),
                np.zeros((IMAGE_SIZE, IMAGE_SIZE), np.uint8)]
        context = [row_1_1, row_1_2, row_1_3, row_2_1, row_2_2, row_2_3, row_3_1, row_3_2]

        modifiable_attr = sample_attr_avail(rule_groups, row_3_3)
        answer_AoT = copy.deepcopy(row_3_3)
        candidates = [answer_AoT]
        for j in range(7):
            component_idx, attr_name, min_level, max_level = sample_attr(modifiable_attr)
            answer_j = copy.deepcopy(answer_AoT)
            answer_j.sample_new(component_idx, attr_name, min_level, max_level, answer_AoT)
            candidates.append(answer_j)

        random.shuffle(candidates)
        answers = []
        for candidate in candidates:
            answers.append(render_panel(candidate))
        # imsave(generate_matrix_answer(imgs + answers), "./experiments/fuse/{}.jpg".format(k))

        image = imgs[0:8] + answers
        target = candidates.index(answer_AoT)
        predicted = solve(rule_groups, context, candidates)
        meta_matrix, meta_target = serialize_rules(rule_groups)
        structure, meta_structure = serialize_aot(start_node)
        np.savez("{}/RAVEN_{}_{}.npz".format(args.save_dir, k, set_name), image=image, 
                                                                          target=target,
                                                                          predict=predicted, 
                                                                          meta_matrix=meta_matrix,
                                                                          meta_target=meta_target, 
                                                                          structure=structure,
                                                                          meta_structure=meta_structure)
        with open("{}/RAVEN_{}_{}.xml".format(args.save_dir, k, set_name), "w") as f:
            dom = dom_problem(context + candidates, rule_groups)
            f.write(dom)
        if target == predicted:
            acc += 1
    print("Accuracy: {}".format(float(acc) / (args.num_samples * len(all_configs))))


def separate(args, all_configs):
    random.seed(args.seed)
    np.random.seed(args.seed)

    for key in all_configs.keys():
        acc = 0
        for k in trange(args.num_samples):
            count_num = k % 10
            if count_num < (10 - args.val - args.test):
                set_name = "train"
            elif count_num < (10 - args.test):
                set_name = "val"
            else:
                set_name = "test"

            root = all_configs[key]
            while True:
                rule_groups = sample_rules()
                new_root = root.prune(rule_groups)    
                if new_root is not None:
                    break
            
            start_node = new_root.sample()

            row_1_1 = copy.deepcopy(start_node)
            for l in range(len(rule_groups)):
                rule_group = rule_groups[l]
                rule_num_pos = rule_group[0]
                row_1_2 = rule_num_pos.apply_rule(row_1_1)
                row_1_3 = rule_num_pos.apply_rule(row_1_2)
                for i in range(1, len(rule_group)):
                    rule = rule_group[i]
                    row_1_2 = rule.apply_rule(row_1_1, row_1_2)
                for i in range(1, len(rule_group)):
                    rule = rule_group[i]
                    row_1_3 = rule.apply_rule(row_1_2, row_1_3)
                if l == 0:
                    to_merge = [row_1_1, row_1_2, row_1_3]
                else:
                    merge_component(to_merge[1], row_1_2, l)
                    merge_component(to_merge[2], row_1_3, l)
            row_1_1, row_1_2, row_1_3 = to_merge

            row_2_1 = copy.deepcopy(start_node)
            row_2_1.resample(True)
            for l in range(len(rule_groups)):
                rule_group = rule_groups[l]
                rule_num_pos = rule_group[0]
                row_2_2 = rule_num_pos.apply_rule(row_2_1)
                row_2_3 = rule_num_pos.apply_rule(row_2_2)
                for i in range(1, len(rule_group)):
                    rule = rule_group[i]
                    row_2_2 = rule.apply_rule(row_2_1, row_2_2)
                for i in range(1, len(rule_group)):
                    rule = rule_group[i]
                    row_2_3 = rule.apply_rule(row_2_2, row_2_3)
                if l == 0:
                    to_merge = [row_2_1, row_2_2, row_2_3]
                else:
                    merge_component(to_merge[1], row_2_2, l)
                    merge_component(to_merge[2], row_2_3, l)
            row_2_1, row_2_2, row_2_3 = to_merge

            row_3_1 = copy.deepcopy(start_node)
            row_3_1.resample(True)
            for l in range(len(rule_groups)):
                rule_group = rule_groups[l]
                rule_num_pos = rule_group[0]
                row_3_2 = rule_num_pos.apply_rule(row_3_1)
                row_3_3 = rule_num_pos.apply_rule(row_3_2)
                for i in range(1, len(rule_group)):
                    rule = rule_group[i]
                    row_3_2 = rule.apply_rule(row_3_1, row_3_2)
                for i in range(1, len(rule_group)):
                    rule = rule_group[i]
                    row_3_3 = rule.apply_rule(row_3_2, row_3_3)
                if l == 0:
                    to_merge = [row_3_1, row_3_2, row_3_3]
                else:
                    merge_component(to_merge[1], row_3_2, l)
                    merge_component(to_merge[2], row_3_3, l)
            row_3_1, row_3_2, row_3_3 = to_merge

            imgs = [render_panel(row_1_1),
                    render_panel(row_1_2),
                    render_panel(row_1_3),
                    render_panel(row_2_1),
                    render_panel(row_2_2),
                    render_panel(row_2_3),
                    render_panel(row_3_1),
                    render_panel(row_3_2),
                    np.zeros((IMAGE_SIZE, IMAGE_SIZE), np.uint8)]
            context = [row_1_1, row_1_2, row_1_3, row_2_1, row_2_2, row_2_3, row_3_1, row_3_2]
            modifiable_attr = sample_attr_avail(rule_groups, row_3_3)
            answer_AoT = copy.deepcopy(row_3_3)
            candidates = [answer_AoT]
            for j in range(7):
                component_idx, attr_name, min_level, max_level = sample_attr(modifiable_attr)
                answer_j = copy.deepcopy(answer_AoT)
                answer_j.sample_new(component_idx, attr_name, min_level, max_level, answer_AoT)
                candidates.append(answer_j)

            random.shuffle(candidates)
            answers = []
            for candidate in candidates:
                answers.append(render_panel(candidate))
            # imsave(generate_matrix_answer(imgs + answers), "./experiments/{}/{}.jpg".format(key, k))    
            
            image = imgs[0:8] + answers
            target = candidates.index(answer_AoT)
            predicted = solve(rule_groups, context, candidates)
            meta_matrix, meta_target = serialize_rules(rule_groups)
            structure, meta_structure = serialize_aot(start_node)
            np.savez("{}/{}/RAVEN_{}_{}.npz".format(args.save_dir, key, k, set_name), image=image, 
                                                                                      target=target, 
                                                                                      predict=predicted,
                                                                                      meta_matrix=meta_matrix,
                                                                                      meta_target=meta_target, 
                                                                                      structure=structure,
                                                                                      meta_structure=meta_structure)
            with open("{}/{}/RAVEN_{}_{}.xml".format(args.save_dir, key, k, set_name), "w") as f:
                dom = dom_problem(context + candidates, rule_groups)
                f.write(dom.decode("utf-8"))
            
            if target == predicted:
                acc += 1
        print ("Accuracy of {}: {}".format(key, float(acc) / args.num_samples))


def main():
    main_arg_parser = argparse.ArgumentParser(description="parser for RAVEN")
    main_arg_parser.add_argument("--num-samples", type=int, default=20000,
                                 help="number of samples for each component configuration")
    main_arg_parser.add_argument("--save-dir", type=str, default="~/Datasets/",
                                 help="path to folder where the generated dataset will be saved.")
    main_arg_parser.add_argument("--seed", type=int, default=1234,
                                 help="random seed for dataset generation")
    main_arg_parser.add_argument("--fuse", type=int, default=0,
                                 help="whether to fuse different configurations")
    main_arg_parser.add_argument("--val", type=float, default=2,
                                 help="the proportion of the size of validation set")
    main_arg_parser.add_argument("--test", type=float, default=2,
                                 help="the proportion of the size of test set")                             
    args = main_arg_parser.parse_args()

    all_configs = {"center_single": build_center_single(),
                   "distribute_four": build_distribute_four(),
                   "distribute_nine": build_distribute_nine(),
                   "left_center_single_right_center_single": build_left_center_single_right_center_single(),
                   "up_center_single_down_center_single": build_up_center_single_down_center_single(),
                   "in_center_single_out_center_single": build_in_center_single_out_center_single(),
                   "in_distribute_four_out_center_single": build_in_distribute_four_out_center_single()}

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    if args.fuse:
        if not os.path.exists(os.path.join(args.save_dir, "fuse")):
            os.mkdir(os.path.join(args.save_dir, "fuse"))
        fuse(args, all_configs)
    else:
        for key in all_configs.keys():
            if not os.path.exists(os.path.join(args.save_dir, key)):
                os.mkdir(os.path.join(args.save_dir, key))
        separate(args, all_configs)
    

if __name__ == "__main__":
    main()
