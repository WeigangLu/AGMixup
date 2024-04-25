import numpy as np
import os

import torch
class Logger(object):
    def __init__(self, args):
        self.results = [0 for _ in range(args.runs)]      
        self.args = args
        self.folder_name = f"./results{args.exp_tag}/{args.setting}/{args.dataset}/{args.mix_method}"

    def add_result(self, run, result, res_type="acc"):
        assert 0 <= run < len(self.results)
        if res_type == "acc":
            self.results[run] = result

    def dump_parameters(self, content):
        if not os.path.exists(self.folder_name):
            os.makedirs(self.folder_name)

        filename = os.path.join(self.folder_name, self.args.model + ".txt")
        with open(filename, "a+") as f:
            f.write(content + "\t")
            f.write("\n")

    def print_statistics(self):
        acc_mean = np.mean(self.results)
        acc_std = np.std(self.results)

        content = f'Final Test: {acc_mean:.2f} ± {acc_std:.2f}'
        print(content)
        self.dump_parameters(content)

    def save(self, dst_foler, dst_file, obj):
        if not os.path.exists(dst_foler):
            os.makedirs(dst_foler)
        file_extension = os.path.splitext(dst_file)[1]
        if file_extension == ".npy":
            np.save(os.path.join(dst_foler, dst_file), obj)
        elif file_extension == ".pt":
            torch.save(obj, os.path.join(dst_foler, dst_file))