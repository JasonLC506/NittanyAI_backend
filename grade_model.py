import numpy as np


DEFAULT_GRADE = 0.6
class GradeModel(object):
    def __init__(self, embds):
        self.embds = embds

    def predict(self, target_ind, input_ind=[], input_grades=None):
        emb_t = self.embds[target_ind]
        input = input_ind + [-1]
        if input_grades is None:
            input_grades = [DEFAULT_GRADE] * len(input_ind)
        input_grades.append(1.0)
        emb_ss = self.embds[input].reshape([len(input), self.embds.shape[-1]])
        emb_ss = np.transpose(np.multiply(np.transpose(emb_ss), np.array(input_grades)))
        emb_s = np.max(emb_ss, axis=0)
        diff = (emb_t - emb_s).clip(min=0.0)
        grade = 1.0 / (1.0 + np.sum(diff, axis=-1))
        return grade

