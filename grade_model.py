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


    def batch_predict(self, input_ind=[], input_grades=None):
        input = input_ind + [-1]
        if input_grades is None:
            input_grades = [DEFAULT_GRADE] * len(input_ind)
        input_grades.append(1.0)
        emb_ss = self.embds[input].reshape([len(input), self.embds.shape[-1]])
        emb_ss = np.transpose(np.multiply(np.transpose(emb_ss), np.array(input_grades)))
        emb_s = np.max(emb_ss, axis=0)
        emb_s_tile = np.tile(emb_s, [self.embds.shape[0], 1])
        diff = (self.embds - emb_s_tile).clip(min=0.0)
        grade = 1.0 / (1.0 + np.sum(diff, axis=-1))
        return grade[:-1]

    def top_courses(self, K=3, input_ind=[], input_grades=None, id_filter=None):
        if K >= self.embds.shape[0]:
            raise ValueError("top K > total courses")
        grades = self.batch_predict(input_ind=input_ind, input_grades=input_grades)
        ids = np.argsort(grades, axis=0)
        id_mask = []
        for i in range(ids.shape[0]):
            id = ids[i]
            if (id_filter is not None and id_filter(id)) or id_filter is None:
                if id in input_ind:
                    continue
                id_mask.append(i)
        id_mask = np.array(id_mask, dtype=int)
        ids = ids[id_mask][-min(id_mask.shape[0], K):]
        gs = grades[ids]
        return ids, gs

