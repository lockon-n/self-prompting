import json
import os
import jsonlines
from squad_evaluate import compute_metric
import random
from string import punctuation


class ODQATextData:
    def __init__(self, split, args, eval_only=False, traindata_obj=None):
        self.dir_path = args.dataset_dir
        self.split = split
        self.load()
        self.build_ref()
        if not eval_only:
            self.traindata_obj = traindata_obj
            flattened_gpt3_gen_filename = args.flattened_gen_data
            retrieve_filename = args.retrieve_filename
            clusters_filename = args.clusters_filename
            clusters_retrieve_filename = args.clusters_retrieve_filename
            fixed_sample_file = args.fixed_sample_file
            self.fixed_samples_func(flattened_gpt3_gen_filename, retrieve_filename, clusters_filename, clusters_retrieve_filename, fixed_sample_file)

    def load(self):
        filename = os.path.join(self.dir_path, self.split + '.jsonl')
        self.data = []
        with open(filename) as f:
            for item in jsonlines.Reader(f):
                self.data.append(item)

    def get_by_idx(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

    def build_ref(self):
        self.ref = []
        for i in range(len(self)):
            item = self.get_by_idx(i)
            self.ref.append({'answers': {'text': item['answer']}, 'id': i})

    def compute_metric(self, raw_pred, num=-1):
        # we receive a list, element i is the predicted answer for Q i
        # num means we only eval on first num samples
        preds = [{'prediction_text': text, 'id': i} for i, text in enumerate(raw_pred)]
        if num < 0:
            res, rw, all_em, all_f1 = compute_metric(preds, self.ref)
        else:
            res, rw, all_em, all_f1 = compute_metric(preds, self.ref[:num])
        return res, rw, all_em, all_f1

    def fixed_samples_func(self, flattened_gpt3_gen_filename, retrieve_filename, clusters_filename,
                           clusters_retrieve_filename, fixed_sample_file=None):
        if fixed_sample_file is not None:
            with open(fixed_sample_file) as f:
                self.fixed_samples = json.load(f)
        else:
            self.fixed_samples = []
        with open(flattened_gpt3_gen_filename) as f:
            self.flattened_fixed_samples_by_gpt3 = json.load(f)
        with open(clusters_filename) as f:
            self.clusters_res = json.load(f)
        with open(clusters_retrieve_filename) as f:
            self.clusters_retrieve_res = json.load(f)

        if retrieve_filename is not None:
            with open(retrieve_filename) as f:
                self.retrieve_res = json.load(f)
        else:
            assert self.clusters_retrieve_res is not None

    def recall_select_answer_template(self,
                                      num_sample,
                                      source='traindata',
                                      sid=-1,
                                      realqid=-1,
                                      seed=-1,
                                      with_restrict="inst",
                                      instruction_way=-1,
                                      demo_way=-1,
                                      in_cot_way=False,
                                      with_short_question_marker=False,
                                      with_short_answer_marker=False):
        """
        :param num_sample:
        :param type: 0 means p/e, 1 means p, 2 means e
        :return:
        """

        instruction_pool = {
            0: "Given a question, first write a short passage that can answer it, then choose the evidence sentence from the passage, and finally answer the question based on this sentence.",

            1: "Given a question, recite a short passage about it, then answer it.",
            2: "Given a question, recite a piece of evidence about it, then answer it.",

            10: "Given a question, recite a short passage about it, then write the question again and answer it.",
            20: "Given a question, recite a piece of evidence about it, then write the question again and answer it.",

            11: "Given a question, recite a short passage about it.",
            12: "Read a short passage and answer the question.",

            21: "Given a question, recite a piece of evidence about it.",
            22: "Read a piece of evidence and answer the question.",

            -1: "Answer the questions.",
            -2: "",
        }
        instruction = instruction_pool.get(instruction_way, "")

        inst_restrict = "Select and write exactly one entity when the answer has multiple ones (e.g. A if A, B, and C)" \
            if with_restrict in ['inst', 'both'] else ""

        tmp = (instruction + '\n' + inst_restrict).strip()
        output = tmp + '\n\n' if tmp != '' else tmp

        if source == 'gpt3gen':
            if seed > 0:
                random.seed(seed)
                used_fixed_samples = random.sample(self.flattened_fixed_samples_by_gpt3, num_sample)
            elif sid in [-7]:  # retrieve from each cluster
                retrieve_res_for_q = self.clusters_retrieve_res[realqid][str(num_sample)]
                assert len(retrieve_res_for_q) == num_sample
                most_similar_in_each_cluster = [it[0] for it in retrieve_res_for_q]
                most_similar_in_each_cluster_id = [iit[0] for iit in most_similar_in_each_cluster]
                used_fixed_samples = [self.flattened_fixed_samples_by_gpt3[idx] for idx in most_similar_in_each_cluster_id]
            else:
                raise ValueError
        else:
            raise ValueError

        answer_restrict = " (just one entity)" if with_restrict in ['ans', 'both'] else ""

        for i in range(min(num_sample, len(used_fixed_samples))):
            sample = used_fixed_samples[i]
            question_marker = "Q: " if with_short_question_marker else "Question: "
            question = (sample['question'] if sample['question'][-1] in punctuation else sample['question'] + '?').lower()
            answer = sample['answer']
            evi_only = sample['evidence_only']
            if demo_way == 4:
                # first answer, then explain
                demo = f"{question_marker}{question} \n\n The answer{answer_restrict} is {answer} because {evi_only}"
            else:
                raise NotImplementedError
            output += f'{demo}\n\n'

        return output

    def get_linetext(self, package, id, question):
        # package is a collection of resolved outputs,
        # answer is always the first,
        # if resolving failed, then the raw output is in the last item
        assert len(package) == 4
        probs_str = '|'.join([str(it) for it in package[3]])
        package_str = '\t'.join(package[:3]) + '\t' + probs_str
        line = f"{id}\t{question}\t{package_str}\n"
        return line

    def build_input(self, question, something_else="", with_restrict='inst', demo_way=-1, in_cot_way=False,
                    with_short_question_marker=False, with_short_answer_marker=False):

        answer_restrict = " (just one entity)" if with_restrict in ['ans', 'both'] else ""
        question_marker = "Q: " if with_short_question_marker else "Question: "

        question = question.lower()

        if demo_way in [4]:
            prompt = f"{question_marker}{question} \n\n The answer{answer_restrict} is"
        else:
            raise NotImplementedError
        return prompt

    def post_process(self, pred, demo_way, in_cot_way, with_restrict, with_short_question_marker=False,
                     with_short_answer_marker=False, probs_output=None):
        if demo_way in [4,]:
            splitter = 'because'
            if splitter in pred:
                ans_end = pred.find(splitter)
                evi_start = ans_end+len(splitter)
                gen_p = 0
                gen_l = 0
                ans_p = 0
                ans_l = 0
                if probs_output is None:
                    OP = (-1,-1,-1)
                else:
                    for (offset, token, prob) in zip(probs_output[0], probs_output[1], probs_output[2]):
                        if token == '\n': continue
                        span = (offset, offset + len(token))
                        if span[0] >= evi_start:
                            gen_p += prob
                            gen_l += 1
                        if span[1] <= ans_end:
                            ans_p += prob
                            ans_l += 1
                    ans_p = 0.0 if ans_l == 0 else ans_p / ans_l
                    gen_p = 0.0 if gen_l == 0 else gen_p / gen_l
                    OP = (ans_p, gen_p, -1)

                A = pred[:ans_end].strip().replace('\n',' ')
                B = pred[evi_start:].strip().replace('\n',' ')
                C = 'none'
            else:
                # no evidence given
                P = 0
                leng = 0
                if probs_output is None:
                    OP = (-1,-1,-1)
                else:
                    for (offset, token, prob) in zip(probs_output[0], probs_output[1], probs_output[2]):
                        if token != '\n':
                            leng += 1
                            P += prob
                    P = 0.0 if leng == 0 else P / leng
                    OP = (P, -1, -1)
                A, B, C = pred.strip().replace('\n', ' '), 'none', 'none'
        else:
            raise ValueError
        return A, B, C, OP

