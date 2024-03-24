from data_utils import ODQATextData
from api_utils import api_handler
from string import punctuation
import argparse
import tqdm

from general_utils import get_output_fn, full_decode
from local_model_new import LocalGenModel

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--api_file', default='./related_files/openai-api.txt')
    parser.add_argument('--model_name', default='instructgpt')
    parser.add_argument('--dataset_name', default='samples_nq')
    parser.add_argument('--dataset_dir', default='./datasets/samples_nq')
    parser.add_argument('--start_pos', type=int, default=0)
    parser.add_argument('--end_pos', type=int, default=10)
    parser.add_argument('--output_files_folder', default='./outputs/samples_nq')

    ### args for method
    parser.add_argument('--rsa_type', type=int, default=-1)

    parser.add_argument('--num_sample', type=int, default=10)
    parser.add_argument('--source', default='gpt3gen')
    parser.add_argument('--sid', type=int, default=-7)
    parser.add_argument('--pick_demo_seed', type=int, default=-1)

    parser.add_argument('--with_restrict', default='ans')
    parser.add_argument('--instruction_way', type=int, default=-2)
    parser.add_argument('--demo_way', type=int, default=4)
    parser.add_argument('--with_short_question_marker',default=False,action='store_true')
    parser.add_argument('--with_short_answer_marker',default=False,action='store_true')
    parser.add_argument('--in_cot_way',default=False,action='store_true')

    ### args for needed files
    # parser.add_argument('--grouped_gen_data', default=None)
    parser.add_argument('--flattened_gen_data', default=None)
    parser.add_argument('--retrieve_filename', default=None)
    parser.add_argument('--clusters_filename', default=None)
    parser.add_argument('--clusters_retrieve_filename', default=None)
    parser.add_argument('--fixed_sample_file', default=None)

    parser.add_argument('--diy_insert',default='')
    args = parser.parse_args()

    ### get handler
    if args.model_name in ['instructgpt','codex']:
        handler = api_handler(args.model_name, args.api_file)
    elif args.model_name in ['gptneox-20B','alpaca-7B']:
        handler = LocalGenModel(args.model_name)
    else:
        raise ValueError

    ### get dataobj
    traindata_obj = ODQATextData('train', args, eval_only=True)
    dataobj = ODQATextData('test', args,traindata_obj=traindata_obj.data)

    ### set test range
    end_pos = len(dataobj) if args.end_pos == -1 else args.end_pos
    test_range = range(args.start_pos, end_pos)  # closed interval

    ### get output_file_name
    exact_output_file = get_output_fn(args, end_pos)

    for idx in tqdm.tqdm(test_range, desc=f"{args.start_pos} ~ {end_pos}"):
        raw_sample = dataobj.get_by_idx(idx)
        question = raw_sample['question'] if raw_sample['question'][-1] in punctuation else raw_sample['question'] + '?'

        realqid = idx

        processed_pred = full_decode(idx, realqid, question, handler, args, dataobj)

        linetext = dataobj.get_linetext(package=processed_pred, id=idx, question=question)

        with open(exact_output_file, 'a') as f:
            f.write(linetext)
