import os


def get_output_fn(args, end_pos):
    cot_string = 'T' if args.in_cot_way else 'F'
    qu_string = 'S' if args.with_short_question_marker else 'L'
    ans_string = 'S' if args.with_short_answer_marker else 'L'

    if args.sid == -7:
        way = 'MostSimilarInEachCluster'
    else:
        raise NotImplementedError

    if args.pick_demo_seed <= 0:
        way += 'SBERT'
        if 'qa' in args.clusters_filename:
            way += 'QA'

    method = f"model_{args.model_name}-source_{args.source}{args.diy_insert}-num_{args.num_sample}-way_{way}-inst_{args.instruction_way}-demo_{args.demo_way}-ifcot_{cot_string}-ifQ_{qu_string}-ifA_{ans_string}-res_{args.with_restrict}"

    print("Method is --> ", method)

    if not os.path.exists(args.output_files_folder):
        os.makedirs(args.output_files_folder)

    se = f"{args.start_pos}-{end_pos}.txt"
    a = os.path.join(args.output_files_folder, method)
    b = os.path.join(a, 'subs')
    c = os.path.join(b, se)
    if not os.path.exists(b):
        os.makedirs(b)
    return c


def full_decode(qid, realqid, question, handler, args, dataobj):
    rsa_type = args.rsa_type
    num_sample = args.num_sample
    source = args.source
    sid = args.sid  # only used in grouped
    seed = args.pick_demo_seed  # in randomly choose from flattened_gen_res
    with_restrict = args.with_restrict  # restrict added to prompt
    instruction_way = args.instruction_way
    demo_way = args.demo_way
    in_cot_way = args.in_cot_way
    with_short_question_marker = args.with_short_question_marker
    with_short_answer_marker = args.with_short_answer_marker

    debug = qid - args.start_pos < 2

    recall_select_answer_prefix = dataobj.recall_select_answer_template(num_sample=num_sample, type=rsa_type,
                                                                        source=source, sid=sid, qid=qid,
                                                                        realqid=realqid,
                                                                        seed=seed,
                                                                        with_restrict=with_restrict,
                                                                        instruction_way=instruction_way,
                                                                        demo_way=demo_way,
                                                                        in_cot_way=in_cot_way,
                                                                        with_short_question_marker=with_short_question_marker,
                                                                        with_short_answer_marker=with_short_answer_marker
                                                                        )

    test_prompt = dataobj.build_input(question=question,
                                      with_restrict=with_restrict,
                                      demo_way=demo_way,
                                      in_cot_way=in_cot_way,
                                      with_short_question_marker=with_short_question_marker,
                                      with_short_answer_marker=with_short_answer_marker)

    final_input = recall_select_answer_prefix + test_prompt

    if debug:
        print(f"======= Example {qid - args.start_pos} =======")
        print('>>>>> FIRST ITER INPUT >>>>>')
        print(final_input)

    max_tokens_mapping = {4: 128,}
    qstop = 'Q:' if with_short_question_marker else 'Question:'
    stop_mapping = {4: [qstop]}

    if args.model_name in ['instructgpt', 'codex']:
        # openai api
        output, probs_output = handler.get_output(final_input, max_tokens_mapping[demo_way],
                                                  stop=stop_mapping[demo_way],
                                                  do_tunc=True if not (with_short_answer_marker or with_short_question_marker) else False)
    elif args.model_name in ['gptneox-20B', 'alpaca-7B']:
        output = handler.get_output(final_input, max_tokens_mapping[demo_way], temperature=0.0, ban_pronoun=False, )
        output = output.strip()
        output = output.split('\n')[0]
        probs_output = None
    else:
        raise ValueError

    if debug:
        print('>>>>> FIRST ITER OUTPUT >>>>>')
        print(output)
        print("--------------------------")
    processed_pred_first_iter = dataobj.post_process(output, demo_way, in_cot_way, with_restrict,
                                                     probs_output=probs_output,
                                                     with_short_question_marker=with_short_question_marker,
                                                     with_short_answer_marker=with_short_answer_marker)

    # potential second process
    preocessed_pred = processed_pred_first_iter

    assert len(preocessed_pred) == 4
    if debug:
        print("=== 1 ===")
        print(preocessed_pred[0])
        print("=== 2 ===")
        print(preocessed_pred[1])
        print("=== 3 ===")
        print(preocessed_pred[2])

    return preocessed_pred
