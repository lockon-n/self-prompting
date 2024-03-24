import json
import os.path
import random
import nltk
import numpy as np
from api_utils import api_handler
from string import punctuation, whitespace, ascii_lowercase
from extract_utils import process_an_item
from squad_evaluate import normalize_answer
import spacy
from pprint import pprint
import time
from local_model_new import LocalGenModel

expanded_punction_and_whitespace = punctuation + '–' + whitespace


def generate_a_detailed_ODQA(handler, topic=None, input_question=None, NE="llm"):
    gen_passage_sent = 'This is a short passage from Wikipedia about'
    temp = 1.0

    if topic is not None:
        gen_passage_sent = f"{gen_passage_sent} {topic}:"
        temp = 0.0
    elif input_question is not None:
        gen_passage_sent = f"Generate a background document from Wikipedia to answer the given question. {input_question}"
        temp = 0.0
    else:
        raise ValueError
    # some codes to generate passage with the above prompt

    raw_passage_by_gpt3 = handler.get_output(input_text=gen_passage_sent, max_tokens=128, temperature=temp, suffix=None,
                                             stop=None, do_tunc=False, echo=True,return_prob=False)
    # we need to do some process

    print(raw_passage_by_gpt3)

    passage_by_gpt3 = None
    lines = raw_passage_by_gpt3.split('\n')
    for line in lines:
        if line.startswith(gen_passage_sent) or line.strip() == '' or len(line) < 100:
            continue
        else:
            passage_by_gpt3 = line.strip()
            break

    sents = nltk.sent_tokenize(passage_by_gpt3)
    if not sents[-1].endswith('.'):  # abandon the last, incomplete sentence
        sents = sents[:-1]
    final_passage = " ".join(sents)

    # print("Passage")
    # print(final_passage)

    print(final_passage)

    len_sent = [len(sent) for sent in sents]

    pick_sent_order = np.argsort(len_sent)
    # random.shuffle(pick_sent_order)

    able_to_find = False
    qass = []
    evidences = []
    decontextualized_evidences = []

    for sent_id in pick_sent_order:
        evidence = sents[sent_id].strip()

        if NE == 'llm':
            extract_entity_prompt = f'Here is a sentence: {evidence}\n\n' \
                                    f'Extract the named entities (like date, location, organization, character, number) in it, and separate them by \'|\'.' \
                                    f'If no named entity in it, write \'None\' only.'
            raw_entities_by_gpt3 = handler.get_output(input_text=extract_entity_prompt, max_tokens=50, temperature=0.0,
                                                      suffix=None, stop=None, do_tunc=False, echo=False,return_prob=False)
            len_space_split_ents = len(evidence.split(' '))
            raw_entities = [s.strip() for s in raw_entities_by_gpt3.strip().split('|')]

            print(evidence)
            print(raw_entities)

            if len(raw_entities) > int(0.8 * len_space_split_ents):
                continue

            entities = [s for s in raw_entities if s.strip() != '' and len(s.split(' ')) <= 5 and s != 'None']
        elif NE == 'nltk':
            entities = process_an_item(evidence, True)
        else:
            raise NotImplementedError

        if len(entities) == 0:
            continue
        # now we have evidence, passage, entities

        decontextualized_evidence_prompt = "Given a passage and a sentence extracted from it, " \
                                           "rewrite the sentence to make its meaning clear enough without a context," \
                                           "the output should contain only one sentence.\n\n" \
                                           f"{final_passage}\n" \
                                           f"Sentence: {evidence}\n" \
                                           "Rewrite:"

        # decontextualize the evidence sentence with the above prompt

        decontextualized_evidence = handler.get_output(input_text=decontextualized_evidence_prompt, max_tokens=128,
                                                       temperature=0.0, suffix=None, stop=['\n'], do_tunc=True,
                                                       echo=False,return_prob=False)
        # print(f"Evidence {sent_id}, sub {subid}")
        # print(evidence)
        # print('D : ', decontextualized_evidence)
        decontextualized_evidences_sent = nltk.sent_tokenize(decontextualized_evidence)
        if len(decontextualized_evidences_sent) > 1:
            continue

        qas = {"Q": [], "A": []}
        # extract_entity_prompt = f'Here is a sentence: {decontextualized_evidence}\n\n' \
        #                         f'Extract the named entities (location, data, character, number etc.) in it, and separate them by \'|\'.'
        # raw_entities_by_gpt3 = handler.get_output(input_text=extract_entity_prompt, max_tokens=128, temperature=0.0,
        #                                           suffix=None, stop=None, do_tunc=False, echo=False)
        # entities = [s.strip() for s in raw_entities_by_gpt3.strip().split('|')]
        # print("2:",entities)
        # short_entities_2 = []
        # for entity in entities:
        #     sid = 0
        #     eid = len(entity) - 1
        #     while sid < len(entity):
        #         if entity[sid] != ' ' and entity[sid] not in punctuation:
        #             break
        #         sid += 1
        #     while eid >= 0:
        #         if entity[eid] != ' ' and entity[eid] not in punctuation:
        #             break
        #         eid -= 1
        #     entity = entity[sid:eid + 1]
        #     ws = entity.split(' ')
        #     if len(ws) <= 5: short_entities_2.append(entity)
        # if len(short_entities_2) == 0:
        #     continue

        # short_entities=list(set(short_entities_1).intersection(set(short_entities_2)))
        # if len(short_entities)==0:
        #     continue
        # able_to_find = True
        for ent in entities:
            answer = ent

            if answer not in decontextualized_evidence:
                continue

            # gen_question_prompt = "Wrie the question and give out the related passage and evidence sentence.\n\n" \
            #                       "Question:"
            #
            # gen_question_suffix = f"\nAnswer: {answer}" \
            #                       f"Passage: {final_passage}\n" \
            #                       f"Evidence sentence: {evidence}"

            raise_question_prompt = f"Write a question whose answer is \"{answer}\".\n" \
                                    f"The question is about the passage: {final_passage}\n" \
                                    f"The key sentence for answering this question is: {evidence}\n" \
                                    f"The question should be answerable without context." \
                                    f"So the most appropriate question is:"
            question = handler.get_output(input_text=raise_question_prompt, max_tokens=30,
                                          temperature=0.0, suffix=None, stop=None, do_tunc=True,
                                          echo=False,return_prob=False)

            # double check

            check_answer_prompt_1 = "Read the passage and the evidence sentence, answer the question with less then 5 words.\n\n" \
                                    f"Passage: {final_passage}\n" \
                                    f"Evidence sentence: {evidence}\n" \
                                    f"Question: {question}\n" \
                                    f"Answer:"

            answer_check_1 = handler.get_output(input_text=check_answer_prompt_1, max_tokens=10,
                                                temperature=0.0, suffix=None, stop=None, do_tunc=True,
                                                echo=False,return_prob=False)

            check_answer_prompt_2 = "Read the evidence sentence and answer the question with less then 5 words.\n\n" \
                                    f"Evidence sentence: {decontextualized_evidence}\n" \
                                    f"Question: {question}\n" \
                                    f"Answer:"

            answer_check_2 = handler.get_output(input_text=check_answer_prompt_2, max_tokens=10,
                                                temperature=0.0, suffix=None, stop=None, do_tunc=True,
                                                echo=False,return_prob=False)

            # print('Q: ',question)
            # print('A: ',answer)
            # print('AC1: ',answer_check_1)
            # print('AC2: ',answer_check_2)

            if normalize_answer(answer_check_1) != normalize_answer(answer) or normalize_answer(
                    answer_check_2) != normalize_answer(answer):
                continue

            qas['Q'].append(question)
            qas['A'].append(answer)
        if len(qas['Q']) > 0:
            able_to_find = True
            evidences.append(evidence)
            decontextualized_evidences.append(decontextualized_evidence)
            qass.append(qas)

    if able_to_find:
        return {'passage': final_passage,
                'QA': qass,
                'evidence': evidences,
                'decontextualized_evi': decontextualized_evidences}
    else:
        return -1
    # extarct_entity_sent = f'Here is a passage: {passage_by_gpt3}\n\n' \
    #                       f'Write down the named entities (like date, location, character, number) in this passage, the named entities should be as short as possible, and separate them by \'|\'.'
    #
    # # some codes to extract named entities with the above prompt
    # raw_entities_by_gpt3 = handler.get_output(input_text=extarct_entity_sent, max_tokens=128, temperature=0.0,
    #                                           suffix=None, stop=None, do_tunc=False, echo=False)
    #
    # entities = [s.strip() for s in raw_entities_by_gpt3.strip().split('|')]
    # short_entities = []
    # for entity in entities:
    #     ws = entity.split(' ')
    #     if len(ws) <= 5: short_entities.append(entity)
    # assert len(short_entities) > 0
    #
    # selected_entity = random.choice(short_entities)
    #
    # gen_question_prompt = "Read the following passage and answer the question.\n\n" \
    #                       f"Question: "
    #
    # gen_question_suffix = f"\nPassage: {passage_by_gpt3}\n" \
    #                       f"Answer: {selected_entity}"
    #
    # # choose one of the entity, and generate question with the above prompt
    # raw_question_by_gpt3 = handler.get_output(input_text=gen_question_prompt, max_tokens=20, temperature=0.0,
    #                                           suffix=gen_question_suffix, stop=None, do_tunc=False, echo=False)
    #
    # question = raw_question_by_gpt3.strip()
    #
    # gen_evidence_prompt = "Read the following passage and answer the question, you need to extract the evidence sentence from the passage.\n\n" \
    #                       f"{passage_by_gpt3}\n\n" \
    #                       f"Question: {question}" \
    #                       f"Evidence sentence: "
    # gen_evidence_suffix = f"\nAnswer: {selected_entity}"
    #
    # # generate evidence sentence with the above prompt
    # raw_evidence_by_gpt3 = handler.get_output(input_text=gen_evidence_prompt, max_tokens=50, temperature=0.0,
    #                                           suffix=gen_evidence_suffix, stop=None, do_tunc=False, echo=False)
    # evidence = raw_evidence_by_gpt3.strip()
    #
    # decontextualized_evidence_prompt = "Given a passage and a sentence extracted from it, rewrite the sentence to make its meaning clear enough without a context.\n\n" \
    #                                    f"{passage_by_gpt3}\n" \
    #                                    f"Sentence: {evidence}\n" \
    #                                    "Rewrite:"
    #
    # # decontextualize the evidence sentence with the above prompt
    #
    # decontextualized_evidence = handler.get_output(input_text=decontextualized_evidence_prompt, max_tokens=50,
    #                                                temperature=0.0, suffix=None, stop=['\n'], do_tunc=True, echo=False)
    #
    # return {'passage': passage_by_gpt3, 'entities': entities, 'answer': selected_entity, 'question': question,
    #         'evidence': evidence, 'decontextualized_evi': decontextualized_evidence}


def generate_a_detailed_ODQA_evionly(handler, input_question=None, NE="llm", genqway='completion'):
    gen_sent = f"This is a short sentence (10 - 20 words) from Wikipedia about"

    if input_question is not None:
        gen_sent = f"Generate a background sentence from Wikipedia to answer the given question. {input_question}"

    # some codes to generate passage with the above prompt

    raw_gen_by_gpt3 = handler.get_output(input_text=gen_sent, max_tokens=40, temperature=0.0, suffix=None,
                                         stop=None, do_tunc=False, echo=True, ban_pronoun=True,return_prob=False)

    print(raw_gen_by_gpt3)

    gen_by_gpt3 = None
    lines = raw_gen_by_gpt3.split('\n')
    for line in lines:
        if line.startswith(gen_sent) or line.strip() == '' or len(line) < 10:
            continue
        else:
            gen_by_gpt3 = line.strip()
            break
    if gen_by_gpt3 is None or gen_by_gpt3[-1] not in punctuation:
        return -1

    evidence = gen_by_gpt3
    able_to_find = False
    # genquestion_prompt = "Generate some questions based on this sentence.\n" \
    #                      "The answer should be an named entity like date, number, character, location or organization in the sentence.\n" \
    #                      "Format as \'Q: A:\'.\n\n" \
    #                      f"{evidence}"
    #
    # genqa = handler.get_output(input_text=genquestion_prompt, max_tokens=128, temperature=0.0, suffix=None,
    #                                      stop=None, do_tunc=False, echo=False, ban_pronoun=True)
    #
    # print(genqa)
    # raise ValueError

    print(evidence)

    if 'llm' in NE:
        extract_entity_prompt = f'Here is a sentence: {evidence}\n\n' \
                                f'Extract the named entities (like date, location, organization, character, number) in it, and separate them by \'|\'.' \
                                f'If no named entity in it, write \'None\' only.'
        raw_entities_by_gpt3 = handler.get_output(input_text=extract_entity_prompt, max_tokens=50, temperature=0.0,
                                                  suffix=None, stop=None, do_tunc=False, echo=False,return_prob=False)
        len_space_split_ents = len(evidence.split(' '))
        raw_entities = [s.strip() for s in raw_entities_by_gpt3.strip().split('|')]

        if len(raw_entities) > int(0.8 * len_space_split_ents):
            return -1

        entities = [s for s in raw_entities if s.strip() != '' and len(s.split(' ')) <= 5 and s != 'None']
    elif 'nltk' in NE:
        entities = process_an_item(evidence, nounonly=True, neonly=False)
    else:
        raise NotImplementedError

    QAs = {"Evidence": evidence, "QAs": {"Q": [], "A": []}}

    print(entities)

    for ent in entities:
        # gen_q_prompt = f"Here is some background information.\n\n" \
        #                f"{evidence}\n" \
        #                f"Write a question about the above information, its answer is {ent}.\n" \
        #                f"Question:"
        if genqway == 'completion':
            raise_question_prompt = f"Write a short question whose answer is \"{ent}\".\n" \
                                    f"The question is about the sentence: {evidence}\n" \
                                    f"The question should be answerable without context." \
                                    f"So the most appropriate question is:"

            question = handler.get_output(input_text=raise_question_prompt, max_tokens=30,
                                          temperature=0.0, suffix=None, stop=None, do_tunc=True,
                                          echo=False,return_prob=False)
        elif genqway == 'insert':
            raise_question_prompt = f"{evidence}\n" \
                                    f"Question:"
            raise_question_suffix = f"Answer: {ent}"
            question = handler.get_output(input_text=raise_question_prompt, max_tokens=30,
                                          temperature=0.0, suffix=raise_question_suffix, stop=None, do_tunc=True,
                                          echo=False,return_prob=False)
        else:
            raise NotImplementedError

        if len(nltk.word_tokenize(question)) > len(nltk.word_tokenize(evidence)):
            continue

        question = question.strip()

        print(f"Q: {question} A: {ent}")

        check_answer_prompt = "Read the evidence sentence and answer the question with less then 5 words.\n\n" \
                              f"Evidence sentence: {evidence}\n" \
                              f"Question: {question}\n" \
                              f"Answer:"

        answer_check = handler.get_output(input_text=check_answer_prompt, max_tokens=20,
                                          temperature=0.0, suffix=None, stop=None, do_tunc=True,return_prob=False)

        if normalize_answer(answer_check) != normalize_answer(ent):
            continue

        QAs["QAs"]["Q"].append(question)
        QAs["QAs"]["A"].append(ent)
        able_to_find = True

    if able_to_find:
        return QAs
    else:
        return -1


def flatten_and_group_evionly(input_file):
    # input a jsonline
    flattened = []

    with open(input_file) as f:
        lines = f.readlines()
    for line in lines:
        res = eval(line.strip())
        evi = res['Evidence']
        for q, a in zip(res['QAs']['Q'], res['QAs']['A']):
            flattened.append({'question': q,
                              "answer": a,
                              "evidence_only": evi,
                              })

    random.shuffle(flattened)
    groups = [1, 2, 3, 4, 5, 6, 7, 8]
    start = 0
    alll = {}
    for sid in range(3):
        for num in groups:
            end = start + num
            selected_flatten = flattened[start:end]
            start += num
            key = f"num{num}_group{sid}"
            alll[key] = selected_flatten

    with open("old_gpt3_gen_samples/grouped_gen_res_evionly.json", "w") as f:
        json.dump(alll, f)


def flatten_and_group(input_file):
    # input a jsonline
    flattened = []

    with open(input_file) as f:
        lines = f.readlines()
    for line in lines:
        res = eval(line.strip())
        passage = res['passage']
        for e, de, qas in zip(res['evidence'], res['decontextualized_evi'], res['QA']):
            for q, a in zip(qas['Q'], qas['A']):
                flattened.append({'question': q,
                                  "answer": a,
                                  "passage": passage,
                                  "keysent": e,
                                  "evidence_only": de,
                                  })
    random.shuffle(flattened)
    groups = [1, 2, 3, 4, 5, 6, 7, 8]
    start = 0
    alll = {}
    for sid in range(3):
        for num in groups:
            end = start + num
            selected_flatten = flattened[start:end]
            start += num
            key = f"num{num}_group{sid}"
            alll[key] = selected_flatten
    with open("old_gpt3_gen_samples/grouped_gen_res.json", "w") as f:
        json.dump(alll, f)


def gen_prompt(handler, suffix, prefix):
    instruction = handler.get_output(input_text=prefix, max_tokens=128, temperature=0.0, suffix=suffix,
                                     stop=None, do_tunc=False, echo=False, ban_pronoun=False,return_prob=False)
    print(instruction)


def generate_w_topic(handler, topic_category, term):
    def clean_entity(span):
        span = span.strip()
        if span == '':
            return ''
        sid, eid = 0, len(span) - 1
        while sid < len(span) and span[sid] in expanded_punction_and_whitespace: sid += 1
        while eid >= 0 and span[eid] in expanded_punction_and_whitespace: eid -= 1
        return span[sid:eid + 1]

    print(f"====== {topic_category} - {term} ======")
    nlp = spacy.load("en_core_web_sm")
    gen_passape_prompt = f"This is a passage from Wikipedia about the {topic_category}, {term}:"
    passage = handler.get_output(input_text=gen_passape_prompt, max_tokens=256, do_tunc=True,return_prob=False)
    sents = nltk.sent_tokenize(passage)
    sents = sents[:-1] if not sents[-1].endswith('.') else sents
    final_passage = " ".join(sents)

    gens = {"passage": final_passage, "EQA": []}

    entities = [[], []]
    # entity_source_1
    doc = nlp(final_passage)
    entities_spacy = [entity.text for entity in doc.ents]
    for ent in entities_spacy:
        rent = clean_entity(ent)
        if normalize_answer(rent) not in entities[1]:
            entities[0].append(rent)
            entities[1].append(normalize_answer(rent))

    # entity_source_2
    extract_entity_prompt = f'Here is a passage: {final_passage}\n\n' \
                            f'Extract the named entities (like date, location, organization, character, number) in it, and separate them by \'|\'.' \
                            f'If no named entity in it, write \'None\' only.'
    raw_entities_by_gpt3 = handler.get_output(input_text=extract_entity_prompt, max_tokens=50, temperature=0.0,return_prob=False)
    len_space_split_ents = len(final_passage.split(' '))
    raw_entities = [s.strip() for s in raw_entities_by_gpt3.strip().split('|')]
    if len(raw_entities) > int(0.8 * len_space_split_ents): raw_entities = []
    entities_gpt3 = [s for s in raw_entities if s.strip() != '' and len(s.split(' ')) <= 5 and s != 'None']
    for ent in entities_gpt3:
        rent = clean_entity(ent)
        if normalize_answer(rent) not in entities[1]:
            entities[0].append(rent)
            entities[1].append(normalize_answer(rent))

    # entity_source_3
    entities_nltk = process_an_item(final_passage, True)
    for ent in entities_nltk:
        rent = clean_entity(ent)
        if normalize_answer(rent) not in entities[1]:
            entities[0].append(rent)
            entities[1].append(normalize_answer(rent))

    # clean entities[0], so that sub_strings are removed
    final_entities = [[], []]
    for idx, item in enumerate(entities[0]):
        addf = True
        for another_item in entities[0][:idx] + entities[0][idx + 1:]:
            if normalize_answer(item) in normalize_answer(another_item):
                addf = False
                break
        if addf:
            only_lower_case = True
            for char in item:
                if char not in whitespace + ascii_lowercase:
                    only_lower_case = False
                    break
            if only_lower_case:
                final_entities[1].append(item)
            else:
                final_entities[0].append(item)
    random.shuffle(final_entities[0])
    random.shuffle(final_entities[1])
    final_entities = final_entities[0] + final_entities[1]

    for entity in final_entities:
        # remove too-long entity
        if len(entity.split()) > 5: continue
        # generate question
        generate_question_prompt = f"\"{final_passage}\"\n\"{entity}\" is the answer to the question:"
        question = handler.get_output(input_text=generate_question_prompt, max_tokens=50, temperature=0.0, do_tunc=True,
                                      stop=None,return_prob=False)
        if question.strip() == "": continue
        question = clean_entity(question) + '?'
        # remove duplicate question
        duplicate_q = False
        for item in gens['EQA']:
            if normalize_answer(item['question']) == normalize_answer(question):
                duplicate_q = True
                break
        if duplicate_q: continue
        # double check
        check_answer_prompt = f"Passage: {final_passage}\nQuestion: {question}\nShort Answer (extracted from the passage, less than 6 words):"
        answer = handler.get_output(input_text=check_answer_prompt, max_tokens=50, temperature=0.0, do_tunc=True,
                                    stop=None,return_prob=False)
        if normalize_answer(answer) != normalize_answer(entity):
            new_answer = clean_entity(answer)
            if len(new_answer.split()) <= 5:
                final_answer = new_answer
            else:
                continue
        else:
            final_answer = entity
        generate_evidence_prompt = f"Passage: {final_passage}\nQuestion: {question} Answer:{final_answer}\n" \
                                   f"You can refer to the passage and write a short explanation to this Question-Answer pair, " \
                                   f"\"{final_answer}\" must in the explanation:"
        evidence = handler.get_output(input_text=generate_evidence_prompt, max_tokens=50, temperature=0.0,
                                      do_tunc=True, stop=None,return_prob=False)

        # we chooose the exact evidence that contains the answer!
        final_evidence_sents = nltk.sent_tokenize(evidence)
        find_answer_flag = False
        final_evidence = None
        for sent in final_evidence_sents:
            if final_answer in sent:
                final_evidence = sent
                find_answer_flag = True
                break
        if not find_answer_flag:
            # must ensure evidence contain answer
            continue

        gens['EQA'].append({'evidence': final_evidence, 'question': question, 'answer': final_answer})
        print(f">> Q: {question} A: {final_answer} E: {final_evidence}")
        if len(gens['EQA']) >= 10:
            # generate at most 10 QAs for each term to ensure diversity
            break

    return gens


def gen_final(handler, involved_topics):
    topics = {'politicians': (100, 'politician'),
              'athletes': (40, 'athlete'),
              'sports teams (basketball, soccer, football, baseball etc)': (40, 'sports team'),
              'sports events (tournaments, leagues, cups etc)': (40, 'sports event'),
              'countries': (40, 'country'),
              'cities': (60, 'city'),
              'historical figures': (50, 'historical figure'),
              'historical events': (50, 'historical event'),
              'wars': (40, 'war'),
              'religions': (20, 'religion'),
              'singers': (50, 'singer'),
              'songs': (50, 'song'),
              'actors and actresses': (50, 'actor or actress'),
              'movies and TV series': (50, 'movie or TV series'),
              'writers': (30, 'writer'),
              'books': (30, 'book'),
              'painters': (30, 'painter'),
              'paintings': (30, 'painting'),
              'composers': (30, 'composer'),
              'classical music': (30, 'classical music'),
              'tourist attractions (artificial and natural)': (100, 'tourist attraction'),
              'scientists': (40, 'scientist'),
              'scientific terms': (40, 'scientific term'),
              'video games': (40, 'video game'),
              'animals': (40, 'animal'),
              'plants': (40, 'plant'),
              'foods': (40, 'food'),
              'enterprises': (50, 'enterprise'),
              'international organizations': (50, 'international organization'),
              }

    ### the function
    def list_topic_terms(topic):
        prompt = f"List some {topic}, separated by '|':"
        output = handler.get_output(input_text=prompt, max_tokens=1024, temperature=1.0, presence_penalty=2.0,
                                    do_tunc=False,return_prob=False)
        terms = [it.strip() for it in output.split('|')]
        return terms

    if not os.path.exists("old_gpt3_gen_samples/topic2entity.json"):
        ### the topics
        all_topics_entities = {}
        for topic, meta in topics.items():
            num = meta[0]
            all_topics_entities[topic] = []
            while len(all_topics_entities[topic]) < num:
                # collect entities
                entities = list_topic_terms(topic)
                for entity in entities:
                    if entity not in all_topics_entities[topic]:
                        all_topics_entities[topic].append(entity)
                print(f"{topic} has collected {len(all_topics_entities[topic])} entities!")
            print(f"{topic} collection, done!")

        if not os.path.exists("old_gpt3_gen_samples"):
            os.makedirs("old_gpt3_gen_samples")

        with open("old_gpt3_gen_samples/topic2entity.json", 'w') as f:
            json.dump(all_topics_entities, f)
    else:
        with open("old_gpt3_gen_samples/topic2entity.json") as f:
            all_topics_entities = json.load(f)
    for topic in involved_topics:
        topic_cate = topics[topic][1]
        entities = all_topics_entities[topic]
        for idx, ent in enumerate(entities):
            if idx < 25 and topic_cate == 'politician': continue
            if idx < 13 and topic_cate == 'sports team': continue
            if idx < 2 and topic_cate == 'athlete': continue
            res = generate_w_topic(handler, topic_cate, ent)
            with open(f"old_gpt3_gen_samples/topic_aware_gen_{topic_cate}.jsonl", 'a') as f:
                f.write(str(res) + '\n')
            print(f"{topic} - {ent} done!")
        print(f"{topic} all done!")


def generate_w_topic_new(handler, topic, term, examplers, nlp):
    def clean_entity(span):
        span = span.strip()
        if span == '':
            return ''
        sid, eid = 0, len(span) - 1
        while sid < len(span) and span[sid] in expanded_punction_and_whitespace: sid += 1
        while eid >= 0 and span[eid] in expanded_punction_and_whitespace: eid -= 1
        return span[sid:eid + 1]
    st = time.time()
    print(f">>>>> {topic} | {term} <<<<<")
    # generate passage for term
    exampler = examplers[0][topic]
    name, text = exampler.split('\t')
    gen_passape_prompt = f"This is a passage from Wikipedia about the {topic}, {term}:\n"
    final_gen_passage_prompt = f"This is a passage from Wikipedia about the {topic}, {name}:\n{text}\n\n{gen_passape_prompt}"
    passage = handler.get_output(input_text=final_gen_passage_prompt, max_tokens=256, temperature=0.4, do_tunc=True,
                                 stop=['This is a passage from Wikipedia about '],return_prob=False)

    # remove the incomplete sentence
    sents = nltk.sent_tokenize(passage)
    sents = sents[:-1] if not sents[-1].endswith('.') else sents
    final_passage = " ".join(sents)
    gens = {"passage": final_passage, "EQA": []}

    decontextualized_sents = [''] * len(sents)

    # print(">>>>> [PASSAGE] <<<<<")
    # print(final_passage)

    # for sent in sents:
    #     print(sent)
    #     entities_llm = []
    #     extract_entity_prompt = f'Sentence: {final_passage}\n\n' \
    #                             f'Named entities (like date, location, organization, character, number) in the above sentence:\n' \
    #                             f'1.'
    #     raw_entities_by_llm = handler.get_output(input_text=extract_entity_prompt, max_tokens=100, temperature=0.0, )
    #     lines = raw_entities_by_llm.split('\n')
    #     for line in lines:
    #         rl = line.strip()
    #         if rl:
    #             ent = '.'.join(rl.split('.')[1:]).strip()
    #             entities_llm.append(ent)
    #         else:
    #             break
    #         print(">>> sent")
    #         print(sent)
    #         for ent in entities_llm:
    #             gen_question_prompt = f"Passage: {final_passage}\n" \
    #                                   f"Refer to the sentence \"{sent}\" in the passage. We know that the answer to the question"
    #             gen_question_suffix = f"is {ent}"
    #             question = handler.get_output(input_text=gen_question_prompt, max_tokens=50, temperature=0.4,
    #                                           do_tunc=True,
    #                                           stop=None, suffix=gen_question_suffix)
    #             print('QA')
    #             print(question)
    #             print(ent)
    #
    # raise ValueError
    # extract entities
    doc = nlp(final_passage)
    entities_spacy = [entity.text for entity in doc.ents]

    # entity_source_2
    entities_llm = []
    extract_entity_prompt = f'Passage: {final_passage}\n\n' \
                            f'Named entities (like date, location, organization, character, number) in the above passage:\n' \
                            f'1.'
    raw_entities_by_llm = handler.get_output(input_text=extract_entity_prompt, max_tokens=100, temperature=0.0 ,return_prob=False)
    lines = raw_entities_by_llm.split('\n')
    for line in lines:
        rl = line.strip()
        if rl:
            ent = '.'.join(rl.split('.')[1:]).strip()
            entities_llm.append(ent)
        else:
            break

    # entity_source_3
    entities_nltk = process_an_item(final_passage, True)
    all_entities = [clean_entity(x) for x in entities_spacy+entities_llm+entities_nltk]

    # all_entities = entities_llm

    entities = [[], []]
    for rent in all_entities:
        if rent.endswith('He'):
            rent = rent[:-len('He')]
        if rent.endswith('She'):
            rent = rent[:-len('She')]
        if rent.endswith('They'):
            rent = rent[:-len('They')]
        if rent.endswith('It'):
            rent = rent[:-len('It')]

        if normalize_answer(rent) not in entities[1]:
            entities[0].append(rent)
            entities[1].append(normalize_answer(rent))

    # clean entities[0], so that sub_strings are removed
    final_entities = [[], []]
    for idx, item in enumerate(entities[0]):
        addf = True
        for another_item in entities[0][:idx] + entities[0][idx + 1:]:
            if normalize_answer(item) in normalize_answer(another_item):
                addf = False
                break
        if addf:
            only_lower_case = True
            for char in item:
                if char not in whitespace + ascii_lowercase:
                    only_lower_case = False
                    break
            if only_lower_case:
                final_entities[1].append(item)
            else:
                final_entities[0].append(item)
    random.shuffle(final_entities[0])
    random.shuffle(final_entities[1])

    # 1 are the entities that contain only lowercase, otherwise 0
    final_entities = final_entities[0] + final_entities[1]

    print(">>>>> [ENTITIES] <<<<<")
    print(final_entities)

    for entity in final_entities:
        # print(f"Current entity : {entity} >>>")
        # remove too-long entity
        if len(entity.split()) > 6: continue
        # generate question, temp=0.7 to ensure diversity
        generate_question_prompt = f"Passage: {final_passage}\n" \
                                   f"Question:"

        generate_question_suffix = f"\nAnswer: {entity}"
        question = handler.get_output(input_text=generate_question_prompt, max_tokens=50, temperature=0.2, do_tunc=True,
                                      stop=None, suffix=generate_question_suffix, ban_pronoun=True,return_prob=False)

        # question = handler.get_output(input_text=generate_question_prompt, max_tokens=50, temperature=0.4, do_tunc=True,
        #                               stop=None, ban_pronoun=True,suffix=generate_question_suffix)

        if '\n' in question:
            question = question[:question.find('\n')]

        # do not generate question
        if question.strip() == "": continue

        question = clean_entity(question) + '?'

        # print(f"Generated question : {question}")

        # remove duplicate question
        duplicate_q = False
        for item in gens['EQA']:
            if normalize_answer(item['question']) == normalize_answer(question):
                duplicate_q = True
                break
        if duplicate_q: continue

        # double check
        check_answer_prompt = f"Passage: {final_passage}\n" \
                              f"Refer to the passage above and answer the question.\n" \
                              f"Question: {question}\n" \
                              f"Answer:"
        answer = handler.get_output(input_text=check_answer_prompt, max_tokens=50, temperature=0.0, do_tunc=True,
                                    stop=['Passage:'],return_prob=False)

        # print(f"Predicted answer: {answer}")

        if normalize_answer(answer) != normalize_answer(entity):
            new_answer = clean_entity(answer)
            # If conflict, accept the new answer when it is short
            if len(new_answer.split()) <= 6:
                final_answer = new_answer
            else:
                continue
        else:
            final_answer = entity

        # final_answer = entity

        findtimes = 0
        need_sent = None
        need_i = -1
        for i, sent in enumerate(sents):
            if normalize_answer(final_answer) in normalize_answer(sent):
                findtimes += 1
                need_sent = sent
                need_i = i

        if findtimes != 1:
            # cannot accurately locate
            continue

        # print(f'keysent >>>')
        # print(need_sent)

        if decontextualized_sents[need_i] == '':
            decontextualize_prompt = ""
            for k, v in examplers[1].items():
                vvs = v.split('\t')
                decontextualize_prompt += f"Passage: {vvs[0]}\n" \
                                          f"Sentence: {vvs[1]}\n" \
                                          f"The above sentence is extracted from the passage, rewrite it " \
                                          f"to make its meaning clear without context: {vvs[2]}\n\n"

            decontextualize_prompt += f"Passage: {final_passage}\n" \
                                      f"Sentence: {need_sent}\n" \
                                      f"The above sentence is extracted from the passage, rewrite it " \
                                      f"to make its meaning clear without context:"

            evidence = handler.get_output(input_text=decontextualize_prompt, max_tokens=50, temperature=0.0,
                                          do_tunc=True, stop=['Passage:'], ban_pronoun=True,return_prob=False)
            decontextualized_sents[need_i] = evidence
        else:
            evidence = decontextualized_sents[need_i]

        # generate_evidence_prompt = f"Passage: {final_passage}\n" \
        #                            f"Question: {question} Answer: {final_answer}\n" \
        #                            f"Refer to the passage, write an explanation for the above Question-Answer pair:"
        # evidence = handler.get_output(input_text=generate_evidence_prompt, max_tokens=50, temperature=0.0,
        #                               do_tunc=True, stop=['Passage:'], ban_pronoun=True)

        # print("Decontextualized >>>")
        # print(evidence)

        # we chooose the exact evidence that contains the answer!
        final_evidence_sents = nltk.sent_tokenize(evidence)
        find_answer_flag = False
        final_evidence = None
        for sent in final_evidence_sents:
            if final_answer in sent:
                final_evidence = sent
                find_answer_flag = True
                break
        if not find_answer_flag:
            # must ensure evidence contain answer
            continue
        final_evidence = final_evidence.strip()

        gens['EQA'].append({'keysent':need_sent, 'evidence_only': final_evidence, 'question': question, 'answer': final_answer})
        print(f">> Q: {question} A: {final_answer} E: {final_evidence}")
        if len(gens['EQA']) >= 20:
            # generate at most 10 QAs for each term to ensure diversity
            break
    print(f'end@{time.time()-st}s')
    return gens


def generate_w_topic_local(localgenmodel, topic, term, examplers, nlp):
    def clean_entity(span):
        span = span.strip()
        if span == '':
            return ''
        sid, eid = 0, len(span) - 1
        while sid < len(span) and span[sid] in expanded_punction_and_whitespace: sid += 1
        while eid >= 0 and span[eid] in expanded_punction_and_whitespace: eid -= 1
        return span[sid:eid + 1]
    st = time.time()
    print(f">>>>> {topic} | {term} <<<<<")
    # generate passage for term
    exampler = examplers[0][topic]
    name, text = exampler.split('\t')
    gen_passape_prompt = f"This is a passage from Wikipedia about the {topic}, {term}:\n"
    final_gen_passage_prompt = f"This is a passage from Wikipedia about the {topic}, {name}:\n{text}\n\n{gen_passape_prompt}"
    # passage = handler.get_output(input_text=final_gen_passage_prompt, max_tokens=256, temperature=0.4, do_tunc=True,
    #                              stop=['This is a passage from Wikipedia about '],return_prob=False)
    passage = localgenmodel.get_output(input_text=final_gen_passage_prompt, max_tokens=256, temperature=0.4,)
    # extract contents before the "This is a passage from Wikipedia about"
    passage = passage.split('This is a passage from Wikipedia about')[0]


    # remove the incomplete sentence
    sents = nltk.sent_tokenize(passage)
    sents = sents[:-1] if not sents[-1].endswith('.') else sents
    final_passage = " ".join(sents)

    print(">>> Final Passage >>>")
    print(final_passage)
    print(">>>  >>>  >>>  >>>")

    gens = {"passage": final_passage, "EQA": []}

    decontextualized_sents = [''] * len(sents)

    doc = nlp(final_passage)
    entities_spacy = [entity.text for entity in doc.ents]
    # entity_source_2
    entities_nltk = process_an_item(final_passage, True)
    all_entities = [clean_entity(x) for x in entities_spacy+entities_nltk]

    entities = [[], []]
    for rent in all_entities:
        if rent.endswith('He'):
            rent = rent[:-len('He')]
        if rent.endswith('She'):
            rent = rent[:-len('She')]
        if rent.endswith('They'):
            rent = rent[:-len('They')]
        if rent.endswith('It'):
            rent = rent[:-len('It')]

        if normalize_answer(rent) not in entities[1]:
            entities[0].append(rent)
            entities[1].append(normalize_answer(rent))

    # clean entities[0], so that sub_strings are removed
    final_entities = [[], []]
    for idx, item in enumerate(entities[0]):
        addf = True
        for another_item in entities[0][:idx] + entities[0][idx + 1:]:
            if normalize_answer(item) in normalize_answer(another_item):
                addf = False
                break
        if addf:
            only_lower_case = True
            for char in item:
                if char not in whitespace + ascii_lowercase:
                    only_lower_case = False
                    break
            if only_lower_case:
                final_entities[1].append(item)
            else:
                final_entities[0].append(item)
    random.shuffle(final_entities[0])
    random.shuffle(final_entities[1])

    # 1 are the entities that contain only lowercase, otherwise 0
    final_entities = final_entities[0] + final_entities[1]

    print(">>>>> [ENTITIES] <<<<<")
    print(final_entities)

    for entity in final_entities:

        if len(entity.split()) > 6: continue
        # generate question, temp=0.2 to ensure diversity
        generate_question_prompt = f"\"{final_passage}\"\n\"{entity}\" is the answer to the question:"
        question = localgenmodel.get_output(generate_question_prompt, max_tokens=50, temperature=0.2, ban_pronoun=True)

        if '\n' in question:
            question = question[:question.find('\n')]

        # do not generate question
        if question.strip() == "": continue

        question = clean_entity(question) + '?'

        print("-"*20)
        print(f"For entity: {entity}, the generated question : {question}")
        print("-"*20)

        # remove duplicate question
        duplicate_q = False
        for item in gens['EQA']:
            if normalize_answer(item['question']) == normalize_answer(question):
                duplicate_q = True
                break
        if duplicate_q: continue

        # double check
        check_answer_prompt = f"Passage: {final_passage}\nQuestion: {question}\nShort Answer (extracted from the passage, less than 6 words):"
        answer = localgenmodel.get_output(check_answer_prompt, max_tokens=50, temperature=0).strip()
        answer = answer[:answer.find('\n')] if '\n' in answer else answer
        # print(f"Predicted answer: {answer}")

        if normalize_answer(answer.lower()) != normalize_answer(entity.lower()):
            continue
            # print("-" * 20)
            # print(f"Double check encounter conflict, the new answer is: {answer}")
            # print("-" * 20)
            # new_answer = clean_entity(answer)
            # # If conflict, accept the new answer when it is short
            # if len(new_answer.split()) <= 6:
            #     final_answer = new_answer
            # else:
            #     continue
        else:
            final_answer = entity

        # check by LM
        # check_same_prompt = f"\"Whig\" and \"Whig Party\" mean the same thing (Yes or No)? Yes\n" \
        #                     f"\"Confederate\" and \"The Confederacy\" mean the same thing (Yes or No)? Yes\n" \
        #                     f"\"Senators\" and \"Abraham Lincoln\" mean the same thing (Yes or No)? No\n" \
        #                     f"\"1865\" and \"1861\" mean the same thing (Yes or No)? No\n" \
        #                     f"\"{normalize_answer(answer)}\" and \"{normalize_answer(entity)}\" mean the same thing (Yes or No)?"
        # check_res = localgenmodel.get_output(check_same_prompt, max_tokens=20, temperature=0).strip()
        # check_res = check_res[:check_res.find('\n')] if '\n' in check_res else check_res
        # print(f"Check result: {check_res} for entity : {entity} and answer : {answer}")
        # if 'yes' not in check_res.lower():
        #     continue
        # final_answer = entity


        # final_answer = entity

        # findtimes = 0
        # need_sent = None
        # need_i = -1
        # for i, sent in enumerate(sents):
        #     if normalize_answer(final_answer) in normalize_answer(sent):
        #         findtimes += 1
        #         need_sent = sent
        #         need_i = i
        #
        # if findtimes != 1:
        #     # cannot accurately locate
        #     continue
        #
        # # print(f'keysent >>>')
        # # print(need_sent)
        #
        # if decontextualized_sents[need_i] == '':
        #     decontextualize_prompt = ""
        #     for k, v in examplers[1].items():
        #         vvs = v.split('\t')
        #         decontextualize_prompt += f"Passage: {vvs[0]}\n" \
        #                                   f"Sentence: {vvs[1]}\n" \
        #                                   f"The above sentence is extracted from the passage, rewrite it " \
        #                                   f"to make its meaning clear without context: {vvs[2]}\n\n"
        #
        #     decontextualize_prompt += f"Passage: {final_passage}\n" \
        #                               f"Sentence: {need_sent}\n" \
        #                               f"The above sentence is extracted from the passage, rewrite it " \
        #                               f"to make its meaning clear without context:"
        #
        #     evidence = localgenmodel.get_output(decontextualize_prompt, max_tokens=50, temperature=0.0,ban_pronoun=True)
        #     # find "Passage:" and extract content before it
        #     evidence = evidence[:evidence.find('Passage:')].strip()
        #
        #     decontextualized_sents[need_i] = evidence
        # else:
        #     evidence = decontextualized_sents[need_i]
        #
        # # we chooose the exact evidence that contains the answer!
        # final_evidence_sents = nltk.sent_tokenize(evidence)
        # find_answer_flag = False
        # final_evidence = None
        # for sent in final_evidence_sents:
        #     if final_answer in sent:
        #         final_evidence = sent
        #         find_answer_flag = True
        #         break
        # if not find_answer_flag:
        #     # must ensure evidence contain answer
        #     continue
        # final_evidence = final_evidence.strip()

        generate_evidence_prompt = f"Passage: {final_passage}\nQuestion: {question} Answer:{final_answer}\n" \
                                   f"You can refer to the passage and write a short explanation to this Question-Answer pair, " \
                                   f"\"{final_answer}\" must in the explanation:"
        evidence = localgenmodel.get_output(generate_evidence_prompt, max_tokens=50, temperature=0.0, ban_pronoun=False)

        # we chooose the exact evidence that contains the answer!
        final_evidence_sents = nltk.sent_tokenize(evidence)
        find_answer_flag = False
        final_evidence = None
        for sent in final_evidence_sents:
            if final_answer in sent:
                final_evidence = sent.strip().replace('\n','')
                find_answer_flag = True
                break
        if not find_answer_flag:
            # must ensure evidence contain answer
            continue

        print(f'final evidence >>>')
        print(final_evidence)
        print("-"*20)

        gens['EQA'].append({'keysent':'', 'evidence_only': final_evidence, 'question': question, 'answer': final_answer})
        print(f">> Q: {question} A: {final_answer} E: {final_evidence}")
        if len(gens['EQA']) >= 10:
            # generate at most 10 QAs for each term to ensure diversity
            break
    print(f'end@{time.time()-st}s')
    return gens


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--api_file',default=None)
    parser.add_argument('--start_line',type=int,default=0)
    parser.add_argument('--end_line',type=int,default=-1)
    args = parser.parse_args()

    # localgenmodel = LocalGenModel('alpaca-7B')

    # nlp = spacy.load("en_core_web_sm")

    with open('./codex_gen_samples/topic_aware_subs/examplers.json') as f:
        examplers = json.load(f)
    with open("./codex_gen_samples/topic_aware_subs/all_entities.txt") as f:
        lines = f.readlines()

    ss = {}
    for line in lines[args.start_line:args.end_line]:
        topic, term = line.strip().split('\t')
        if topic not in ss:
            ss[topic] = []
        ss[topic].append(term)
    counter = {k:0 for k in ss.keys()}
    topic_list = list(ss.keys())
    t=0
    while True:
        for topic in topic_list:
            # choose term according to counter
            if counter[topic] >= len(ss[topic]):
                continue
            t+=1
            term = ss[topic][counter[topic]]
            counter[topic] += 1
            # res = generate_w_topic_local(localgenmodel, topic, term, examplers,nlp)

            # with open(f'./alpaca_gen_samples/topic_aware_subs/{args.start_line}-{args.end_line}.jsonl','a') as f:
            #     f.write(str(res)+'\n')

        # all counter exceed max, break
        if all([v >= len(ss[k]) for k,v in counter.items()]):
            break

    print(t)





