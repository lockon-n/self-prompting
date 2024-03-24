"""
This is a simple application for sentence embeddings: semantic search

We have a corpus with various sentences. Then, for a given query sentence,
we want to find the most similar sentence in this corpus.

This script outputs for various queries the top 5 most similar sentences in the corpus.
"""
import json

import tqdm
from sentence_transformers import SentenceTransformer, util
import torch
import argparse
from data_utils import ODQATextData

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--genfile",default='./gpt3_gen_samples/filtered_flattened_topic_aware_gens.json')
    parser.add_argument("--cluster_file",default='./gpt3_gen_samples/filtered_clustering_results_roberta-stsb.json')
    parser.add_argument("--device",default='cuda:0')
    parser.add_argument("--way",default='sbert')
    parser.add_argument("--qapair",action='store_true')
    parser.add_argument("--model_suffix",default='sbert')
    parser.add_argument("--diy_insert",default='')
    args = parser.parse_args()

    # Corpus with example sentences
    with open(args.genfile) as f:
        rd = json.load(f)
    corpus = [item['question']+" "+item["answer"] if args.qapair else item['question'] for item in rd]

    # # embedder = SentenceTransformer('all-mpnet-base-v2')
    # embedder = SentenceTransformer('sentence-transformers/stsb-roberta-large')
    # corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True,device=args.device)

    if args.way == 'sbert':
        embedder = SentenceTransformer('all-mpnet-base-v2')
        # embedder = SentenceTransformer('sentence-transformers/stsb-roberta-large')
        # Corpus with example sentences
        corpus_embeddings = embedder.encode(corpus,device=args.device,convert_to_tensor=True,)  # a list of tensor
    else:
        raise ValueError

    with open(args.cluster_file) as f:
        cluster_res = json.load(f)
    cluster_embs = {}
    for k,v in cluster_res.items():
        cluster_embs[k] = [corpus_embeddings.index_select(dim=0,index=torch.tensor(v_i,device=args.device),) for v_i in v]
        print("==={}===".format(k))
        for sth in cluster_embs[k]:
            print(sth.shape)



    for task in ['samples_nq']:
        print("Processing {} ... ".format(task))
        args.dataset_dir = './datasets/{}'.format(task)
        obj = ODQATextData('test',args,eval_only=True)

        # Query sentences:
        queries = [obj.get_by_idx(i)['question'] for i in range(len(obj))]

        # Find the closest 64 sentences of the corpus for each query sentence based on cosine similarity

        cluster_retrieve_res = []
        if args.way =='sbert':
            all_query_embeddings = embedder.encode(queries, convert_to_tensor=True,device=args.device)
        else:
            raise ValueError

        for qid,query_embedding in enumerate(tqdm.tqdm(all_query_embeddings)):
            # query_embedding = embedder.encode(query, convert_to_tensor=True,device=args.device)
            query = queries[qid]

            xx = {}
            for k,v in cluster_embs.items():
                # k be like "16"
                xx[k]=[]
                for i,sub_corpus_embeddings in enumerate(v):
                    corr_ids_in_flat = cluster_res[k][i]
                    # i is the cluster-id, cluster_emb: (x,768) is
                    top_k = min(max(64//int(k),2), len(corr_ids_in_flat))
                    cos_scores = util.cos_sim(query_embedding, sub_corpus_embeddings)[0]
                    top_results = torch.topk(cos_scores, k=top_k)
                    ss = top_results[1].cpu().numpy().tolist()
                    tt = top_results[0].cpu().numpy().tolist()
                    sss = [(corr_ids_in_flat[j],score) for j,score in zip(ss,tt)]
                    xx[k].append(sss)

                    if qid < 10:
                        print("\n\n======================\n\n")
                        print("Query:", query)
                        print(f"\nTop 5 most similar sentences in the {i}/{k} cluster:")
                        for score, idx in zip(top_results[0][:5], top_results[1][:5]):
                            print(corpus[corr_ids_in_flat[idx]], "(Score: {:.4f})".format(score))

            cluster_retrieve_res.append(xx)


        # whatgen = 'codex_gen_samples' if 'codex_gen_samples' in args.genfile else 'gpt3_gen_samples'
        if 'codex_gen_samples' in args.genfile: whatgen = 'codex_gen_samples'
        elif 'gpt3_gen_samples' in args.genfile: whatgen = 'gpt3_gen_samples'
        elif 'gptneox_gen_samples' in args.genfile: whatgen = 'gptneox_gen_samples'
        elif 'alpaca_gen_samples' in args.genfile: whatgen = 'alpaca_gen_samples'
        else: raise ValueError

        qaorq = "qa" if args.qapair else ""

        with open(f"./{whatgen}/filtered_clustered_retrieve_res{args.diy_insert}_{task}_{args.model_suffix}_{qaorq}.json",'w') as f:
            json.dump(cluster_retrieve_res,f)