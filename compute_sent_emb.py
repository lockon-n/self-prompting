"""
This is a simple application for sentence embeddings: clustering
Sentences are mapped to sentence embeddings and then k-mean clustering is applied.
"""
import json
import random

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--genfile", default='./gpt3_gen_samples/filtered_flattened_topic_aware_gens.json')
    parser.add_argument("--num_clusters_list", default="[1,2,4,6,8,10,12,14,16]")
    parser.add_argument("--clusterfile", default='./gpt3_gen_samples/filtered_clustering_results_sbert_qa.json')
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--way",default='sbert')
    parser.add_argument("--qapair",action='store_true')
    args = parser.parse_args()

    print("We will do clustering in this program...")
    with open(args.genfile) as f:
        rd = json.load(f)
    corpus = [item['question']+" "+item["answer"] if args.qapair else item['question'] for item in rd]

    if args.way == 'sbert':
        embedder = SentenceTransformer('all-mpnet-base-v2')
        # embedder = SentenceTransformer('sentence-transformers/stsb-roberta-large')
        # Corpus with example sentences
        corpus_embeddings = embedder.encode(corpus,device=args.device)  # a list of tensor
    else:
        raise ValueError

    # cluster_res_file = "./old_gpt3_gen_samples/clustering_results.json"

    num_clusters_list = eval(args.num_clusters_list)


    all_clus_res = {}

    for num_cluster in num_clusters_list:
        # clustering
        clustering_model = KMeans(n_clusters=num_cluster, random_state=42)
        clustering_model.fit(corpus_embeddings)
        cluster_assignment = clustering_model.labels_
        cluster_centers = clustering_model.cluster_centers_

        # putting samples into corresponding clusters
        clustered_id = [[] for i in range(num_cluster)]
        for sentence_id, cluster_id in enumerate(cluster_assignment):
            clustered_id[cluster_id].append(sentence_id)

        # sort each cluster-sequence in the order of closest2center to farthest
        sorted_clustered_id = []

        print(f"For clustering with {num_cluster} clusters")

        for cid in range(num_cluster):
            raw_seq = clustered_id[cid]
            center = cluster_centers[cid]
            embds = np.concatenate([corpus_embeddings[idx].reshape(1, -1) for idx in raw_seq])
            diff = embds - center  # bs,dim
            norm = np.sum(np.square(diff), axis=1)
            rank = np.argsort(norm)
            new_seq = [raw_seq[idx] for idx in rank]
            sorted_clustered_id.append(new_seq)
            print(f"=========cluster {cid}==========")
            for sid in new_seq[:10]:
                print(corpus[sid])

        all_clus_res[num_cluster]=sorted_clustered_id

    with open(args.clusterfile,'w') as f:
        json.dump(all_clus_res,f)

    print('Done!')


