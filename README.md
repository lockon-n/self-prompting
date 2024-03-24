## README
We provide a sample code for Self-Prompting, you can run Self-Prompting (InstructGPT) on a toy dataset.

### Requirements
- python 3.7
- openai==0.25.0
- sentence-transformers==2.2.2
- torch==1.13.1
- transformers==4.28.1

### Steps
#### Preparation
Save your openai api key into `./related_files/openai_api.txt`.

We provide a sample test dataset in `./datasets/samples_nq/test.jsonl`.
#### Data generation
We provide the generated data by InstructGPT in `./gpt3_gen_samples/filtered_flattened_topic_aware_gens.json`

#### Clustering & Selection
Do clustering
```
python compute_sent_emb.py \
--genfile ./gpt3_gen_samples/filtered_flattened_topic_aware_gens.json \
--num_clusters_list [1,2,4,6,8,10] \
--clsuterfile ./gpt3_gen_samples/filtered_clustering_results_sbert_qa.json \
--device cuda:0 \
--way sbert \
--qapair
```
Do selection
```
python sbert_retrieve.py \
--genfile ./gpt3_gen_samples/filtered_flattened_topic_aware_gens.json \
--clusterfile ./gpt3_gen_samples/filtered_clustering_results_sbert_qa.json \
--device cuda:0 \
--way sbert \
--qapair \
--model_suffix sbert
```
#### Inference
```
python -u new_main.py \
--api_file ./related_files/openai-api.txt \
--model_name instructgpt \
--dataset_name samples_nq \
--dataset_dir ./datasets/samples_nq \
--start_pos 0 \
--end_pos -1 \
--output_files_folder ./outputs/samples_nq \
--num_sample 10 \
--source gpt3gen \
--pick_demo_seed -1 \
--sid -7 \
--instruction_way -2 \
--demo_way 4 \
--with_restrict ans \
--clusters_filename ./gpt3gen/filtered_clustering_results_sbert_qa.json \
--flattened_gen_data ./gpt3gen/filtered_flattened_topic_aware_gens.json \
--clusters_retrieve_filename ./gpt3gen/filtered_clustered_retrieve_res_samples_nq_sbert_qa.json
```
#### Evaluation
```
python collect_merge_delete_eval.py
```