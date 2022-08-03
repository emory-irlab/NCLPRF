# NCLPRF

Neural PRF based re-ranking on CLIR

You can find [our NCLPRF paper here](https://dl.acm.org/doi/10.1145/3477495.3532013)


# Data Preparation

## Training Data

We used [CLIRMatrix](https://github.com/ssun32/CLIRMatrix) to train the bilingual XLM-R model. For the corresponding Persian(fa), Russian(ru), and Chinese(zh), we downloaded the bilingual pairs of query and passages to train the model. Use the script `train_data_download.py` to download the bilingual pairs with relevance labels of 6, 5, and 4. Then we used pairs of relevance 6 for positive pairs and remaning pairs of relevances of 5 and 4 for pseudo-relevance feedback (PRF) signals.

`$ python3 train_data_download.py --outDir ./irds_out/ --dataDir ./ --clir_matrix_fname irds_list.txt --relevance_score 6`

`$ python github_preparate_data.py --write_data_dir=ProcessedData --data_root=./irds_out --lang=ru`


## Testing Data

We used following test collections for those corresponding langauges.
1. CLEF Persian
2. CLEF Russian
3. NTCIR Chinese


# Baseline

The baseline is a casted monolingual BM25 and RM3 search. We used [Patapsco](https://github.com/hltcoe/patapsco) tool to get the baseline scores. To get the machine translated queries from English to corresponding languages, we used [COE's](https://github.com/hltcoe) internal NMT system.

After installing Patapsco and changing the flags in the config.yml file to run baseline search, run the following command. To run RM3, switch the **retrieve/rm3** flag to True.
`$ patapsco config.yml`


# Training

We used **train/train.py** script to train NCLPRF model with 1 or 2 PRF documents using different vector aggregation weightings.

# Testing

We used **test/test.py** script to do the inference and evaluate the validation and test collections over the loop of checkpoint files saved during the training iterations. [**ir-measures**](https://ir-measur.es/en/latest/) library is used to evaluate the model performance.


```@inproceedings{10.1145/3477495.3532013,
author = {Chandradevan, Ramraj and Yang, Eugene and Yarmohammadi, Mahsa and Agichtein, Eugene},
title = {Learning to Enrich Query Representation with Pseudo-Relevance Feedback for Cross-Lingual Retrieval},
year = {2022},
doi = {10.1145/3477495.3532013},
booktitle = {Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval},
pages = {1790–1795},
numpages = {6},
location = {Madrid, Spain},
series = {SIGIR '22}
}```