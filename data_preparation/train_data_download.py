import ir_datasets
import logging
from datetime import datetime
from collections import defaultdict
from tqdm.auto import tqdm
from pathlib import Path
import json
import csv
import os
import argparse

'''
This script is written to support the work of "C3: Continued Pretraining with Contrastive Weak Supervision for Cross Language Ad-Hoc Retrieval."

Eugene Yang, Suraj Nair, Ramraj Chandradevan, Rebecca Iglesias-Flores, and Douglas W. Oard. 2022. 
C3: Continued Pretraining with Contrastive Weak Supervision for Cross Language Ad-Hoc Retrieval. 
In Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR ’22), 
July 11–15, 2022, Madrid, Spain. ACM, New York, NY, USA, 6 pages. https://doi.org/10.1145/3477495.3531886
'''


'''global variables'''
LOGGER = None
args = None


def get_dataset_examples(dataset_fname):

    # local result variables
    queries_dict, docs_dict, training_examples = defaultdict(lambda: 0), defaultdict(lambda: 0), []

    # dataset_fname example: 'clirmatrix/zh/bi139-base/en/train'
    curr_dataset = ir_datasets.load(dataset_fname)

    # returns list of tuples: [(query_id, doc_id, relevance, iteration)]
    most_relevant_docs = get_list_of_most_relevant_docs(curr_dataset)

    # many-many relationship
    # two dict qid:qtext and docid:doctext (size of doc_ids is size of most_relevant_docs)
    query_ids, doc_ids, _, _ = zip(*most_relevant_docs)

    # create subDir
    output_dir = Path(args.outDir) / ('relevance_score' + str(args.relevance_score))
    output_dir.mkdir(exist_ok=True, parents=True)

    docs_dict = curr_dataset.docs.lookup(doc_ids)

    queries_dict = curr_dataset.queries.lookup(query_ids)

    # generate tuples
    for qid, did in zip(query_ids, doc_ids):
        training_examples.append((queries_dict[qid].text, docs_dict[did].text))

    print(f'ntraining_examples generated for {dataset_fname}: {len(training_examples)}')
    
    return training_examples


def write_examples_to_csv(dataset_fname, training_examples):

    csv_outfile = Path(args.outDir) / ('relevance_score' + str(args.relevance_score)) / (dataset_fname.replace('/','_') + '.csv')
    print(f'confirm csv_outfile: {csv_outfile}') 
    
    with open(csv_outfile, mode='w') as csv_file_run_id:
        writer = csv.DictWriter(csv_file_run_id, fieldnames=['query_text', 'doc_text'], quoting=csv.QUOTE_ALL)
        # writer.writeheader()
        for example in training_examples:
            query_text, doc_text = example[0], example[1]
            writer.writerow({'query_text': query_text, 'doc_text': doc_text})

def get_list_of_most_relevant_docs(dataset):
    return [ qrel 
        for qrel in tqdm(dataset.qrels, desc='reading qrels') 
        if qrel.relevance >= args.relevance_score 
    ]

def json_dump(data, output_json_file):
    with open(output_json_file, 'w') as fout:
        json.dump(data, fout)

def json_load(filename):
    with open(filename, 'r') as fin:
        data = json.load(fin)
    fin.close()
    return data

def set_logger():
    """Helper function that formats a logger use programmers can easily debug their scripts.
    Args:
      N/A
    Returns:
      logger object
    Note:
      You can refer to this tutorial for more info on how to use logger: https://towardsdatascience.com/stop-using-print-and-start-using-logging-a3f50bc8ab0
    """

    # STEP 1
    # create a logger object instance
    logger = logging.getLogger()

    # STEP 2
    # specifies the lowest severity for logging
    logger.setLevel(logging.NOTSET)

    # STEP 3
    # set a destination for your logs or a "handler"
    # here, we choose to print on console (a consoler handler)
    console_handler = logging.StreamHandler()
    # here, we choose to output the console to an output file
    # file_handler = logging.FileHandler("mylog.log")

    # STEP 4
    # set the logging format for your handler
    log_format = '\n%(asctime)s | Line %(lineno)d in %(filename)s: %(funcName)s() | %(levelname)s: \n%(message)s'
    console_handler.setFormatter(logging.Formatter(log_format))

    # we add console handler to the logger
    logger.addHandler(console_handler)
    # we add file_handler to the logger
    # logger.addHandler(file_handler)

    return logger

def register_arguments():
    """Registers the arguments in the argparser into a global variable.
    Args:
      N/A
    Returns:
      N/A, sets the global args variable
    """

    global args

    parser = argparse.ArgumentParser()

    # Specify command line arguments.
    parser.add_argument(
        '--outDir', type=str,
        required=True,
        help="Name of output directory, this will help in keeping your runs organized."
        )
    parser.add_argument(
        '--dataDir', type=str,
        required=True,
        help="Name of data directory, tell script where to get input data files from."
        )
    parser.add_argument(
        '--clir_matrix_fname', type=str,
        required=True,
        help=".txt file containing a list of each of the CLIRmatrix datasets we are generating training pairs from."
        )
    parser.add_argument(
        '--relevance_score', type=int,
        required=True,
        help="Indicate the relevance score desired for training. Ex. 5 or 6"
        )

    # Parse command line arguments.
    args = parser.parse_args()

    # print command line arguments for this run
    LOGGER.info("---confirm argparser---")
    for arg in vars(args):
        print(arg, getattr(args, arg))


def main():


    main_start = datetime.now()
    file_obj = open(args.dataDir + args.clir_matrix_fname, "r")
    clir_matrix_fnames = [fname.strip() for fname in file_obj.readlines()]

    LOGGER.info("list of datasets to generate: \n%s", clir_matrix_fnames)

    for dataset_fname in clir_matrix_fnames:

        LOGGER.info('Start Processing %s...', dataset_fname)

        curr_dataset_processing_start = datetime.now()
        curr_training_examples = get_dataset_examples(dataset_fname)
        write_examples_to_csv(dataset_fname, curr_training_examples)
        curr_dataset_processing_time = datetime.now() - curr_dataset_processing_start
        print(f'Finished processing {dataset_fname}.... computation time: {curr_dataset_processing_time}')
    main_time = datetime.now() - main_start
    LOGGER.info('Total time in main: %s', main_time)



if __name__ == '__main__':
    LOGGER = set_logger()
    register_arguments()
    main()