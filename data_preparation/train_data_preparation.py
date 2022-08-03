import pandas as pd
import json
from pathlib import Path
import csv
from collections import OrderedDict, defaultdict
import argparse


def prepare_data(write_data_dir='ProcessedData', pos_rel=6, prf1_rel=5, prf2_rel=4, prf3_rel=3, data_root='./irds_out', lang='ru', split='train'):
    """main function"""

    # load datasets
    positive_data_filename = Path(data_root) / ('relevance_score' + str(pos_rel)) / ('clirmatrix_%s_bi139-full_en_%s.csv' % (lang, split))
    prf1_data_filename = Path(data_root) / ('relevance_score' + str(prf1_rel)) / ('clirmatrix_%s_bi139-full_en_%s.csv' % (lang, split))
    prf2_data_filename = Path(data_root) / ('relevance_score' + str(prf2_rel)) / ('clirmatrix_%s_bi139-full_en_%s.csv' % (lang, split))
    prf3_data_filename = Path(data_root) / ('relevance_score' + str(prf3_rel)) / ('clirmatrix_%s_bi139-full_en_%s.csv' % (lang, split))

    pos_df = pd.read_csv(positive_data_filename, header=None)
    print(pos_df.shape)
    prf1_df = pd.read_csv(prf1_data_filename, header=None)
    print(prf1_df.shape)
    prf2_df = pd.read_csv(prf2_data_filename, header=None)
    print(prf2_df.shape)
    prf3_df = pd.read_csv(prf3_data_filename, header=None)
    print(prf3_df.shape)
    

    # convert pandas table into dict
    pos_df.columns = ['query_text', 'doc_text']
    pos_df_dict = pos_df.groupby(by='query_text', sort=False).apply(lambda x: x.to_dict(orient='records'))

    prf1_df.columns = ['query_text', 'doc_text']
    prf1_df_dict = prf1_df.groupby(by='query_text', sort=False).apply(lambda x: x.to_dict(orient='records'))

    prf2_df.columns = ['query_text', 'doc_text']
    prf2_df_dict = prf2_df.groupby(by='query_text', sort=False).apply(lambda x: x.to_dict(orient='records'))

    prf3_df.columns = ['query_text', 'doc_text']
    prf3_df_dict = prf3_df.groupby(by='query_text', sort=False).apply(lambda x: x.to_dict(orient='records'))


    # pair up queries from different relevance grade passages
    pos_df_dict_pruned = defaultdict(list)
    for k, v in pos_df_dict.items():
        pos_df_dict_pruned[k] = [e['doc_text'] for e in v]
        
    prf1_df_dict_pruned = defaultdict(list)
    for k, v in prf1_df_dict.items():
        prf1_df_dict_pruned[k] = [e['doc_text'] for e in v]

    prf2_df_dict_pruned = defaultdict(list)
    for k, v in prf2_df_dict.items():
        prf2_df_dict_pruned[k] = [e['doc_text'] for e in v]

    prf2_df_dict_pruned = defaultdict(list)
    for k, v in prf2_df_dict.items():
        prf2_df_dict_pruned[k] = [e['doc_text'] for e in v]


    # separete relevances 6, 5, and 4 into positive & prf passages. Also clean the query a bit
    formatted_data_structure = {}
    formatted_data_structure['query_text'] = []
    formatted_data_structure['doc_text'] = []
    formatted_data_structure['feedback_doc1_text'] = []
    formatted_data_structure['feedback_doc2_text'] = []
    formatted_data_structure['feedback_doc3_text'] = []
    formatted_data_structure['feedback_doc4_text'] = []
    formatted_data_structure['feedback_doc5_text'] = []
    for k, v in pos_df_dict_pruned.items():
        pos_doc_text = v[0]
        
        other_rel_5_4_doc_text_list = []
        
        # looking at rel-5
        prf1_doc_list = prf1_df_dict_pruned[k]
        for doc_text in prf1_doc_list:
            if doc_text == pos_doc_text:
                continue        
            else: 
                other_rel_5_4_doc_text_list.append(doc_text)

        # looking at rel-4
        prf2_doc_list = prf2_df_dict_pruned[k]
        for doc_text in prf2_doc_list:
            if doc_text == pos_doc_text:
                continue        
            else: 
                other_rel_5_4_doc_text_list.append(doc_text)
                
        # looking at rel-3
        prf3_doc_list = prf3_df_dict_pruned[k]
        for doc_text in prf3_doc_list:
            if doc_text == pos_doc_text:
                continue        
            else: 
                other_rel_5_4_doc_text_list.append(doc_text)
                
        if len(other_rel_5_4_doc_text_list) < 5: continue
            
        try:
            if len(k) < 4:
                continue            
        except: # (AttributeError or TypeError):
            print(q_text)
            #break
                    
        formatted_data_structure['query_text'].append(k)
        formatted_data_structure['doc_text'].append(pos_doc_text)
        formatted_data_structure['feedback_doc1_text'].append(other_rel_5_4_doc_text_list[0])
        formatted_data_structure['feedback_doc2_text'].append(other_rel_5_4_doc_text_list[1])
        formatted_data_structure['feedback_doc3_text'].append(other_rel_5_4_doc_text_list[2])
        formatted_data_structure['feedback_doc4_text'].append(other_rel_5_4_doc_text_list[3])
        formatted_data_structure['feedback_doc5_text'].append(other_rel_5_4_doc_text_list[4])


    # create a pandas table from the created data
    formatted_data_df = pd.DataFrame(formatted_data_structure, columns=['query_text', 'doc_text', 
                                                                        'feedback_doc1_text', 
                                                                        'feedback_doc2_text',
                                                                        'feedback_doc3_text', 
                                                                        'feedback_doc4_text', 
                                                                        'feedback_doc5_text'])
    print(formatted_data_df.shape)
        


    # write the data files into disk
    write_filename = Path(write_data_dir) / ('final_pos_clirmatrix_%s_bi139-full_en_%s.csv' % (lang, split))
    formatted_data_df[['query_text','doc_text']].to_csv(write_filename, index=False)

    # baseline ANCE alone data
    write_filename = Path(write_data_dir) / ('final_prf1_clirmatrix_%s_bi139-full_en_%s.csv' % (lang, split))
    formatted_data_df[['query_text','doc_text','feedback_doc1_text']].to_csv(write_filename, index=False)

    # baseline ANCE alone data
    write_filename = Path(write_data_dir) / ('final_prf2_clirmatrix_%s_bi139-full_en_%s.csv' % (lang, split))
    formatted_data_df[['query_text','doc_text','feedback_doc1_text','feedback_doc2_text']].to_csv(write_filename, index=False)

    # baseline ANCE alone data
    write_filename = Path(write_data_dir) / ('final_prf3_clirmatrix_%s_bi139-full_en_%s.csv' % (lang, split))
    formatted_data_df.to_csv(write_filename, index=False)

    # baseline ANCE alone data
    write_filename = Path(write_data_dir) / ('final_prf4_clirmatrix_%s_bi139-full_en_%s.csv' % (lang, SPLIT))
    formatted_data_df[['query_text','doc_text','feedback_doc1_text','feedback_doc2_text', 'feedback_doc3_text', 'feedback_doc4_text']].to_csv(write_filename, index=False)

    # baseline ANCE alone data
    write_filename = Path(write_data_dir) / ('final_prf5_clirmatrix_%s_bi139-full_en_%s.csv' % (lang, split))
    formatted_data_df.to_csv(write_filename, index=False)
            

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Specify command line arguments.
    parser.add_argument(
        '--write_data_dir', type=str,
        required=True,
        help="Write output directory."
        )
    parser.add_argument(
        '--data_root', type=str,
        required=True,
        help="Source data directory."
        )
    parser.add_argument(
        '--lang', type=str,
        required=True,
        help="Language of interest."
        )
    parser.add_argument(
        '--split', type=str,
        required=False,
        help="Data split to download (train, val, test).",
        default='train'
        )

    parser.add_argument(
        '--pos_rel', type=int,
        required=False,
        help="Relevance label for positive pairs.",
        default=6
        )
    parser.add_argument(
        '--prf1_rel', type=int,
        required=False,
        help="Relevance label for prf-1 pairs.",
        default=5
        )
    parser.add_argument(
        '--prf2_rel', type=int,
        required=False,
        help="Relevance label for prf-2 pairs.",
        default=4
        )
    parser.add_argument(
        '--prf3_rel', type=int,
        required=False,
        help="Relevance label for prf-3 pairs.",
        default=3
        )

    # Parse command line arguments.
    args = parser.parse_args()

    #prepare_data(**args)
    prepare_data(write_data_dir=args.write_data_dir, pos_rel=args.pos_rel, 
        prf1_rel=args.prf1_rel, prf2_rel=args.prf2_rel, prf3_rel=args.prf3_rel, 
        data_root=args.data_root, lang=args.lang, split=args.split)

