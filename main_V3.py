import torch
from models_V3 import NetCapClassifier, NetCapRegressorEnsemble
from SRAM_dataset_gbatch import SRAMDataset, SRAMDatasetList
# from datetime import datetime
# from utils.circuits2graph import run_struct2g
from train_gbatch_V3 import train
# from transformers import BertTokenizer, BertModel, pipeline
from transformers import AutoTokenizer, AutoModel
# from sentence_transformers import SentenceTransformer
# from transformers import AutoTokenizer, AutoModel, pipeline
import os

if __name__ == '__main__':
    # tokenizer = BertTokenizer.from_pretrained("/data1/shenshan/hugface_model/bert-base-uncased")
    # model = BertModel.from_pretrained("/data1/shenshan/hugface_model/bert-base-uncased")
    # print('loaded tokenizer and model')
    # sentence = '.SUBCKT INV XINV/x1/x10/NET4_OUT'
    # model = SentenceTransformer('/data1/shenshan/huggingface_models/all-MiniLM-L6-v2')
    # sentences = ['This framework generates embeddings for each input sentence',
    #     'Sentences are passed as a list of string.', 
    #     'The quick brown fox jumps over the lazy dog.']
    #Sentences are encoded by calling model.encode()
    # sentence_embeddings = model.encode(sentences)

    # tokenizer = AutoTokenizer.from_pretrained("/data1/shenshan/huggingface_models/all-MiniLM-L6-v2")
    # model = AutoModel.from_pretrained("/data1/shenshan/huggingface_models/all-MiniLM-L6-v2")
    # #Tokenize sentences
    # encoded_input = tokenizer(sentences, padding=True, truncation=True, max_length=128, return_tensors='pt')
    # print(encoded_input)

    # #Compute token embeddings
    # with torch.no_grad():
    #     model_output = model(**encoded_input)

    # #Print the embeddings
    # for sentence, embedding in zip(sentences, sentence_embeddings):
    #     print("Sentence:", sentence)
    #     print("Embedding:", embedding)
    #     print("")

    # token_ids = tokenizer.encode("This is a sample text to test the tokenizer.")
    # print( token_ids )
    # print( tokenizer.convert_ids_to_tokens( token_ids ) )
    # wrapped_input = tokenizer(sentence, max_length=15, add_special_tokens=True, truncation=True, 
    #                       padding='max_length', return_tensors="pt")
    # print(wrapped_input)
    # output = model(**wrapped_input)
    # last_hidden_state, pooler_output = output[0], output[1]
    # print(last_hidden_state.shape)
    # print(pooler_output.shape)

    # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    # model.resize_token_embeddings(len(tokenizer))
    # pipe = pipeline('feature-extraction', model=model, tokenizer=tokenizer)

    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # model = TFBertModel.from_pretrained("bert-base-cased")
    # text_1 = "Replace me by any text you'd like."
    # encoded_text = tokenizer(text_1, return_tensors='tf')
    # output = model(encoded_text)

    # device = torch.device('cuda:6' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cuda:3') 
    dataset = SRAMDatasetList(nameList=['ultra_8T', 'sandwich', 'sram_sp_8192w'], #   ], #'ssram',
                              device=device, test_ds=False)
    datasetTest = SRAMDatasetList(nameList=['ssram', ], device=device, #'array_128_32_8t',
                                  test_ds=True, featMax=dataset.featMax)
    # raw_dir = '/data1/shenshan/SPF_examples_cdlspf/Python_data/'
    # dataset = SRAMDataset(name='array_128_32_8t', raw_dir=raw_dir, test=False)

    # tokenizer = AutoTokenizer.from_pretrained("/data1/shenshan/huggingface_models/all-MiniLM-L6-v2")
    modelst= AutoModel.from_pretrained("/data1/shenshan/huggingface_models/all-MiniLM-L6-v2").to(torch.device('cuda:0') )

    linear_dict = {'device': [dataset._d_feat_dim, 56, 56], 
                   'inst':   [dataset._i_feat_dim, 56, 56], 
                   'net':    [dataset._n_feat_dim, 56, 56]}
    model = NetCapClassifier(num_classes=dataset._num_classes, proj_dim_dict=linear_dict, 
                             gnn='sage-mean', has_l2norm=True, has_bn=False, dropout=0.1, 
                             device=device)
    # modelr = MLPRegressor(num_classes=dataset._num_classes, 
    #                          reg_dim_list=[64+dataset._n_feat_dim+1, 128, 128, 64, 1],
    #                          has_l2norm=False, 
    #                          has_bn=True, device=device)
    # modelr = []
    # modelr.append(NetCapRegressor(num_classes=dataset._num_classes, proj_dim_dict=linear_dict, 
    #                         gnn='sage-mean', has_l2norm=False, has_bn=True, dropout=0.1, 
    #                         device=device))
    # modelr.append(NetCapRegressor(num_classes=dataset._num_classes, proj_dim_dict=linear_dict, 
    #                         gnn='sage-mean', has_l2norm=False, has_bn=True, dropout=0.1, 
    #                         device=device))
    modelens = NetCapRegressorEnsemble(num_classes=dataset._num_classes, proj_dim_dict=linear_dict, 
                                       gnn='sage-mean', has_l2norm=True, has_bn=False, dropout=0.1, 
                                       device=device)
    print("PID =", os.getpid())
    train(dataset, datasetTest, model, modelens, modelst)