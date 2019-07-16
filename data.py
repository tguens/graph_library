#%%
import torch 
import os.path as osp
import torch_geometric as geometric
import os 
import pickle
from torchtext import data
import torchtext
import itertools 
import time
from tqdm import tqdm 

class GraphDataset(torch.utils.data.Dataset):
    """ Graph Dataset for the inclusion of KG data to NLP tasks.
    More flexible tha torhc_geometic and supports the interface to torch_geometric.
    Corresponds to the graphs scanned on the document.   
    """
    #TODO-Create a collate_fn to load the graph by batch, directly from the batcher
    #and increase the number of worlers to parallelize and fasten the graph generation. 
    #TODO-Interface with torch_geometric

    def __init__(self, raw_path, prepro_path, vocab=None, string_names=False):
        super(GraphDataset, self).__init__()
        if not osp.exists(raw_path):
            raise ImportError("No raw graph found.")
        else: 
            print('Found {} Raw Graphs'.format(len(os.listdir(raw_path))))
        if not osp.exists(prepro_path):
            print('No folder at the prepro_path specified. Creating one')
            os.makedirs(prepro_path)
        else:
            print('Found {} Prepro Graphs'.format(len(os.listdir(prepro_path))))
        self.raw_path = raw_path
        self.prepro_path = prepro_path
        if not string_names: #It means that we want the ids to be int
            self.list_ids = [int(idx) for idx in os.listdir(raw_path) if idx !='.ipynb_checkpoints']
        else: #we want the ids to be str
            self.list_ids = [name for name in os.listdir(raw_path) if name!='.ipynb_checkpoint'] 
        self.vocab = vocab
        self.prepro_graphs = {}
        self.sparse_graphs = {}
        
    def __len__(self):
        return len(self.list_ids)

    def __getitem__(self, index):
        if isinstance(index, int):
            try:
                return self.get_sparse([index])
            except:
                return self.get_sparse( self.list_ids[index] )
        else:
            return index

    def get_raw(self, index):
        """Returns a dict with keys: 'doc_nodes', 'doc_edges'
        Not putting those raw graphs in memory unless needed. """
        assert(index in self.list_ids)
        path = osp.join(self.raw_path, str(index))
        with open(path, 'rb') as reader:
            graph = pickle.load(reader)
        return graph

    def _get_prepro(self, raw_graph_id, key='doc'):
        """
        A prepro graph is represented by a tuple:
        list of nodes, doc_indices_strat_slice, doc_indices_end_slice, doc_E_tensor_slice)
        """
        try: 
            assert (key in ('doc','query'))
        except:
            raise KeyError("A key ('doc' or 'query') must be specified")

        if raw_graph_id in self.prepro_graphs.keys():
            #self.prepro_graphs[raw_graph_id]
            #print('Loaded from dict')
            pass
        elif osp.exists(osp.join(self.prepro_path,str(raw_graph_id)) +'.pt'):
            graph = torch.load(osp.join(self.prepro_path, str(raw_graph_id))+'.pt')
            self.prepro_graphs[raw_graph_id] = graph
            #print('Loaded from disk')
        else: #Not in memory nor on disk. Need to load and construct the indices tensors.
            raw_graph = self.get_raw(raw_graph_id)
            key_nodes = key+'_nodes'
            key_edges = key+'_edges'
            if isinstance(raw_graph, tuple): #We need to turn it into a dictionnary
                raw_graph= {key_nodes:raw_graph[0], key_edges:raw_graph[1]} 
            doc_nodes_len = len(raw_graph[key_nodes])
            doc_edges_len = len(raw_graph[key_edges])
    
                
            doc_E_tensor = torch.LongTensor(doc_edges_len)
            doc_indices_start_slice = torch.LongTensor(doc_edges_len) #not really needed
            doc_indices_end_slice = torch.LongTensor(doc_edges_len) #same
            doc_E_tensor_slice = torch.LongTensor(doc_edges_len) #same
            for j, edge in enumerate(raw_graph[key_edges]):
            #print(j, edge, len(edge))
                doc_indices_start_slice[j] = edge[0][0]
                doc_indices_end_slice[j] = edge[0][1]
                doc_E_tensor_slice[j] = edge[1]      
            self.prepro_graphs[raw_graph_id]  = raw_graph['doc_nodes'], doc_indices_start_slice, doc_indices_end_slice, doc_E_tensor_slice

            torch.save(self.prepro_graphs[raw_graph_id], osp.join(self.prepro_path, str(raw_graph_id))+'.pt')
            print('Constructed and saved on disk')
        return self.prepro_graphs[raw_graph_id]

    def get_sparse(self, graph_list_ids):
        """A Graph is represented  as tuple, where 0->nodes.
        1-> doc_indices_start_slice, 2->doc_indices_end_slice
        3-> doc_E_tensor_slice
        List of graph ids in the batch as input."""
        assert (isinstance(graph_list_ids, list) or
         isinstance(graph_list_ids, torch.Tensor))
        if isinstance(graph_list_ids, torch.Tensor):
            #print('Tensor->tolist')
            graph_list_ids = graph_list_ids.tolist()
        #doc_graphs = [self.get_raw(index) for index in graph_list_ids]
        doc_graphs = [self._get_prepro(idx) for idx in graph_list_ids]
        #doc_nodes_batch = [doc_node for x in 
        #                           doc_graphs for doc_node in x['doc_nodes']]
        doc_nodes_batch = [node for graph in doc_graphs for node in graph[0]]
        #doc_nodes_numbers = [len(x['doc_nodes']) for x in doc_graphs]
        #doc_edges_numbers = [len(x['doc_edges']) for x in doc_graphs]
        doc_nodes_numbers = [len(graph[0]) for graph in doc_graphs]
        doc_edges_numbers = [graph[1].size(0) for graph in doc_graphs]
        #doc_nodes_len = sum(doc_nodes_numbers)
        #doc_edges_len = sum(doc_edges_numbers)
        doc_nodes_len = sum(doc_nodes_numbers)
        doc_edges_len = sum(doc_edges_numbers)
        
        doc_indices_start = torch.LongTensor(2, doc_edges_len)
        doc_indices_start[0] = torch.tensor(range(doc_edges_len))

        doc_indices_end = torch.LongTensor(2, doc_edges_len)
        doc_indices_end[0] = torch.tensor(range(doc_edges_len))

        doc_values_start = torch.ones(doc_edges_len, dtype=torch.float)
        doc_values_end = torch.ones(doc_edges_len, dtype=torch.float)

        doc_E_tensor = torch.LongTensor(doc_edges_len)
        doc_edges_counter = 0
    
        #doc_built_graphs_dict = {}
        for i, sample in enumerate(doc_graphs):
            #doc_graph_id = sample["doc_graph_id"]
            doc_graph_id = graph_list_ids[i]
            doc_nodes, doc_indices_start_slice, doc_indices_end_slice, doc_E_tensor_slice = self.prepro_graphs[doc_graph_id]

            doc_indices_start[1, doc_edges_counter:
                                         doc_edges_counter+doc_edges_numbers[i]] = doc_indices_start_slice
            doc_indices_end[1, doc_edges_counter:
                                       doc_edges_counter+doc_edges_numbers[i]] = doc_indices_end_slice
            doc_E_tensor[doc_edges_counter:
                                 doc_edges_counter+doc_edges_numbers[i]] = doc_E_tensor_slice

            doc_edges_counter += doc_edges_numbers[i]

        doc_E_start_sparse = torch.sparse.FloatTensor(doc_indices_start, 
                                                              doc_values_start,
                                                              torch.Size([doc_edges_len, 
                                                                          doc_nodes_len]))
        doc_E_end_sparse = torch.sparse.FloatTensor(doc_indices_end, doc_values_end,
                                                    torch.Size([doc_edges_len, doc_nodes_len]))
        return doc_nodes_batch, (doc_E_start_sparse, doc_E_end_sparse, doc_E_tensor), doc_nodes_numbers
    
    def to_batch(self, graph_list_ids):
        """A Graph is represented  as tuple, where 0->nodes.
        1-> doc_indices_start_slice, 2->doc_indices_end_slice
        3-> doc_E_tensor_slice
        List of graph ids in the batch as input."""
        assert (isinstance(graph_list_ids, list) or
         isinstance(graph_list_ids, torch.Tensor))
        if isinstance(graph_list_ids, torch.Tensor):
            #print('Tensor->tolist')
            graph_list_ids = graph_list_ids.tolist()
        #doc_graphs = [self.get_raw(index) for index in graph_list_ids]
        doc_graphs = [self._get_prepro(idx) for idx in graph_list_ids]
        #doc_nodes_batch = [doc_node for x in 
        #                           doc_graphs for doc_node in x['doc_nodes']]
        doc_nodes_batch = [node for graph in doc_graphs for node in graph[0]]
        #doc_nodes_numbers = [len(x['doc_nodes']) for x in doc_graphs]
        #doc_edges_numbers = [len(x['doc_edges']) for x in doc_graphs]
        doc_nodes_numbers = [len(graph[0]) for graph in doc_graphs]
        doc_edges_numbers = [graph[1].size(0) for graph in doc_graphs]
        #doc_nodes_len = sum(doc_nodes_numbers)
        #doc_edges_len = sum(doc_edges_numbers)
        doc_nodes_len = sum(doc_nodes_numbers)
        doc_edges_len = sum(doc_edges_numbers)
        
        doc_indices_start = torch.LongTensor(2, doc_edges_len)
        doc_indices_start[0] = torch.tensor(range(doc_edges_len))

        doc_indices_end = torch.LongTensor(2, doc_edges_len)
        doc_indices_end[0] = torch.tensor(range(doc_edges_len))

        doc_values_start = torch.ones(doc_edges_len, dtype=torch.float)
        doc_values_end = torch.ones(doc_edges_len, dtype=torch.float)

        doc_E_tensor = torch.LongTensor(doc_edges_len)
        doc_edges_counter = 0
    
        #doc_built_graphs_dict = {}
        for i, sample in enumerate(doc_graphs):
            #doc_graph_id = sample["doc_graph_id"]
            doc_graph_id = graph_list_ids[i]
            doc_nodes, doc_indices_start_slice, doc_indices_end_slice, doc_E_tensor_slice = self.prepro_graphs[doc_graph_id]

            doc_indices_start[1, doc_edges_counter:
                                         doc_edges_counter+doc_edges_numbers[i]] = doc_indices_start_slice
            doc_indices_end[1, doc_edges_counter:
                                       doc_edges_counter+doc_edges_numbers[i]] = doc_indices_end_slice
            doc_E_tensor[doc_edges_counter:
                                 doc_edges_counter+doc_edges_numbers[i]] = doc_E_tensor_slice

            doc_edges_counter += doc_edges_numbers[i]

        #doc_E_start_sparse = torch.sparse.FloatTensor(doc_indices_start, 
        #                                                      doc_values_start,
        #                                                      torch.Size([doc_edges_len, 
        #                                                                  doc_nodes_len]))
        #doc_E_end_sparse = torch.sparse.FloatTensor(doc_indices_end, doc_values_end,
        #                                            torch.Size([doc_edges_len, doc_nodes_len]))
        #return doc_nodes_batch, (doc_E_start_sparse, doc_E_end_sparse, doc_E_tensor), doc_nodes_numbers
        V_tokenized = torch.tensor([self.vocab.stoi[word] for word in doc_nodes_batch])
        nodes_numbers_tensor = torch.tensor(doc_nodes_numbers)
        return V_tokenized, doc_indices_start, doc_indices_end, doc_E_tensor, nodes_numbers_tensor
    
    def build_vocab(self, path_to_embedding='glove'):
        vectors_init_graph = torchtext.vocab.Vectors(path_to_embedding)
        #if path_to_embedding=='glove->glove'torchtext.vocab.GloVe() to download in cache
        begin = time.time()
        nodes = []
        for graph_id in tqdm(self.list_ids):
            #raw_graph = self.get_raw(graph_id) #Naive-we can build the prepro at the same time
            raw_graph = self._get_prepro(graph_id)
            if isinstance(raw_graph, dict):
                nodes.append(raw_graph['doc_nodes'])
            elif isinstance(raw_graph, tuple):
                nodes.append(raw_graph[0])
        #nodes_compare = list(itertools.chain(*nodes))
        #text_graph_compare = torchtext.data.Field(tokenize='spacy')
        #text_graph_compare.build_vocab(nodes_compare, vectors=vectors_init_graph)
        #print('Number of nodes: ', len(nodes))
        text_graph = torchtext.data.Field(tokenize='spacy')
        text_graph.build_vocab(nodes, vectors=vectors_init_graph)
        print("Graph Vocab built in {}".format(time.time() - begin))
        self.vocab = text_graph.vocab
        return text_graph
    
    def save_vocab(self, path_to_save):
        return None


def build_vocab_graph_squad(path_to_embedding, is_naive_bert):
    """ Slightly different from the vocab built for SAN.
    Previously: len(d_nodes) = nb. of paragraphs.
    Now: len(d_nodes) = nb. of QAs.
    """
    if is_naive_bert:
        vectors_init_graph = None
    else:
        vectors_init_graph = vocab.Vectors(path_to_embedding)
    #def _create_vocab():
    beg = time.time()
    q_nodes = []        
    d_nodes = []

    for idx, ex in enumerate(train_examples+dev_examples):
        q_nodes.append(ex.graphs['query_nodes'])
        d_nodes.append(ex.graphs['doc_nodes']) 

    end = time.time()
    print('Create the total query/doc nodes lists in {}:'.format(end-beg)) #Takes about 6 minutes...
    merged_nodes = q_nodes+d_nodes

    text_graph = data.Field(tokenize='spacy')
    text_graph.build_vocab(merged_nodes, vectors=vectors_init_graph)
    #pretrained_nodes_embedding = text_graph.vocab.vectors
    print("Graph Vocab built- {}".format(time.time()-end))

    if is_naive_bert:
        path_to_stoi = path_to_embedding + '/STOI_251081_768_bert-base-uncased.pick'
        path_to_tensor = path_to_embedding + '/tensor_251081_768_bert-base-uncased.pt'
        with open(path_to_stoi, 'rb') as reader:
            stoi = pickle.load(reader)
        vocab_tensor = torch.load(path_to_tensor)
        text_graph.vocab.set_vectors(stoi=stoi,vectors=vocab_tensor, dim=768)
        pretrained_nodes_embedding = text_graph.vocab.vectors
    else:
        pretrained_nodes_embedding = text_graph.vocab.vectors


    return (q_nodes,d_nodes,merged_nodes), text_graph, pretrained_nodes_embedding 

#TODO-Add Generate Graph
def generate_graph():
    return None

def split_for_multi_gpu(doc_ids, graph_dataset, devices):
    """Based upon the to_batch() of GraphDataset
    """
    assert isinstance(doc_ids, torch.tensor)

    if isinstance(devices, list):
        num_devices = len(devices)
    elif isinstance(devices, int):
        num_devices = devices
    else:
        raise ValueError('Provide a list or integer.')
    
    split = int(doc_ids.size(0)/num_devices) #batch_size/num_devices
    split_doc_list = doc_id.split(split)
    split_G = tuple(graph_dataset.to_batch(split_doc) for split_doc in split_doc_list)
    
    if num_devices>1:
        assert len(split_G) == num_devices
    return split_G

#TODO-Make sure it works well 
def build_vocab_from_graphs(train_graphs, dev_graphs, path_to_embedding):
    """
    Build the vocab from two Graph DS=train and dev are splitted.
    """
    vectors_init_graph = torchtext.vocab.Vectors(path_to_embedding)
    begin = time.time()
    nodes = []
    for graph_id in tqdm(train_graphs.list_ids, total=len(train_graphs), desc='Train Graphs'):
        prepro_graph = train_graphs._get_prepro(graph_id)
        nodes.append(prepro_graph[0])
    for graph_id in tqdm(dev_graphs.list_ids, total=len(dev_graphs), desc='Dev Graphs'):
        prepro_graph = dev_graphs.get_prepro(graph_id)
        nodes.append(prepro_graph[0])

#TODO-change the collate_fn to load graphs directly. 
def data_loader():
    return None




#%%
