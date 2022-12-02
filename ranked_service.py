from flask import Flask, request
from langdetect import detect
from typing import Dict, List, Tuple, Union, Callable

import numpy as np
import pandas as pd
import torch.nn.functional as F

import torch
import json
import string
import nltk
import faiss
import os


class GaussianKernel(torch.nn.Module):
    def __init__(self, mu: float = 1., sigma: float = 1.):
        super().__init__()
        self.mu = mu
        self.sigma = sigma

    def forward(self, x):
        return torch.exp(
            -0.5 * ((x - self.mu) ** 2) / (self.sigma ** 2)
        )


class KNRM(torch.nn.Module):
    def __init__(self, embedding_matrix: np.ndarray,
                 freeze_embeddings: bool = True, kernel_num: int = 21,
                 sigma: float = 0.1, exact_sigma: float = 0.001,
                 out_layers: List[int] = []):
        super().__init__()
        self.embeddings = torch.nn.Embedding.from_pretrained(
            torch.FloatTensor(embedding_matrix),
            freeze=freeze_embeddings,
            padding_idx=0
        )

        self.kernel_num = kernel_num
        self.sigma = sigma
        self.exact_sigma = exact_sigma
        self.out_layers = out_layers

        self.kernels = self._get_kernels_layers()

        self.mlp = self._get_mlp()

        self.out_activation = torch.nn.Sigmoid()

    def _get_kernels_layers(self) -> torch.nn.ModuleList:
        kernels = torch.nn.ModuleList()
        for i in range(self.kernel_num):
            mu = 1. / (self.kernel_num - 1) + (2. * i) / (
                self.kernel_num - 1) - 1.0
            sigma = self.sigma
            if mu > 1.0:
                sigma = self.exact_sigma
                mu = 1.0
            kernels.append(GaussianKernel(mu=mu, sigma=sigma))
        return kernels

    def _get_mlp(self) -> torch.nn.Sequential:
        out_cont = [self.kernel_num] + self.out_layers + [1]
        mlp = [
            torch.nn.Sequential(
                torch.nn.Linear(in_f, out_f),
                torch.nn.ReLU()
            )
            for in_f, out_f in zip(out_cont, out_cont[1:])
        ]
        mlp[-1] = mlp[-1][:-1]
        return torch.nn.Sequential(*mlp)

    def forward(self, input_1: Dict[str, torch.Tensor], input_2: Dict[str, torch.Tensor]) -> torch.FloatTensor:
        logits_1 = self.predict(input_1)
        logits_2 = self.predict(input_2)

        logits_diff = logits_1 - logits_2

        out = self.out_activation(logits_diff)
        return out

    def _get_matching_matrix(self, query: torch.Tensor, doc: torch.Tensor) -> torch.FloatTensor:
        # shape = [B, L, D]
        embed_query = self.embeddings(query.long())
        # shape = [B, R, D]
        embed_doc = self.embeddings(doc.long())

        # shape = [B, L, R]
        matching_matrix = torch.einsum(
            'bld,brd->blr',
            F.normalize(embed_query, p=2, dim=-1),
            F.normalize(embed_doc, p=2, dim=-1)
        )
        return matching_matrix

    def _apply_kernels(self, matching_matrix: torch.FloatTensor) -> torch.FloatTensor:
        KM = []
        for kernel in self.kernels:
            # shape = [B]
            K = torch.log1p(kernel(matching_matrix).sum(dim=-1)).sum(dim=-1)
            KM.append(K)

        # shape = [B, K]
        kernels_out = torch.stack(KM, dim=1)
        return kernels_out

    def predict(self, inputs: Dict[str, torch.Tensor]) -> torch.FloatTensor:
        query, doc = inputs['query'], inputs['document']
        # shape = [B, L, R]
        matching_matrix = self._get_matching_matrix(query, doc)
        # shape [B, K]
        kernels_out = self._apply_kernels(matching_matrix)
        # shape [B]
        out = self.mlp(kernels_out)
        return out


class Solution:
    def __init__(self):
        self.mlp_state = torch.load(os.environ["MLP_PATH"])
        self.emb_knrm = torch.load(os.environ["EMB_PATH_KNRM"])['weight']
        with open(os.environ["VOCAB_PATH"], "r", encoding="utf-8") as vocab:
            self.vocab = json.load(vocab)
        # self.mlp_state = torch.load("knrm_mlp.bin")
        # self.emb_knrm = torch.load("knrm_emb.bin")['weight']
        # with open('vocab.json', "r", encoding="utf-8") as vocab:
        #     self.vocab = json.load(vocab)
        self.index = None
        self.docs = None

    def hadle_punctuation(self, inp_str: str) -> str:
        for punct in string.punctuation:
            inp_str = inp_str.replace(punct, ' ')
        return inp_str

    def simple_preproc(self, inp_str: str) -> List[str]:
        inp_str = self.hadle_punctuation(inp_str).lower()
        out_list = nltk.word_tokenize(inp_str)
        return out_list

    def make_vector(self, inp_str):
        value_list = self.simple_preproc(inp_str)
        vector = self.emb_knrm[0]
        for v in value_list:
            try:
                vector += self.emb_knrm[self.vocab[v]]
            except KeyError:
                vector += self.emb_knrm[self.vocab['OOV']]
        vector /= len(value_list) + 0.0001
        return vector.tolist()


sol = Solution()
knrm = KNRM(sol.emb_knrm)
knrm.mlp.load_state_dict(sol.mlp_state)

app = Flask(__name__)


@app.route('/ping')
def ping():
    return {'status': 'ok'}


@app.route('/query', methods=['POST'])
def query():
    if sol.index is None:
        return {'status': 'FAISS is not initialized!'}

    # queries = json.loads(request.json)['queries']
    queries = query_ex['queries']
    # queries = request.json['queries']
    lang_check_list = []
    suggestions_list = []

    for q in queries:
        if detect(q) != 'en':
            lang_check_list.append(False)
            suggestions_list.append(None)
        else:
            lang_check_list.append(True)
            q_vector = np.array([sol.make_vector(q)]).astype('float32')
            topf = 10
            if sol.index.ntotal < topf:
                topf = sol.index.ntotal
            D, I = sol.index.search(q_vector, topf)
            # print(D)
            # print(I)

            query_ind = []
            for word in sol.simple_preproc(q):
                try:
                    query_ind.append(sol.vocab[word])
                except KeyError:
                    query_ind.append(sol.vocab['OOV'])

            preds = []

            for d in I[0]:
                doc_ind = []
                for w in sol.simple_preproc(list(sol.docs.values())[d]):
                    try:
                        doc_ind.append(sol.vocab[w])
                    except KeyError:
                        doc_ind.append(sol.vocab['OOV'])
                input_dict = {'query': torch.IntTensor([query_ind]),
                              'document': torch.IntTensor([doc_ind])}
                one_pred = knrm.predict(input_dict)
                preds.append(one_pred)

            suggestion_list = []

            topn = 10
            if sol.index.ntotal < 10:
                topn = sol.index.ntotal

            preds = torch.FloatTensor(preds)
            for pred in I[0][preds.argsort(descending=True).tolist()][:topn]:
                suggestion_list.append((list(sol.docs.keys())[pred],
                                        list(sol.docs.values())[pred]))

            suggestions_list.append(suggestion_list)

    # return {'lang_check': lang_check_list,
    #         'suggestions': suggestions_list}
    cnt = 0
    for i in range(len(suggestions_list)):
        print(queries[i], suggestions_list[i])
        if suggestions_list[i] is None:
            cnt += 1
            continue
        for j in range(len(suggestions_list[i])):
            if queries[i] == suggestions_list[i][j][1]:
                cnt += 1
    print(cnt, '/', len(suggestions_list))
    print(cnt/len(suggestions_list))


@app.route('/update_index', methods=['POST'])
def update_index():
    # docs = json.loads(request.json)['documents']
    docs = docs_ex['documents']
    # docs = request.json['documents']
    sol.docs = docs
    vectors = []
    for key, value in docs.items():
        vector = sol.make_vector(value)
        vectors.append(vector)
    k = int(len(docs) ** 0.05)
    index = faiss.index_factory(len(sol.emb_knrm[0]), 'IVF{}_HNSW32,Flat'.format(k), faiss.METRIC_L2)
    index.train(np.array(vectors).astype('float32'))
    index.add(np.array(vectors).astype('float32'))
    sol.index = index
    return {'status': 'ok',
            'index_size': sol.index.ntotal}


# if __name__ == '__main__':
#     app.run(debug=True, port=11000)

df = pd.read_csv('df_train.csv', index_col='id')[:400000]
query_ex = {"queries": df['documents'].sample(1000, random_state=42).tolist()}
df.index = df.index.astype('str')
docs_ex = df.to_dict('dict')

update_index()

print(query())


