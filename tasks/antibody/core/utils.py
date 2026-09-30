import os

import numpy as np
import torch


def batch_iterator(data1, step=8):
    size = len(data1)
    for i in range(0, size, step):
        yield data1[i:min(i + step, size)]


class BERTFeatures:
    """Compute BERT Features"""

    def __init__(self, model, tokeniser):
        AAs = 'ACDEFGHIKLMNPQRSTVWY'
        self.AA_to_idx = {aa: i for i, aa in enumerate(AAs)}
        self.idx_to_AA = {value: key for key, value in self.AA_to_idx.items()}
        self.model = model
        self.tokeniser = tokeniser

    def compute_features(self, x1):
        assert x1.ndim == 2
        inp_device = x1.device
        self.model = self.model.to(inp_device)
        with torch.no_grad():
            x1 = [" ".join(self.idx_to_AA[i.item()] for i in x_i) for x_i in x1]
            ids1 = self.tokeniser.batch_encode_plus(x1, add_special_tokens=False, padding=True)
            input_ids1 = torch.tensor(ids1['input_ids']).to(inp_device)
            attention_mask1 = torch.tensor(ids1['attention_mask']).to(inp_device)
            reprsn1 = self.model.to(inp_device)(input_ids=input_ids1, attention_mask=attention_mask1)[0]
        return reprsn1.mean(1)


if __name__ == '__main__':
    bert_config = {'datapath': '/nfs/aiml/asif/CDRdata',
                   'path': '/nfs/aiml/asif/ProtBERT',
                   'modelname': 'OutputFinetuneBERTprot_bert_bfd',
                   'use_cuda': True,
                   'batch_size': 256
                   }
    device_ids = [2, 3]
    import glob
    import numpy as np

    os.environ['CUDA_VISIBLE_DEVICES'] = ",".join(str(id_) for id_ in device_ids)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from transformers import AutoTokenizer, \
        AutoModel

    device = torch.device("cuda" if torch.cuda.is_available() and bert_config['use_cuda'] else "cpu")
    tokeniser = AutoTokenizer.from_pretrained(f"{bert_config['path']}/{bert_config['modelname']}")
    model = AutoModel.from_pretrained(f"{bert_config['path']}/{bert_config['modelname']}").to(device)
    bert_features = BERTFeatures(model, tokeniser)

    antigens = ['1ADQ_A', '1FBI_X', '1H0D_C', '1NSN_S', '1OB1_C', '1WEJ_F', '2YPV_A', '3RAJ_A', '3VRL_C', '2DD8_S',
                '1S78_B', '2JEL_P']

    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from joblib import dump
    import pandas as pd

    for antigen in antigens:
        print(f"PCA for antigen {antigen}")
        try:
            filenames = glob.glob(f"{bert_config['datapath']}/RawBindingsMurine/{antigen}/*.txt")
            for i in range(len(filenames)):
                try:
                    if i == 0:
                        sequences = pd.read_csv(filenames[i], skiprows=1, sep='\t')
                        sequences = list(set(sequences['Slide'].dropna().values))
                    else:
                        df_i = pd.read_csv(filenames[i], skiprows=1, sep='\t')
                        df_i = list(set(df_i['Slide'].dropna().values))
                        sequences = list(set(sequences + df_i))
                except pd.errors.ParserError as err:
                    print(f"{filenames[i]} causes an error {err}")
                    continue
        except:
            continue

        if len(filenames) != 0:
            reprsns = []
            for seq_batch in batch_iterator(sequences, bert_config['batch_size']):
                seq_batch = torch.tensor([[bert_features.AA_to_idx[aa] for aa in seq] for seq in seq_batch]).to(device)
                seq_reprsn = bert_features.compute_features(seq_batch)
                reprsns.append(seq_reprsn.cpu().numpy())
                if len(reprsns) == 1000:
                    break
            reprsns = np.concatenate(reprsns, 0)
            scaler = StandardScaler()
            scaler.fit(reprsns)
            scaled_reprsns = scaler.transform(reprsns)
            pca = PCA(n_components=100)
            pca.fit(scaled_reprsns)
            results_path = f"{bert_config['datapath']}/finetune_pca"
            if not os.path.exists(results_path):
                os.makedirs(results_path)
            dump(pca, f"{results_path}/{antigen}_pca.joblib")
            dump(scaler, f"{results_path}/{antigen}_scaler.joblib")
