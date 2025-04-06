import sys
from random import shuffle, seed
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from time import sleep
import pickle


class PromptSelector:
    """ Steps:
    1, load the data in a list
    2, Split the data into training and testing
    3, Do selection on the training
    4, Testing the performance on the testing
    """

    def __init__(self, prompts: list, labels: list = None):
        self.prompts = prompts
        self.train_data = []
        self.test_data = []
        self.train_labels = []
        self.test_labels = []
        self.labels = labels

    def split_data(self, ratio: float = 0.8, sort_label: bool = False):
        """ Split the data into training and testing
        """
        seed(42)
        n = len(self.prompts)
        indexes = list(range(n))
        shuffle(indexes)
        train_n = int(n * ratio)
        self.train_data = self.prompts[:train_n]
        self.test_data = self.prompts[train_n:]
        self.train_labels = [self.labels[i] for i in indexes[:train_n]]
        self.test_labels = [self.labels[i] for i in indexes[train_n:]]
        if sort_label:
            sorted_idx = sorted(range(len(self.train_labels)), key=lambda k: self.train_labels[k])
            self.train_data = [self.train_data[i] for i in sorted_idx]
            self.train_labels = [self.train_labels[i] for i in sorted_idx]

            sorted_idx = sorted(range(len(self.test_labels)), key=lambda k: self.test_labels[k])
            self.test_data = [self.test_data[i] for i in sorted_idx]
            self.test_labels = [self.test_labels[i] for i in sorted_idx]

    def calculate_embeddings(self):
        """ Calculate the embeddings for the prompts
        """
        model_name = "distilgpt2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)

        res = []

        with tqdm(total=len(self.train_data)) as pbar:
            for text in self.train_data:
                inputs = tokenizer(text, return_tensors="pt")
                with torch.no_grad():
                    outputs = model(**inputs)
                    token_embeddings = outputs.last_hidden_state
                    token_embeddings_mean = token_embeddings.mean(dim=1)
                    res.append(token_embeddings_mean)
                if pbar.n % 10 == 0:
                    sleep(0.1)
                    pbar.update(10)

        return res
    
    def calc_similarity_matrix(self, tensor: torch.Tensor):
        """ Calculate the similarity matrix
        """
        tensor_nm = tensor / tensor.norm(dim=1, keepdim=True)
        similarity_matrix = torch.matmul(tensor_nm, tensor_nm.T)
        return similarity_matrix


if __name__ == "__main__":
    # Calculate the embeddings for the prompts

    prompts = []
    with open("prompts/iris_promp.txt", 'r') as f:
        for line in f:
            prompts.append(line.strip())

    labels = []
    with open("prompts/iris_promp.label", 'r') as f:
        for line in f:
            labels.append(line.strip())
    
    ps = PromptSelector(prompts, labels)
    ps.split_data(0.8, sort_label=True)  #sort the data for better visualization
    embeddings = ps.calculate_embeddings()

    stacked_embeddings = torch.stack(embeddings)
    stacked_embeddings = stacked_embeddings.squeeze(1)
    print(stacked_embeddings.shape)

    similarity_matrix = ps.calc_similarity_matrix(stacked_embeddings)

    # Form the data
    processed_data = {
        "train_data": ps.train_data,
        "test_data": ps.test_data,
        "train_labels": ps.train_labels,
        "test_labels": ps.test_labels,
        "train_embedding": stacked_embeddings,
        "similarity_matrix": similarity_matrix
    }

    with open("data/iris_processed_data.pkl", 'wb') as f:
        pickle.dump(processed_data, f)
