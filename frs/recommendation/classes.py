"""
Contains all classes used for training.
"""
import pandas as pd
from torch.utils.data import Dataset
import torch
from torch import nn

class Accumulator():
    """
    For accumulating sums over 'n' variables.

    Attributes
    data : list
        Sum of the different variables.

    Methods
    add(*args)
        Add ints to sums of the different variables.
    reset()
        Reset all sums to zero.
    """
    def __init__(self, n):
        """
        Constructs all necessary attributes.

        Parameters
        n : int
            Number of variables to keep.
        """
        self.data = [0.0] * n

    def add(self, *args):
        """
        Add ints to sums of the different variables.

        Parameters
        *args : list
            numbers to add for the different variables.
        
        Returns
        None
        """
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        """
        Reset all sums to zero.

        Returns
        None 
        """
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        """
        Called when object is indexed.

        Parameters
        idx : int
            Index of variable to get.
        
        Returns
        var_sum : sum for a variable
        """
        var_sum = self.data[idx]
        return var_sum

class Vocab():
    """
    Vocabulary of ingredients.

    Attributes
    ing2idx : dict
        Mapping of ingredients to indices.
    ings : list
        All ingredient strings.
    embs : list
        All ingredient embeddings.

    Args:
        path: Path to DataFrame with all ingredients
    """
    def __init__(self, path):
        """
        Constructs all necesssary attributes.

        Parameters
        path : str
            Path to pickled DataFrame containing ingredients and embeddings.
        
        Returns
        None
        """
        df = pd.read_pickle(path)
        self.ing2idx, self.ings, self.embs = self._build_vocab(df)

    def __len__(self):
        """
        Called when len() is called on object.
        
        Returns
        length : int
            The number of ingredients.
        """
        length = len(self.ings)
        return length
        
    def _build_vocab(self, df):
        """
        Create dictionary mapping ingredients to indices, and lists mapping indices to ingredients or embeddings.

        Parameters
        df : DataFrame
            All ingredients and embeddings.
        
        Returns
        ing2idx : dict
            Mapping of ingredients to indices.
        ings : list
            All ingredient strings.
        embs : list
            All ingredient embeddings.
        """
        vocab = [(ing, emb) for ing, emb in df.values] # unpack
        vocab.insert(0, ('<pad>', [0.0]*len(df['embedding'][0]))) # add pad
        ing2idx, ings, embs = {}, [], []
        for idx, (ing, emb) in enumerate(vocab):
            ing2idx[ing] = idx
            ings.append(ing)
            embs.append(emb)
        return ing2idx, ings, embs
    
class RNNData(Dataset):
    """
    Inherits from the PyTorch Dataset class. Creates X and Y from recipes,
    where X is the recipe except one and Y is the missing ingredient. Shuffles the recipes and retrieves vectors.

    Attributes
    ing2idx : dict
        Dictionary with ingredients as key, and indices as value.
    max_length : int
        Number of ingredients to keep per recipe.
    prepad : bool
        Pad recipes from the start if true.

    Methods
    shuffle(products)
        Shuffles the ingredients within a recipe N-1 times.
    """
    def __init__(self, df, vocab, max_length=None, prepad=True):
        """
        Constructs all necessary attributes.

        Parameters
        df : DataFrame
            Product recipes.
        vocab : Vocab
            Object with attributes mapping ingredients and embeddings.
        max_length : int
            Number of ingredients to keep per recipe.
        prepad : bool
            Pad recipes from the start if true.
        Returns
        None
        """
        # assign variables
        self.ing2idx = vocab.ing2idx
        self.max_length = max_length if max_length \
        else len(max(df['ingredients'], key=len))
        self.prepad = prepad

        # shuffle all recipes n-1 times
        self.recipes = self.shuffle(df['ingredients'])
        
    def __len__(self):
        """
        Called when len is called on object.
        
        Returns
        length : int
            The number of shuffled recipes.
        """
        length = len(self.recipes)
        return length

    def __getitem__(self, idx):
        """
        Used by iterator to get labels and target values.
        
        Parameters
        idx : int
            Index for recipe to create labels and target.
        Returns
        x : tensor
            Vector of all ingredients of the recipe except one (labels).
        y : tensor
            Vector of the missing ingredient (target).
        """
        # Cut recipe and get X,Y
        recipe = self.recipes[idx][:self.max_length + 1] # cut long recipes
        X, Y = self._split_targets(recipe) # get X (labels) and Y (target)
        padding = ['<pad>'] * (self.max_length - len(X)) # pad recipe
        X = padding + X if self.prepad else X + padding
        x = torch.tensor([self.ing2idx[ing] for ing in X]) # vectorise
        y = torch.tensor([self.ing2idx[Y]])
        return x, y

    def __repr__(self):
        """
        Called when print is called on object.

        Returns
        view : DataFrame
            First five rows of recipes.
        """
        view = self.recipes.head().to_string()
        return view
    
    def _split_targets(self, recipe):
        """
        Split a product recipe into labels and target.

        Parameters
        recipe : list
            All ingredients of one product recipe.
        
        Returns
        x : list
            All ingredients of the recipe except one (labels).
        y : str
            The missing ingredient (target).
        """
        x = recipe[:-1]
        y = recipe[-1]
        return x, y

    def shuffle(self, products):
        """
        Shuffles the ingredients within a recipe N-1 times.

        Parameters
        products : Series
            Product recipes.
        
        Returns
        products : Series
            All different combinations of product recipes.
        """
        combinations = list()
        for recipe in products:
            for i in range(len(recipe)):
                combinations.append((recipe[i:] + recipe[:i]))
        products = pd.Series(combinations)
        return products
    
class RNN(nn.Module):
    """
    Implementation of a Recurrent Neural Network taking a recipe to predict one ingredient.

    Attributes
    num_hiddens : int
        Number of hidden layers.
    num_layers : int
        Number of GRUs.
    num_directions : int
        Number of directions to use for training.
    n_classes : int
        Number of possible classes to predict (number of ingredients).
    embedding : nn.Embedding
        Embedding layer.
    rnn : nn.GRU
        GRU layer.
    fc : nn.Linear
        Fully connected layer.

    Methods
    forward(inputs, state, device)
        Called during training to get prediction.
    begin_state(device, batch_size)
        Initiates state with a tensor of zeros.
    """
    def __init__(self, vocab, num_hiddens=1, num_layers=1, dropout=0, 
                 bidirectional=False, embed_weights=False, embed_dim=4):
        """
        Constructs all necessary attributes.

        Parameters
        vocab : Vocab
            Object with attributes mapping ingredients and embeddings.
        num_hiddens : int
            Number of hidden layers.
        num_layers : int
            Number of GRUs.
        dropout : int
            Dropout probability.
        bidirectional : bool
            Consider sequence in both directions if true.
        embed_weights : bool
            Use pre-trained embeddings in embedding layer.
        embed_dim : int
            Number of embedding dimensions.

        Returns
        None
        """
        super().__init__()
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1
        self.n_classes = len(vocab)

        # define layers
        if embed_weights: 
            embeddings = torch.tensor(vocab.embs)
            self.embedding = nn.Embedding.from_pretrained(embeddings)
            embed_dim = len(embeddings[0]) # infer embedding dimensions
        else:
            self.embedding = nn.Embedding(self.n_classes, embed_dim)
        
        self.rnn = nn.GRU(embed_dim, num_hiddens, num_layers, dropout=dropout,
                          bidirectional=bidirectional, batch_first=True)
        
        self.fc = nn.Linear(num_hiddens * self.num_directions, self.n_classes)

    def forward(self, inputs, state):
        """
        Get prediction from the Recurrent Neural Network.

        Parameters
        inputs : tensor
            Vector of labels for predicting the target.
        state : tensor
            Hidden state.
        device : torch.device
            CPU or GPU.
        
        Returns
        output : tensor
            Predicted probabilities for each ingredient.
        state : tensor
            Hidden state after considering recipe.
        """
        X = self.embedding(inputs) # embed input
        Y, state = self.rnn(X, state) # make prediction
        y = Y[:, -1] # get final prediction (missing ingredient)
        output = self.fc(y)
        return output, state

    def begin_state(self, device, batch_size=1):
        """
        Initiates hidden state with a tensor of zeros.

        Parameters
        device : torch.device
            CPU or GPU.
        batch_size : int
            Number of recipes per batch.
        
        Returns
        state : tensor
            Hidden state of zeros.
        """
        state = torch.zeros((self.num_directions * self.num_layers,
                             batch_size, self.num_hiddens), device=device)
        return  state
    