import torch
import torch.nn as nn


# noinspection SpellCheckingInspection
class DependencyParser(nn.Module):
    def __init__(self, vocab_size, pos_size, **kwargs):
        super(DependencyParser, self).__init__()

        external_word_embedding = kwargs.get('external_word_embedding', None)
        w_embed_dim = kwargs.get('word_embedding_dim', 100)
        p_embed_dim = kwargs.get('pos_embedding_dim', 25)
        lstm_layers = kwargs.get('lstm_layers', 2)
        lstm_hidden_dim = kwargs.get('lstm_hidden_dim', 100)
        mlp_hidden_dim = kwargs.get('mlp_hidden_dim', 100)

        e_embed_dim = 0

        self.w_embed = nn.Embedding(vocab_size, w_embed_dim)
        self.p_embed = nn.Embedding(pos_size, p_embed_dim)
        self.e_embed = external_word_embedding
        self.lstm = nn.LSTM(
            input_size=w_embed_dim + p_embed_dim + e_embed_dim,
            hidden_size=lstm_hidden_dim, num_layers=lstm_layers,
            bidirectional=True, batch_first=False
        )
        self.mlp_hidden_dim = mlp_hidden_dim
        self.mlp = nn.Linear(lstm_hidden_dim * 2, mlp_hidden_dim)
        self.mlp_activation = nn.Tanh()
        self.out_layer = nn.Linear(mlp_hidden_dim, 1)

    def forward(self, word_idx: torch.tensor, pos_idx: torch.tensor):
        length = word_idx.shape[0]
        w_embed = self.w_embed(word_idx)
        p_embed = self.p_embed(pos_idx)
        if self.e_embed is not None:
            e_embed = self.e_embed(word_idx)
            cat = torch.cat((w_embed, p_embed, e_embed), dim=1)
        else:
            cat = torch.cat((w_embed, p_embed), dim=1)
        lstm_out, _ = self.lstm(cat.view(length, 1, -1))
        scores = self.score(lstm_out, length)
        return scores

    def score(self, lstm_out, length):
        scores = torch.zeros(size=(length, length - 1), requires_grad=False)
        mlp = torch.zeros(size=(length, self.mlp_hidden_dim), requires_grad=False)
        for node in range(length):
            mlp[node, :] = self.mlp(lstm_out[node, :, :])
        for head in range(length):
            for child in range(length - 1):
                mlp_out = self.mlp_activation(torch.add(mlp[head, :], mlp[child + 1, :]))
                scores[head, child] = self.out_layer(mlp_out)
        return scores
