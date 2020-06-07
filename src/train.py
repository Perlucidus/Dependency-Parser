from dependency_parser import DependencyParser
from chu_liu_edmonds import decode_mst
from conllu import IndexedConllUDataset
import torch
import torch.utils.data
from pathlib import Path
from time import time
import matplotlib.pyplot as plt

threads = 2


def unlabeled_attachment_score(scores, heads):
    parse_tree, _ = decode_mst(scores[1:, :].detach().numpy(), heads.shape[0] - 1, has_labels=False)
    return sum(parse_tree[i] + 1 == heads[i].item() for i in range(heads.shape[0] - 1)) / (heads.shape[0] - 1)


def collate(batch):
    sentences = []
    for words, pos, heads in batch:
        word_tensor = torch.tensor(words, dtype=torch.long)
        pos_tensor = torch.tensor(pos, dtype=torch.long)
        head_tensor = torch.tensor(heads)
        sentences.append((word_tensor, pos_tensor, head_tensor[1:]))
    return sentences


def train(parser: DependencyParser, train_data, test_data=None, word_dropout=0.25, epochs=30, batch_size=500):
    data_loader = torch.utils.data.DataLoader(train_data, batch_size, shuffle=True, collate_fn=collate)
    test_data_loader = torch.utils.data.DataLoader(test_data, collate_fn=collate)
    loss_func = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(parser.parameters(), lr=2e-3)
    losses = []
    uas = []
    test_uas = []
    for epoch in range(epochs):
        start = time()
        epoch_loss = 0
        epoch_uas = 0
        for b, batch in enumerate(data_loader):
            batch_time = time()
            batch_loss = 0
            batch_uas = 0
            for words, pos, heads in batch:
                words = words.clone()
                drop_idx = torch.rand(words.shape[0])
                for i, p in enumerate(drop_idx):
                    if p < word_dropout / (train_data.word_counts[words[i].item()] + word_dropout):
                        words[i] = 0
                scores = parser(words, pos)
                loss = loss_func(torch.nn.functional.log_softmax(scores, dim=0).t(), heads)
                batch_loss += loss.item()
                batch_uas += unlabeled_attachment_score(scores, heads)
                loss /= len(batch)
                loss.backward()
            optimizer.step()
            parser.zero_grad()
            epoch_loss += batch_loss
            epoch_uas += batch_uas
            print(f'Batch {b} completed in {time() - batch_time:.2f}s\tLoss: {batch_loss / len(batch):.4f}'
                  f'\tUAS: {batch_uas / len(batch)}:.4f')
        epoch_loss /= len(data_loader.dataset)
        epoch_uas /= len(data_loader.dataset)
        losses.append(epoch_loss)
        uas.append(epoch_uas)
        print(f'Epoch {epoch + 1} complete in {time() - start:.2f}s\tLoss: {epoch_loss:.4f}\tUAS: {epoch_uas:.4f}')
        if test_data:
            epoch_test_uas = 0
            with torch.no_grad():
                for batch in test_data_loader:
                    for words, pos, heads in batch:
                        epoch_test_uas += unlabeled_attachment_score(parser(words, pos), heads)
            epoch_test_uas /= len(test_data_loader.dataset)
            test_uas.append(epoch_test_uas)
            print(f'Test UAS: {epoch_test_uas:.4f}')
    print(f'Losses: {losses}')
    plt.subplot(2, 1, 1)
    plt.plot(range(epochs), losses, c='g')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss')
    plt.subplot(2, 1, 2)
    plt.plot(range(epochs), uas, label='Train UAS')
    if test_data:
        plt.plot(range(epochs), test_uas, label='Test UAS')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('UAS')
    plt.title('UAS')
    plt.subplots_adjust()
    plt.show()


def main():
    train_data = IndexedConllUDataset(Path('../data/train.labeled'), transform=True)
    test_data = IndexedConllUDataset(Path('../data/test.labeled'), transform=False)
    test_data.transform(train_data.word_to_idx, train_data.pos_to_idx)
    for n in (1,):  # (1, 2):
        model_path = Path(f'../model/model{n}')
        parser = None
        try:
            with model_path.open('rb') as f:
                parser = torch.load(f)
        except FileNotFoundError:
            print('Loading data')
            print('Initializing')
            parser = DependencyParser(vocab_size=len(train_data.word_to_idx), pos_size=len(train_data.pos_to_idx))
            print(parser)
            print('Training')
            now = time()
            train(parser, train_data, test_data)
            print(f'Training done in {time() - now:.2f}s')
            with model_path.open('wb') as f:
                torch.save(parser, f)
        finally:
            pass  # TODO evaluate comp


if __name__ == '__main__':
    torch.set_num_threads(threads)
    main()
