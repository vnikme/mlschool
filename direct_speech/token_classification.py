import direct_speech
import json
import random
import sys
import torch
import transformers


class TokenClassificator(torch.nn.Module):
    def __init__(self, bert_model_name, hidden_size, layer_count, output_size, device):
        super(TokenClassificator, self).__init__()

        # input parameters
        self.hidden_size = hidden_size
        self.layer_count = layer_count
        self.output_size = output_size

        # base bert model
        self.bert = transformers.XLMRobertaModel.from_pretrained(bert_model_name).to(device)
        for name, parameter in self.bert.named_parameters():
            parameter.requires_grad_(False)

        # RNN on top
        self.rnn = torch.nn.GRU(
            self.bert.encoder.layer[-1].output.dense.out_features,
            hidden_size,
            layer_count,
            batch_first=True,
            bidirectional=True
        ).to(device)
        self.fc = torch.nn.Linear(
            2 * hidden_size,
            output_size
        ).to(device)


    def forward(self, input_ids, attention_mask, device):
        x = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        h0 = torch.zeros(2 * self.layer_count, x.size(0), self.hidden_size).to(device).requires_grad_(False)
        x, _ = self.rnn(x, h0)
        return self.fc(x)


def encode_and_propagate_labels(input_str, markup, tokeniser, max_length):
    if max_length:
        encoded = tokeniser(input_str, return_offsets_mapping=True, padding='max_length', truncation=True, max_length=max_length)
    else:
        encoded = tokeniser(input_str, return_offsets_mapping=True)
    input_ids = encoded['input_ids']
    offsets = encoded['offset_mapping']
    input_len, markup_len = len(input_ids), len(markup)
    labels = [0 for _ in range(input_len)]
    token_id, markup_id = 0, 0
    while token_id < input_len and markup_id < markup_len:
        if offsets[token_id][0] >= markup[markup_id][1]:
            markup_id += 1
            continue
        if offsets[token_id][1] <= markup[markup_id][0]:
            token_id += 1
            continue
        current_label = markup[markup_id][2] * 2
        end_position = markup[markup_id][1]
        markup_id += 1
        labels[token_id] = current_label - 1
        while True:
            token_id += 1
            if token_id == input_len:
                break
            if offsets[token_id][0] >= end_position or (offsets[token_id][0] == 0 and offsets[token_id][1] == 0):
                break
            labels[token_id] = current_label
    return encoded, labels


def prepare_markup(text, tokeniser):
    markup = []
    for quote_open_position, quote_close_position, separator_end_position, trigger_end_position, clarification_begin_position, clarification_end_position in direct_speech.iterate_direct_speech(text):
        markup.append((quote_open_position + 1, quote_close_position - 1, 1))
        #markup.append((separator_end_position, trigger_end_position, 2))
        #markup.append((clarification_begin_position, clarification_end_position, 3))
        markup.append((separator_end_position, clarification_end_position, 2))
    return markup


def prepare_dataset(paths, tokeniser, max_length, neg_prob):
    all_input_ids, all_attention_masks, all_labels = [], [], []
    lengths, speech_count, total_count = [], 0, 0
    for path in paths:
        data = json.load(open(path, 'rt'))
        for obj in data:
            for text in obj['text'].split('\n'):
            #for text in [obj['text']]:
                if not text:
                    continue
                markup = prepare_markup(text, tokeniser)
                if len(markup) != 0:
                    speech_count += 1
                elif neg_prob < 1 and random.random() >= neg_prob:
                    continue
                encoded, labels = encode_and_propagate_labels(text, markup, tokeniser, max_length)
                all_input_ids.append(torch.LongTensor(encoded['input_ids']))
                all_attention_masks.append(torch.LongTensor(encoded['attention_mask']))
                all_labels.append(torch.LongTensor(labels))
                total_count += 1
                lengths.append(int(sum(all_attention_masks[-1])))
    lengths = sorted(lengths)
    print('Total count: {}\nHas speech: {} ({})\n90th percentile: {}\n95th percentile: {}\n99th percentile: {}'.format(total_count, speech_count, speech_count / total_count, lengths[len(lengths) * 9 // 10], lengths[len(lengths) * 95 // 100], lengths[len(lengths) * 99 // 100]))
    return all_input_ids, all_attention_masks, all_labels


def split_train_test(all_input_ids, all_attention_masks, all_labels, test_prob):
    train_input_ids, test_input_ids, train_attention_masks, test_attention_masks, train_labels, test_labels = [], [], [], [], [], []
    n = len(all_input_ids)
    idx = list(range(n))
    random.shuffle(idx)
    #for i in idx:
    for i in idx[:2500]:
        if random.random() > test_prob:
            train_input_ids.append(all_input_ids[i])
            train_attention_masks.append(all_attention_masks[i])
            train_labels.append(all_labels[i])
        else:
            test_input_ids.append(all_input_ids[i])
            test_attention_masks.append(all_attention_masks[i])
            test_labels.append(all_labels[i])
    return train_input_ids, test_input_ids, train_attention_masks, test_attention_masks, train_labels, test_labels


def predict_markup(model, tokeniser, input_str, max_length, device):
    #encoded = tokeniser(input_str, return_offsets_mapping=True, padding='max_length', truncation=True, max_length=max_length)
    markup = prepare_markup(input_str, tokeniser)
    encoded, labels = encode_and_propagate_labels(input_str, markup, tokeniser, max_length)
    input_ids = torch.LongTensor([encoded['input_ids']]).to(device)
    attention_mask = torch.LongTensor([encoded['attention_mask']]).to(device)
    offsets = encoded['offset_mapping']
    print('{} {}'.format(int(sum(attention_mask[0])), input_str))
    with torch.no_grad():
        output = model.forward(input_ids, attention_mask, device)[0]
        max_indices = [torch.argmax(x).item() for x in output]
        print(' '.join(['{}{}'.format(x, y) for x, y in zip(labels, max_indices)]))
        #print(labels)
        #print(max_indices)
        markup, i, n = [], 0, len(max_indices)
        while i < n:
            if max_indices[i] % 2 == 0:
                i += 1
                continue
            label = (max_indices[i] + 1) // 2
            j = i + 1
            left, right = offsets[i][0], offsets[i][1]
            while j < n and max_indices[j] % 2 == 0 and (max_indices[j] + 1) // 2 == label:
                right = max(right, offsets[j][1])
                j += 1
            markup.append((left, right, label))
            print('{}: {}'.format(label, input_str[markup[-1][0]:markup[-1][1]]))
            i = j
        print("")
        return markup


def main():
    cuda = torch.cuda.is_available() #and False
    device = torch.device("cuda" if cuda else "cpu")
    print(cuda, device)

    bert_model_name = 'xlm-roberta-base'
    tokeniser = transformers.XLMRobertaTokenizerFast.from_pretrained(bert_model_name)
    print(len(tokeniser.vocab))
    number_of_classes, max_length, hidden_size, layer_count = 2, 150, 512, 8
    model = TokenClassificator(bert_model_name, hidden_size, layer_count, number_of_classes * 2 + 1, device)
    #model.load_state_dict(torch.load('{}/{}.pt'.format('models', 26)))
    criterion = torch.nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-5, eps=1e-8)
    #opt = torch.optim.Adam(model.parameters(), lr=0.001)
    #opt = torch.optim.SGD(model.parameters(), lr=0.001)

    all_input_ids, all_attention_masks, all_labels = prepare_dataset(sys.argv[1:], tokeniser, max_length, 0.05)
    train_input_ids, test_input_ids, train_attention_masks, test_attention_masks, train_labels, test_labels = split_train_test(all_input_ids, all_attention_masks, all_labels, 0.1)

    for epoch in range(100):
        train_loss, train_count, test_loss, test_count, batch_size = 0.0, 0, 0.0, 0, 128

        idx = list(range(len(train_input_ids)))
        random.shuffle(idx)
        model.train()
        for i in range(0, len(idx), batch_size):
            if i + batch_size >= len(idx):
                break
            input_ids, attention_mask, labels = [], [], []
            for j in range(batch_size):
                input_ids.append(train_input_ids[idx[i + j]])
                attention_mask.append(train_attention_masks[idx[i + j]])
                labels.append(train_labels[idx[i + j]])
            input_ids = torch.stack(input_ids).to(device)
            attention_mask = torch.stack(attention_mask).to(device)
            labels = torch.stack(labels).to(device)
            output = model.forward(input_ids, attention_mask, device)
            output = output.reshape((-1, number_of_classes * 2 + 1))
            labels = labels.reshape((-1,))
            loss = criterion(output, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=1.0)
            opt.step()
            train_loss += float(loss)
            train_count += 1

        model.eval()
        idx = list(range(len(test_input_ids)))
        random.shuffle(idx)
        full_match_count = 0
        for i in range(0, len(idx), batch_size):
            input_ids, attention_mask, labels = [], [], []
            for j in range(batch_size):
                if i + j >= len(idx):
                    break
                input_ids.append(test_input_ids[idx[i + j]])
                attention_mask.append(test_attention_masks[idx[i + j]])
                labels.append(test_labels[idx[i + j]])
            input_ids = torch.stack(input_ids).to(device)
            attention_mask = torch.stack(attention_mask).to(device)
            labels = torch.stack(labels).to(device)
            with torch.no_grad():
                output = model.forward(input_ids, attention_mask, device)
                for j in range(output.shape[0]):
                    is_match = True
                    max_indices = [torch.argmax(x).item() for x in output[j]]
                    for k in range(output.shape[1]):
                        if max_indices[k] != labels[j][k]:
                            is_match = False
                            break
                    if is_match:
                        full_match_count += 1
                    else:
                        predict_markup(model, tokeniser, tokeniser.decode(input_ids[j], skip_special_tokens=True), None, device)
                output = output.reshape((-1, number_of_classes * 2 + 1))
                labels = labels.reshape((-1,))
                loss = criterion(output, labels)
                test_loss += (float(loss) * labels.shape[0])
                test_count += labels.shape[0]

        print('Epoch {}, train {}, test {}, accuracy {}'.format(epoch, train_loss / train_count, test_loss / test_count, full_match_count / len(idx)))
        predict_markup(model, tokeniser, 'Песков прокомментировал вероятность всеобщей мобилизации. "Мы решили её некоторое время не начинать", - заверил он, добавив, что эмигрировать пока рано.', None, device)
        predict_markup(model, tokeniser, '"Песков прокомментировал вероятность всеобщей мобилизации. Мы решили её некоторое время не начинать", - заверил он, добавив, что эмигрировать пока рано.', None, device)
        predict_markup(model, tokeniser, 'Песков прокомментировал вероятность всеобщей мобилизации. "Мы решили её некоторое время не начинать", - заверил он, добавив, что эмигрировать пока рано. Цитата по "Интерфаксу".', None, device)
        sys.stdout.flush()
        #torch.save(model.state_dict(), '{}/{}.pt'.format('models', epoch))


if __name__ == '__main__':
    main()

