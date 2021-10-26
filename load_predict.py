import paddle
import paddlenlp
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import paddle.nn.functional as F
from paddlenlp.data import Stack, Tuple, Pad
from utils import convert_example, create_dataloader

batch_size = 128
test_number = 1000

def txt_to_list(file_name):
    with open(file_name, 'r', encoding='utf8') as f:
        res_list = []
        for line in f:
            line = line.strip('\n')
            res_list.append(line)

        train_list, test_list = train_test_split(res_list, random_state=123)
        return train_list, test_list


data_dict = {
    'work/Pfam-A.clans.corpus.txt': 0,
    'work/cdd_specific_arch.corpus.txt': 1,
    'work/scop-des-corpus.txt': 2,
    'work/cddmasters_comments.corpus.txt': 3
}
train_list, test_list = [], []

for key, value in data_dict.items():
    train, test = txt_to_list(key)
    for text in train:
        train_list.append({'text': text, 'label': value})

    for text in test:
        test_list.append({'text': text, 'label': value})
# 混洗一下
np.random.shuffle(train_list)
np.random.shuffle(test_list)



MODEL_NAME = "ernie-2.0-en"

model = ppnlp.transformers.ErnieForSequenceClassification.from_pretrained(MODEL_NAME, num_classes=4)

tokenizer = ppnlp.transformers.ErnieTokenizer.from_pretrained(MODEL_NAME)

def predict1(model, data, tokenizer, batch_size=1):
    """
    Predicts the data labels.

    Args:
        model (obj:`paddle.nn.Layer`): A model to classify texts.
        data (obj:`List(Example)`): The processed data whose each element is a Example (numedtuple) object.
            A Example object contains `text`(word_ids) and `se_len`(sequence length).
        tokenizer(obj:`PretrainedTokenizer`): This tokenizer inherits from :class:`~paddlenlp.transformers.PretrainedTokenizer`
            which contains most of the methods. Users should refer to the superclass for more information regarding methods.
        label_map(obj:`dict`): The label id (key) to label str (value) map.
        batch_size(obj:`int`, defaults to 1): The number of batch.

    Returns:
        results(obj:`dict`): All the predictions labels.
    """
    examples = []
    for text in data:
        input_ids, segment_ids = convert_example(
            text,
            tokenizer,
            max_seq_length=128,
            is_test=True)
        examples.append((input_ids, segment_ids))

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input id
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # segment id
    ): fn(samples)

    # Seperates data into some batches.
    batches = []
    one_batch = []
    for example in examples:
        one_batch.append(example)
        if len(one_batch) == batch_size:
            batches.append(one_batch)
            one_batch = []
    if one_batch:
        # The last batch whose size is less than the config batch_size setting.
        batches.append(one_batch)

    results = []
    model.eval()
    for batch in batches:
        input_ids, segment_ids = batchify_fn(batch)
        input_ids = paddle.to_tensor(input_ids)
        segment_ids = paddle.to_tensor(segment_ids)
        probs = F.softmax(model(input_ids, segment_ids), axis=1).numpy()
        idx = np.argmax(probs, axis=1)
        idx = idx.tolist()
        results.extend(idx)
    return results

test_titles, test_labels = [], []
for item in test_list:
    test_titles.append(item['text'])
    test_labels.append(item['label'])
results=[]

model.set_state_dict(paddle.load('checkpoint/model_state.pdparams'))

results = predict1(
    model, test_list[:test_number], tokenizer, batch_size=batch_size)

print(classification_report(test_labels[:test_number], results))