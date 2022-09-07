# SQuAD-issues
Issues of SQuAD 2.0

## Introduction
[The Stanford Question Answering Dataset (SQuAD) 2.0](https://rajpurkar.github.io/SQuAD-explorer/) is a prominent reading comprehension dataset, featuring 100,000 questions in the form of paragraphs of reading material, followed by questions regarding the content of the paragraph. In addition, 50,000 questions unanswerable based on the provided paragraphs alone have been added to the 2.0 version of the dataset. For the answerable questions, the model's task is to identify the span of the paragraph that constitute an answer to the question.

Perhaps inevitable for dataset of such size, there are apparent errors in some of the provided answers. Specifically, this repo presents issues surfaced by attempts to preprocess the dataset by [bit pair encoding (BPE)](https://en.wikipedia.org/wiki/Byte_pair_encoding), the tokenization used by models such as GPT-2 and [RoBERa](https://github.com/EIFY/fairseq/tree/main/examples/roberta):

```
import json

path = 'train-v2.0.json'
s = next(open(path, "r", encoding="utf-8"))
j = json.loads(s.strip())

def squad2_ligature_check(data, threshold):
    l = []
    def check(l1, l2):
        n = len(l2)
        i = 0
        while i < n and l1[i] == l2[i]:
            i += 1
        return n - i > threshold
    for i, d in enumerate(data):
        for j, p in enumerate(d['paragraphs']):
            context = bpe.encode(p['context'])
            for k, q in enumerate(p['qas']):
                question = bpe.encode(q['question'])
                if not q['is_impossible']:
                    a = q['answers']
                    assert len(a) == 1
                    a = a[0]
                    start_index = a['answer_start']
                    prefix = bpe.encode(p['context'][:start_index].rstrip())
                    up_to = bpe.encode(p['conte[286, 599, 72]xt'][:start_index + len(a['text'])])
                    if check(context, prefix) or check(contex
                    t, up_to):
                        l.append((i, j, k))
    return l

squad2_ligature_check(j['data'], 1)
```
```
[(3, 3, 0),
 (11, 82, 3),
 (23, 29, 1),
 (26, 22, 1),
 (26, 31, 2),
 (40, 16, 2),
 (48, 2, 2),
 (68, 12, 2),
 (100, 25, 2),
 (165, 0, 1),
 (210, 67, 3),
 (306, 30, 2),
 (319, 15, 2),
 (351, 0, 8),
 (359, 12, 1),
 (374, 29, 0),
 (431, 13, 3)]
```
A little more explanation on why the check above surfaces suspicious answers: As a variant of subword-level tokenization, BPE overall is coarser than character-level tokenization and finer than word-level tokenization. Furthermore, the following characters can change the tokenization of the preceding characters analogous to [ligature](https://en.wikipedia.org/wiki/Ligature_(writing)):
```
bpe.encode(' of spi')
```
```
[286, 599, 72]
```
```
bpe.encode(' of spiritual')
```
```
[286, 8557]
```
Since such case indicates that the preceding characters and following characters are expected to occur together but have been cut off inadvertently, it's worthwhile examining such answer spans more closely. For the examined answer spans below, I took the liberty of presenting the ~~original answer span~~ with strikethrough and ***my suggestion*** with bold and italic.

## i = 3, j = 3, k = 0
```
p = j['data'][3]['paragraphs'][3]
p['qas'][0]
```
```
{'question': 'Prior to iOS 5, how many apps were required to play music and videos on iPhone and iPad?',
 'id': '56cc57466d243a140015ef24',
 'answers': [{'text': 'one', 'answer_start': 98}],
 'is_impossible': False}
```
```
p['context']
```
>Before the release of iOS 5, the iPod branding was used for the media player included with the iPh***one*** and iPad, a combination of the Music and Videos apps on the iPod Touch. As of iOS 5, separate apps named "Music" and "Videos" are standardized across all iOS-powered products. While the iPhone and iPad have essentially the same media player capabilities as the iPod line, they are generally treated as separate products. During the middle of 2010, iPhone sales overtook those of the iPod.'

The answer span uses "one" within iPh***one***, which is strange but I am leaving this alone due to the lack of better alternatives.

## i = 11, j = 82, k = 3
```
p = j['data'][11]['paragraphs'][82]
p['qas'][3]
```
```
{'question': 'Psycho-physical energy is harnessed through what?',
 'id': '56d24a6fb329da140004ed00',
 'answers': [{'text': 'ritual', 'answer_start': 291}],
 'is_impossible': False}
```
```
p['context']
```
>Though based upon Mahayana, Tibeto-Mongolian Buddhism is one of the schools that practice Vajrayana or "Diamond Vehicle" (also referred to as Mantrayāna, Tantrayāna, Tantric Buddhism, or esoteric Buddhism). It accepts all the basic concepts of Mahāyāna, but also includes a vast array of spi~~ritual~~ and physical techniques designed to enhance Buddhist practice. Tantric Buddhism is largely concerned with ritual and meditative practices. One component of the Vajrayāna is harnessing psycho-physical energy through ***ritual, visualization, physical exercises, and meditation*** as a means of developing the mind. Using these techniques, it is claimed that a practitioner can achieve Buddhahood in one lifetime, or even as little as three years. In the Tibetan tradition, these practices can include sexual yoga, though only for some very advanced practitioners.

This is where the encoding example above  came from. Using "ritual" within spi~~ritual~~ is not only strange but also incomplete: ***ritual, visualization, physical exercises, and meditation*** is the complete answer.
