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

from fairseq.data.encoders.gpt2_bpe import get_encoder
encoder_json, vocab_bpe = 'gpt2_bpe/encoder.json', 'gpt2_bpe/vocab.bpe'
bpe = get_encoder(encoder_json, vocab_bpe)

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

l = squad2_ligature_check(j['data'], 1)
l
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

As the next step, I exempt the known examples above, lower the threshold, and count the distinct differences in the decoded strings:
```
known = set(l)
import collections

c = collections.Counter()

def squad2_ligature_check(data, threshold, known):
    l = []
    def check(l1, l2):
        n = len(l2)
        i = 0
        while i < n and l1[i] == l2[i]:
            i += 1
        if n - i <= threshold:
            return False
        c[bpe.decode(l1[i:n]), bpe.decode(l2[i:n])] += 1
        return True
    for i, d in enumerate(data):
        for j, p in enumerate(d['paragraphs']):
            context = bpe.encode(p['context'])
            for k, q in enumerate(p['qas']):
                if (i, j, k) not in known:
                    question = bpe.encode(q['question'])
                    if not q['is_impossible']:
                        a = q['answers']
                        assert len(a) == 1
                        a = a[0]
                        start_index = a['answer_start']
                        prefix = bpe.encode(p['context'][:start_index].rstrip())
                        up_to = bpe.encode(p['context'][:start_index + len(a['text'])])
                        if check(context, prefix) or check(context, up_to):
                            l.append((i, j, k))
    return l

l = squad2_ligature_check(j['data'], 0, known)

for k, v in c.most_common():
    print(k, v)
```
```
(').', ')') 176
('),', ')') 146
('".', '"') 145
('",', '"') 113
('%.', '%') 66
('%,', '%') 44
('%)', '%') 43
('.,', '.') 30
('."', '.') 26
('%),', '%') 17
('")', '"') 16
('%).', '%') 15
('.[', '.') 10
(' ("', ' (') 8
('"),', '")') 8
('%;', '%') 7
(');', ')') 7
('").', '")') 6
('"—', '"') 6
("'.", "'") 6
(' ($', ' (') 5
(' 2015', ' 20') 5
(' European', ' Europe') 5
('))', ')') 5
('!"', '!') 4
('+,', '+') 4
('").', '"') 4
(' sixth', ' six') 3
(' "\'', ' "') 3
(' 18', ' 181') 3
(' 23', ' 2') 3
(' fourth', ' four') 3
('".[', '"') 3
('/.', '/') 3
('.)', '.') 3
('%),', '%)') 3
('...', '.') 2
('fourth', 'four') 2
('2007', '200') 2
(' 32', ' 3') 2
(' 2003', ' 200') 2
(' 2004', ' 2') 2
(' 100', ' 10') 2
(' 2008', ' 2') 2
("'", '\'"') 2
(' 353', ' 3') 2
(' Buddh', ' Buddha') 2
(' 18', ' 1') 2
('.:', '.') 2
(' 12', ' 1') 2
(' notes', ' no') 2
(' 35', ' 3') 2
('?"', '?') 2
(' 15', ' 1') 2
(' 18', ' 183') 2
(' Postal', ' Post') 2
("',", "'") 2
(' 2016', ' 20') 2
(' (£', ' (') 2
('");', '")') 2
(' nobility', ' no') 2
(',"', ',') 2
('%);', '%') 2
(')—', ')') 2
(' evolutionary', ' evolution') 2
('.).', '.') 2
('"),', '"') 2
('.),', '.') 2
(' Rob', ' Rober') 1
(' oper', ' opera') 1
(' Tibetan', ' Tibet') 1
('2015', '201') 1
('..."', '...') 1
(' manually', ' manual') 1
(' Japanese', ' Japan') 1
(' Mall', ' M') 1
(' City', ' C') 1
(' 280', ' 2') 1
(' 11', ' 1') 1
(' 550', ' 5') 1
('33', '3') 1
(' 200', ' 2') 1
(' 2002', ' 2') 1
(' 2013', ' 2') 1
(' century', ' ce') 1
('ay', 'aya') 1
('aks', 'ak') 1
(' personal', ' person') 1
(' four', ' fourteen') 1
(' ha', ' hair') 1
(' 2015', ' 201') 1
(' 1990', ' 1') 1
(' 2009', ' 200') 1
(' 2008', ' 200') 1
(' fear', ' fearless') 1
(' breeding', ' breed') 1
(' emotional', ' emotion') 1
(' hunting', ' hunt') 1
(' 72', ' 7') 1
(' southwestern', ' southwest') 1
(' 2004', ' 20') 1
('os', 'o') 1
(']', ']"') 1
(' Islamic', ' Islam') 1
(' watt', ' w') 1
(' lighting', ' light') 1
(' architecture', ' architect') 1
(' Architecture', ' Architect') 1
(' environmentally', ' environment') 1
(' d', ' dash') 1
(' 2009', ' 20') 1
(' 1930', ' 19') 1
(' northern', ' north') 1
(' General', ' Ge') 1
('500', '5') 1
('27', '2') 1
(' 1917', ' 1') 1
(' negatively', ' negative') 1
(' 1897', ' 18') 1
('487', '4') 1
(' III', ' I') 1
(' 158', ' 15') 1
(' 2010', ' 20') 1
(' 1967', ' 1') 1
('290', '2') 1
(' 10', ' 1') 1
(' 230', ' 2') 1
('/,', '/') 1
('Hyd', 'H') 1
(' electr', ' electro') 1
(' 100', ' 1') 1
(' 1957', ' 195') 1
(' 400', ' 4') 1
(' 1962', ' 196') 1
(' Spain', ' Spa') 1
(' higher', ' high') 1
(' 227', ' 22') 1
('+.', '+') 1
(' Transportation', ' T') 1
(' B', ' Bi') 1
(' phon', ' ph') 1
(' 216', ' 2') 1
(' Zeal', ' Zealand') 1
(' extremely', ' extreme') 1
(' gol', ' golf') 1
(' 13', ' 1') 1
('esse', 'es') 1
(' 14', ' 148') 1
(' 259', ' 2') 1
(' indirectly', ' indirect') 1
('The', 'T') 1
(' peacefully', ' peaceful') 1
(' regional', ' region') 1
(' 1950', ' 19') 1
('].', ']') 1
(' The', ' Th') 1
(' represents', ' represent') 1
(' western', ' west') 1
(' largest', ' large') 1
(' Nonetheless', ' No') 1
(' sex', ' se') 1
('...', '..') 1
(' subjects', ' s') 1
(".'", '.') 1
(' avoiding', ' avoid') 1
(' opening', ' open') 1
('played', 'play') 1
(' inv', ' in') 1
('In', 'I') 1
('uses', 'us') 1
(' businesses', ' business') 1
(' Face', ' F') 1
(' resulted', ' r') 1
('uss', 'ussia') 1
(' Florida', ' Fl') 1
(' 1966', ' 1') 1
(' 1968', ' 1') 1
('!.', '!') 1
(' finally', ' final') 1
(')"', ')') 1
(' and', ' a') 1
(' freedoms', ' freedom') 1
(' 26', ' 2') 1
('The', 'Th') 1
(' the', ' th') 1
(')),', ')') 1
(').[', ')') 1
('Many', 'M') 1
('.",', '.') 1
(' singing', ' sing') 1
(' Speaking', ' Speak') 1
('!",', '!"') 1
(')[', ')') 1
('-"', '-') 1
('ity', 'i') 1
(' usually', ' u') 1
('Federal', 'F') 1
('..."', '..') 1
(' b', ' bilateral') 1
(' maintained', ' maintain') 1
('?).', '?)') 1
(' (~', ' (') 1
('.).', '.)') 1
(' occurs', ' occur') 1
(' (#', ' (') 1
('!,', '!') 1
("'),", "'") 1
(' 140', ' 1') 1
(')-', ')') 1
(' whit', ' white') 1
('.-', '.') 1
(' D', ' Do') 1
(' 1999', ' 199') 1
('uls', 'ul') 1
('-,', '-') 1
(' 18', ' 182') 1
(' 300', ' 3') 1
(' world', ' wo') 1
('":', '"') 1
('";', '"') 1
(')', ')?') 1
(' (>', ' (') 1
('ans', 'a') 1
(' issued', ' i') 1
(' enriched', ' enrich') 1
('$.', '$') 1
('?",', '?"') 1
(' (−', ' (') 1
(' 1800', ' 180') 1
(' was', ' wa') 1
```
As we can see, most of them are grouped punctuations that reflect the inherent limitations of BPE. However, the others look worrying and may also be words or numbers cut off inadvertently.

## Train set

### i = 3, j = 3, k = 0
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
>Before the release of iOS 5, the iPod branding was used for the media player included with the iPh***one*** and iPad, a combination of the Music and Videos apps on the iPod Touch. As of iOS 5, separate apps named "Music" and "Videos" are standardized across all iOS-powered products. While the iPhone and iPad have essentially the same media player capabilities as the iPod line, they are generally treated as separate products. During the middle of 2010, iPhone sales overtook those of the iPod.

The answer span uses "one" within "iPh***one***", which is strange but I am leaving this alone due to the lack of better alternatives.

### i = 11, j = 82, k = 3
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

### i = 23, j = 29, k = 1
```
p = j['data'][23]['paragraphs'][29]
p['qas'][1]
```
```
{'question': 'Each Brigade contains how many regiments?',
 'id': '56deebdc3277331400b4d81f',
 'answers': [{'text': 'one', 'answer_start': 33}],
 'is_impossible': False}
```
```
p['context']
```
>Currently, the Regular Force comp~~one~~nt of the Army consists of three field-ready brigade groups: 1 Canadian Mechanized Brigade Group, at CFB Edmonton and CFB Shilo; 2 Canadian Mechanized Brigade Group, at CFB Petawawa and CFB Gagetown; and 5 Canadian Mechanized Brigade Group, at CFB Valcartier and Quebec City. Each contains ***one*** regiment each of artillery, armour, and combat engineers, three battalions of infantry (all scaled in the British fashion), one battalion for logistics, a squadron for headquarters/signals, and several smaller support organizations. A tactical helicopter squadron and a field ambulance are co-located with each brigade, but do not form part of the brigade's command structure.

### i = 26, j = 22, k = 1
```
p = j['data'][26]['paragraphs'][22]
p['qas'][1]
```
```
{'question': 'Would you consider aesthetic elements alone in architectural lighting design?',
 'id': '56df865956340a1900b29ceb',
 'answers': [{'text': 'kno', 'answer_start': 58}],
 'is_impossible': False}
```
```
p['context']
```
>Lighting design as it applies to the built environment is ~~k~~***no***wn as 'architectural lighting design'. Lighting of structures considers aesthetic elements as well as practical considerations of quantity of light required, occupants of the structure, energy efficiency and cost. Artificial lighting takes into account the amount of daylight received in an internal space by using Daylight factor calculation. For simple installations, hand-calculations based on tabular data are used to provide an acceptable lighting design. More critical or optimized designs now routinely use mathematical modeling on a computer using software such as Radiance which can allow an Architect to quickly undertake complex calculations to review the benefit of a particular design.

The original answer span "kno" is nonsensical. Using "no" within "k***no***wn" is still strange but at least correct.

### i = 26, j = 31, k = 2
```
p = j['data'][26]['paragraphs'][31]
p['qas'][2]
```
```
{'question': 'Would a lower GAI mean higher apparent saturation or vividness of object colors?',
 'id': '56df95d44a1a83140091eb81',
 'answers': [{'text': 'ano', 'answer_start': 156}],
 'is_impossible': False}
```
```
p['context']
```
>For example, in order to meet the expectations for good color rendering in retail applications, research suggests using the well-established CRI along with ~~a~~***no***ther metric called gamut area index (GAI). GAI represents the relative separation of object colors illuminated by a light source; the greater the GAI, the greater the apparent saturation or vividness of the object colors. As a result, light sources which balance both CRI and GAI are generally preferred over ones that have only high CRI or only high GAI.

Same as the above.

### i = 40, j = 16, k = 2
```
p = j['data'][40]['paragraphs'][16]
p['qas'][2]
```
```
{'question': 'What group members are the big game hunters?',
 'id': '56dfa04b38dc421700152128',
 'answers': [{'text': 'men', 'answer_start': 115}],
 'is_impossible': False}
```
```
p['context']
```
>It is easy for Western-educated scholars to fall into the trap of viewing hunter-gatherer social and sexual arrange~~men~~ts in the light of Western values.[editorializing] One common arrangement is the sexual division of labour, with women doing most of the gathering, while ***men*** concentrate on big game hunting. It might be imagined that this arrangement oppresses women, keeping them in the domestic sphere. However, according to some observers, hunter-gatherer women would not understand this interpretation. Since childcare is collective, with every baby having multiple mothers and male carers, the domestic sphere is not atomised or privatised but an empowering place to be.[citation needed] In all hunter-gatherer societies, women appreciate the meat brought back to camp by men. An illustrative account is Megan Biesele's study of the southern African Ju/'hoan, 'Women Like Meat'. Recent archaeological research suggests that the sexual division of labor was the fundamental organisational innovation that gave Homo sapiens the edge over the Neanderthals, allowing our ancestors to migrate from Africa and spread across the globe.


### i = 48, j = 2, k = 2
```
p = j['data'][48]['paragraphs'][2]
p['qas'][2]
```
```
{'question': 'What modifier indicates a voiceless bilabial stop?',
 'id': '56e042487aa994140058e409',
 'answers': [{'text': 'p', 'answer_start': 32}],
 'is_impossible': False} 
```
```
p['context']
```
>In the International Phonetic Al~~p~~habet (IPA), aspirated consonants are written using the symbols for voiceless consonants followed by the aspiration modifier letter ⟨◌ʰ⟩, a superscript form of the symbol for the voiceless glottal fricative ⟨h⟩. For instance, ⟨***p***⟩ represents the voiceless bilabial stop, and ⟨pʰ⟩ represents the aspirated bilabial stop.

### i = 68, j = 12, k = 2
```
p = j['data'][68]['paragraphs'][12]
p['qas'][2]
```
```
{'question': 'Along with variety and languoid, what is another term used for a language without determining its independent status?',
 'id': '56e824c637bdd419002c446a',
 'answers': [{'text': 'lect', 'answer_start': 163}],
 'is_impossible': False}
```
```
p['context']
```
>There are various terms that linguists may use to avoid taking a position on whether the speech of a community is an independent language in its own right or a dia~~lect~~ of another language. Perhaps the most common is "variety"; "***lect***" is another. A more general term is "languoid", which does not distinguish between dialects, languages, and groups of languages, whether genealogically related or not.

### i = 100, j = 25, k = 2
```
p = j['data'][100]['paragraphs'][25]
p['qas'][2]
```
```
{'question': 'What is used to identify the begining of a valid frame of an MP3 header?',
 'id': '5706300775f01819005e7a62',
 'answers': [{'text': 'dard', 'answer_start': 550}],
 'is_impossible': False}
```
```
p['context']
```
>An MP3 file is made up of MP3 frames, which consist of a header and a data block. This sequence of frames is called an elementary stream. Due to the "byte reservoir", frames are not independent items and cannot usually be extracted on arbitrary frame boundaries. The MP3 Data blocks contain the (compressed) audio information in terms of frequencies and amplitudes. The diagram shows that the MP3 Header consists of a ***sync word***, which is used to identify the beginning of a valid frame. This is followed by a bit indicating that this is the MPEG stan~~dard~~ and two bits that indicate that layer 3 is used; hence MPEG-1 Audio Layer 3 or MP3. After this, the values will differ, depending on the MP3 file. ISO/IEC 11172-3 defines the range of values for each section of the header along with the specification of the header. Most MP3 files today contain ID3 metadata, which precedes or follows the MP3 frames, as noted in the diagram.

### i = 165, j = 0, k = 1
```
p = j['data'][165]['paragraphs'][0]
p['qas'][1]
```
```
{'question': 'What happens to information during the encoding process?', 'id': '571a275210f8ca1400304f06', 'answers': [{'text': 'g allows information from the outside world to be sensed in the form of chemical and physical stimuli.', 'answer_start': 100}], 'is_impossible': False}
```
```
p['context']
```
>In psychology, memory is the process in which information is encoded, stored, and retrieved. ***Encoding allows information from the outside world to be sensed in the form of chemical and physical stimuli.*** In the first stage the information must be changed so that it may be put into the encoding process. Storage is the second memory stage or process. This entails that information is maintained over short periods of time. Finally the third process is the retrieval of information that has been stored. Such information must be located and returned to the consciousness. Some retrieval attempts may be effortless due to the type of information, and other attempts to remember stored information may be more demanding for various reasons.

### i = 210, j = 67, k = 3
```
p = j['data'][210]['paragraphs'][67]
p['qas'][3]
```
```
{'question': 'Are there any alternatives to the public school system in Burma ?', 'id': '572908166aef0514001549cb', 'answers': [{'text': 'privately funded English language schoo', 'answer_start': 315}], 'is_impossible': False}
```
```
p['context']
```
>The educational system of Myanmar is operated by the government agency, the Ministry of Education. The education system is based on the United Kingdom's system due to nearly a century of British and Christian presences in Myanmar. Nearly all schools are government-operated, but there has been a recent increase in ***privately funded English language schools***. Schooling is compulsory until the end of elementary school, approximately about 9 years old, while the compulsory schooling age is 15 or 16 at international level.

### i = 306, j = 30, k = 2
```
p = j['data'][306]['paragraphs'][30]
p['qas'][2]
```
```
{'question': 'Who made the first US mandolin?', 'id': '5728b4714b864d1900164c70', 'answers': [{'text': 'Joseph Bohm', 'answer_start': 244}], 'is_impossible': False}
```
```
p['context']
```
>Mandolin awareness in the United States blossomed in the 1880s, as the instrument became part of a fad that continued into the mid-1920s. According to Clarence L. Partee, the first mandolin made in the United States was made in 1883 or 1884 by ***Joseph Bohmann***, who was an established maker of violins in Chicago. Partee characterized the early instrument as being larger than the European instruments he was used to, with a "peculiar shape" and "crude construction," and said that the quality improved, until American instruments were "superior" to imported instruments. At the time, Partee was using an imported French-made mandolin.

### i = 319, j = 15, k = 2
```
p = j['data'][319]['paragraphs'][15]
p['qas'][2]
```
```
{'question': 'What did Palermo expansion lack?', 'id': '572967351d046914007793ad', 'answers': [{'text': 't parks, schools, public buildings, proper roads and the other amenities that characterise a modern city', 'answer_start': 518}], 'is_impossible': False}
```
```
p['context']
```
>The so-called "Sack of Palermo" is one of the major visible faces of the problem. The term is used to indicate the speculative building practices that have filled the city with poor buildings. The reduced importance of agriculture in the Sicilian economy has led to a massive migration to the cities, especially Palermo, which swelled in size, leading to rapid expansion towards the north. The regulatory plans for expansion was largely ignored in the boom. New parts of town appeared almost out of nowhere, but withou~~t~~ ***parks, schools, public buildings, proper roads and the other amenities that characterise a modern city***.

### i = 351, j = 0, k = 8
```
p = j['data'][351]['paragraphs'][0]
p['qas'][8]
```
```
{'question': 'What type of films did Spielberg find early success with?', 'id': '573189d6e6313a140071d066', 'answers': [{'text': 'y science-fiction and adventure', 'answer_start': 116}], 'is_impossible': False}
```
```
p['context']
```
>In a career spanning more than four decades, Spielberg's films have covered many themes and genres. Spielberg's earl~~y~~ ***science-fiction and adventure*** films were seen as archetypes of modern Hollywood blockbuster filmmaking. In later years, his films began addressing humanistic issues such as the Holocaust (in Schindler's List), the transatlantic slave trade (in Amistad), war (in Empire of the Sun, Saving Private Ryan, War Horse and Bridge of Spies) and terrorism (in Munich). His other films include Close Encounters of the Third Kind, the Indiana Jones film series, and A.I. Artificial Intelligence.

### i = 359, j = 12, k = 1
```
p = j['data'][359]['paragraphs'][12]
p['qas'][1]
```
```
{'question': 'Why were experiments done on luminiferous aether in the 19 Century?', 'id': '572ebe0a03f98919007569d2', 'answers': [{'text': '"While the interstellar absorbing medium may be simply the ether, [it] is characteris', 'answer_start': 901}], 'is_impossible': False}
```
```
p['context']
```
>While outer space provides the most rarefied example of a naturally occurring partial vacuum, the heavens were originally thought to be seamlessly filled by a rigid indestructible material called aether. Borrowing somewhat from the pneuma of Stoic physics, aether came to be regarded as the rarefied air from which it took its name, (see Aether (mythology)). Early theories of light posited a ubiquitous terrestrial and celestial medium through which light propagated. Additionally, the concept informed Isaac Newton's explanations of both refraction and of radiant heat. ***19th century experiments into this luminiferous aether attempted to detect a minute drag on the Earth's orbit.*** While the Earth does, in fact, move through a relatively dense medium in comparison to that of interstellar space, the drag is so minuscule that it could not be detected. In 1912, astronomer Henry Pickering commented: ~~"While the interstellar absorbing medium may be simply the ether, [it] is characteris~~tic of a gas, and free gaseous molecules are certainly there".

### i = 374, j = 29, k = 0
```
p = j['data'][374]['paragraphs'][29]
p['qas'][0]
```
```
{'question': 'Can a DBMS be transfered to a different DBMS?', 'id': '572fb6f904bcaa1900d76c27', 'answers': [{'text': 'ano', 'answer_start': 50}], 'is_impossible': False}
```
```
p['context']
```
>***A database built with one DBMS is not portable to another DBMS (i.e., the other DBMS cannot run it).*** However, in some situations it is desirable to move, migrate a database from one DBMS to another. The reasons are primarily economical (different DBMSs may have different total costs of ownership or TCOs), functional, and operational (different DBMSs may have different capabilities). The migration involves the database's transformation from one DBMS type to another. The transformation should maintain (if possible) the database related application (i.e., all related application programs) intact. Thus, the database's conceptual and external architectural levels should be maintained in the transformation. It may be desired that also some aspects of the architecture internal level are maintained. A complex or large database migration may be a complicated and costly (one-time) project by itself, which should be factored into the decision to migrate. This in spite of the fact that tools may exist to help migration between specific DBMSs. Typically a DBMS vendor provides tools to help importing databases from other popular DBMSs.

### i = 431, j = 13, k = 3
```
p = j['data'][431]['paragraphs'][13]
p['qas'][3]
```
```
{'question': 'Where is the main base for the Tajikistan air force?', 'id': '5733b06ad058e614000b605c', 'answers': [{'text': 't located 15 km southwest of Dushanbe', 'answer_start': 489}], 'is_impossible': False}
```
```
p['context']
```
>Russian border troops were stationed along the Tajik–Afghan border until summer 2005. Since the September 11, 2001 attacks, French troops have been stationed at the Dushanbe Airport in support of air operations of NATO's International Security Assistance Force in Afghanistan. United States Army and Marine Corps personnel periodically visit Tajikistan to conduct joint training missions of up to several weeks duration. The Government of India rebuilt the ***Ayni Air Base, a military airport located 15 km southwest of Dushanbe***, at a cost of $70 million, completing the repairs in September 2010. It is now the main base of the Tajikistan air force. There have been talks with Russia concerning use of the Ayni facility, and Russia continues to maintain a large base on the outskirts of Dushanbe.
