import nltk
from nltk.corpus import stopwords
from string import punctuation
import re

allowed_pos = {'FW', 'CD',
               'JJ', 'JJR', 'JJS',
               'NN', 'NNS', 'NNP', 'NNPS',
               'RB', 'RBR', 'RBS',
               'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'
               }

nounonly_allowed_pos = {'FW', 'CD',
                        'NN', 'NNS', 'NNP', 'NNPS',
                        }

ban_word = ['am', 'is', 'are', 'being', 'was', 'were', 'has', 'have', 'been', 'be', 'done', 'which', 'whom', 'what',
            'when',
            'why',
            'how', 'do', 'did',
            'almost',
            'most', 'had', '\'t', '\'s',
            'Am', 'Is', 'Are', 'Being', 'Was', 'Were', 'Has', 'Have', 'Been', 'Be', 'Done', 'Which', 'Whom', 'What',
            'When',
            'Why', 'How', 'Do', 'Did',
            'Almost',
            'Most', 'Had']
stopset = set(stopwords.words('english')).union(ban_word)

def get_ne(sentence):
    tokens = nltk.word_tokenize(sentence)
    # pos
    postags = nltk.pos_tag(tokens)
    # ner
    namedEnt = nltk.ne_chunk(postags, binary=False)
    ss = []
    for t in namedEnt:
        if hasattr(t, "label"):
            b = " ".join([x[0] for x in t])
        elif (t[1] in nounonly_allowed_pos) and (t[0] not in stopset) and (t[0] not in punctuation):
            b = t[0]
        else:
            continue

        if b not in stopset and b not in punctuation and b not in ss:
            ss.append(b)
    return ss

def process_an_item(sentence, nounonly=True, neonly=True):
    """
    :param sentence:
    :param relational_prompt:
    :param relational_prompt_used:
    :return: The function returns the keywords/entity in their order in original sentence
    """

    real_allowed_pos = nounonly_allowed_pos if nounonly else allowed_pos

    # import re
    # from nltk import word_tokenize
    # sent = "‘I Predict a Riot’ (2004) and \"Never Miss a Beat\" (2008) were Top 10 hits for whom?"
    quote_pattern = r"‘.*?’|“.*?”|(?='.*?')(?=(?!'t\s))(?=(?!'s\s))(?=(?!'ve\s))(?=(?!'m\s))|\".*?\""
    quotes = re.findall(quote_pattern, sentence)
    sentence = re.sub(quote_pattern, "' '", sentence)
    # print(m) # all quotes
    # print(s) # all quotes to ' ', a placeholder

    continuous_cap_words_pattern = r"([A-Z][^\s\?]*(?=\s[A-Z])(?:\s[A-Z][^\s\?]*)*)"
    cont_cap_words = re.findall(continuous_cap_words_pattern, sentence)
    refined_ones = []
    for item in cont_cap_words:
        flag = True
        for banword in ban_word:
            if item.startswith(banword + ' '):
                refined_ones.append(item[len(banword):].strip())
                flag = False
                break
        if flag:
            # not detected in above
            refined_ones.append(item)

    sentence = re.sub(continuous_cap_words_pattern, "' '", sentence)

    tokens = nltk.word_tokenize(sentence)
    # pos
    postags = nltk.pos_tag(tokens)
    # ner
    namedEnt = nltk.ne_chunk(postags, binary=False)
    ss = []
    for t in namedEnt:
        if hasattr(t, "label"):
            b = " ".join([x[0] for x in t])
        # elif (not neonly) and (t[1] in real_allowed_pos) and (t[0] not in stopset) and (t[0] not in punctuation):
        #     b = t[0]
        else:
            continue
        if b not in stopset and b not in punctuation and b not in ss:
            ss.append(b)
    return quotes + refined_ones + ss

if __name__ == '__main__':
    res = process_an_item("Ernest Hemingway was an American author and journalist.")
    print(res)