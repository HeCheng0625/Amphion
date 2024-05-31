'''https://github.com/bootphon/phonemizer'''
import re

'''
    Text clean time
'''
rep_map = {
    "：": ",",
    "；": ",",
    "，": ",",
    "。": ".",
    "！": "!",
    "？": "?",
    "\n": ".",
    "·": ",",
    "、": ",",
    "...": ".",
    "…": ".",
    "$": ".",
    "“": "",
    "”": "",
    "‘": "",
    "’": "",
    "（": "",
    "）": "",
    "(": "",
    ")": "",
    "《": "",
    "》": "",
    "【": "",
    "】": "",
    "[": "",
    "]": "",
    "—": "",
    "～": "-",
    "~": "-",
    "「": "",
    "」": "",
    "¿" : "",
    "¡" : ""
}

_special_map = [(re.compile('%s' % x[0]), x[1]) for x in [
    ('ø', 'ɸ'),
    ('\u0303', '~'),
    ('ɜ', 'ʒ'),
    ('ɑ̃', 'ɑ~'),
]]

def collapse_whitespace(text):
    # Regular expression matching whitespace:
    _whitespace_re = re.compile(r"\s+")
    return re.sub(_whitespace_re, " ", text).strip()

def remove_punctuation_at_begin(text):
    return re.sub(r'^[,.!?]+', '', text)

def remove_aux_symbols(text):
    text = re.sub(r"[\<\>\(\)\[\]\"\«\»]+", "", text)
    return text

def replace_symbols(text):
    text = text.replace(";", ",")
    text = text.replace("-", " ")
    text = text.replace(":", ",")
    return text

def replace_punctuation(text):
    pattern = re.compile("|".join(re.escape(p) for p in rep_map.keys()))
    replaced_text = pattern.sub(lambda x: rep_map[x.group()], text)
    return replaced_text

def text_normalize(text):
    text = replace_punctuation(text)
    text = replace_symbols(text)
    text = remove_aux_symbols(text)
    text = remove_punctuation_at_begin(text)
    text = collapse_whitespace(text)
    text = re.sub(r'([^\.,!\?\-…])$', r'\1', text)
    return text

# special map
def special_map(text):
    for regex, replacement in _special_map:
        text = re.sub(regex, replacement, text)
    return text

def german_to_ipa(text, text_tokenizer):
    text = text_normalize(text)
    phonemes = text_tokenizer(text.strip())
    phonemes = '|'.join(phonemes)
    return phonemes

'''
    Phoneme merge time
'''
connect_list = [
    [46, 41],                   # "ɑ~"
    [18, 58],                   # "aʊ"
    [21, 644],                  # "eː"
    [54, 644],                  # "ɸː"
    [39, 644],                  # "yː"
    [20, 60],                   # "dʒ"
    [34, 33],                   # "ts"
    [47, 54],                   # "ɔɸ"
    [31, 644],                  # "oː"
    [21, 51],                   # "eɪ"
    [46, 644],                  # "ɑː"
    [18, 51],                   # "aɪ"
    [32, 22],                   # "pf"
    [25, 644],                  # "iː"
    [49, 644],                  # "ɛː"
    [35, 644],                  # "uː"
] 


final_result = [
    640,                    # "ɑ~"
    658,                    # "aʊ"
    679,                    # "eː"
    680,                    # "ɸː"
    681,                    # "yː"
    669,                    # "dʒ"
    81,                     # "ts"
    682,                    # "ɔɸ"
    674,                    # "oː"
    97,                     # "eɪ"
    645,                    # "ɑː"
    96,                     # "aɪ"
    683,                    # "pf"
    672,                    # "iː"
    676,                    # "ɛː"
    665,                    # "uː"
]

def _connect_phone(phoneme_tokens, vocab):

    separator = ',' + str(663) + ','
    token_str = ','.join(map(str, phoneme_tokens))
    token_str = separator + token_str + separator
    for idx, sub in enumerate(connect_list):
        sub_str = separator + ','.join(map(str, sub)) + separator
        if sub_str in token_str:
            replace_str = separator + str(final_result[idx]) + separator
            token_str = token_str.replace(sub_str, replace_str)
    token_str = token_str.replace(separator, ',')[1:-1]
    result_tokens = list(map(int, token_str.split(',')))
    del token_str
    return result_tokens

# Convert IPA to token
def german_merge_phoneme(phoneme_tokens, vocab):

    phoneme_tokens = _connect_phone(phoneme_tokens, vocab)
    # print('merge phoneme: ', phoneme_tokens)

    return phoneme_tokens