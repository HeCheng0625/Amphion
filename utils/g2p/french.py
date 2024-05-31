'''https://github.com/bootphon/phonemizer'''
import re

'''
    Text clean time
'''
# List of (regular expression, replacement) pairs for abbreviations in french:
_abbreviations = [
    (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
    for x in [
        ("M", "monsieur"),
        ("Mlle", "mademoiselle"),
        ("Mlles", "mesdemoiselles"),
        ("Mme", "Madame"),
        ("Mmes", "Mesdames"),
        ("N.B", "nota bene"),
        ("M", "monsieur"),
        ("p.c.q", "parce que"),
        ("Pr", "professeur"),
        ("qqch", "quelque chose"),
        ("rdv", "rendez-vous"),
        ("max", "maximum"),
        ("min", "minimum"),
        ("no", "numéro"),
        ("adr", "adresse"),
        ("dr", "docteur"),
        ("st", "saint"),
        ("co", "companie"),
        ("jr", "junior"),
        ("sgt", "sergent"),
        ("capt", "capitain"),
        ("col", "colonel"),
        ("av", "avenue"),
        ("av. J.-C", "avant Jésus-Christ"),
        ("apr. J.-C", "après Jésus-Christ"),
        ("art", "article"),
        ("boul", "boulevard"),
        ("c.-à-d", "c’est-à-dire"),
        ("etc", "et cetera"),
        ("ex", "exemple"),
        ("excl", "exclusivement"),
        ("boul", "boulevard"),
    ]
] + [
    (re.compile("\\b%s" % x[0]), x[1])
    for x in [
        ("Mlle", "mademoiselle"),
        ("Mlles", "mesdemoiselles"),
        ("Mme", "Madame"),
        ("Mmes", "Mesdames"),
    ]
]

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
    ('ã', 'a~'),
    ('ɑ̃', 'ɑ~'),
    ('ɔ̃', 'ɔ~'),
    ('ɛ̃', 'ɛ~'),
    ('œ̃', 'œ~'),
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
    text = text.replace("&", " et ")
    return text

def expand_abbreviations(text):
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    return text

def replace_punctuation(text):
    pattern = re.compile("|".join(re.escape(p) for p in rep_map.keys()))
    replaced_text = pattern.sub(lambda x: rep_map[x.group()], text)
    return replaced_text

def text_normalize(text):
    text = expand_abbreviations(text)
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

def french_to_ipa(text, text_tokenizer):
    text = text_normalize(text)
    phonemes = text_tokenizer(text.strip())
    phonemes = '|'.join(phonemes)
    return phonemes

'''
    Phoneme merge time
'''
connect_list = [
    [47, 644],                  # "ɔː"
    [18, 644],                  # "aː"
    [25, 644],                  # "iː"
    [31, 644],                  # "oː"
    [19, 41],                   # "a~"
    [46, 41],                   # "ɑ~"
    [47, 41],                   # "ɔ~"
    [49, 41],                   # "ɛ~"
    [636, 41]                   # "œ~"
] 


final_result = [
    656,                # "ɔː"
    638,                # "aː"
    672,                # "iː"
    674,                # "oː"
    639,                # "a~"
    640,                # "ɑ~"
    641,                # "ɔ~"
    642,                # "ɛ~"
    643                 # "œ~"
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
def french_merge_phoneme(phoneme_tokens, vocab):

    phoneme_tokens = _connect_phone(phoneme_tokens, vocab)
    # print('merge phoneme: ', phoneme_tokens)

    return phoneme_tokens
