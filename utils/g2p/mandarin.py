"""from https://github.com/Plachtaa/VALL-E-X/g2p"""
import re
import jieba
import cn2an

'''
    Text clean time
'''
# List of (Latin alphabet, bopomofo) pairs:
_latin_to_bopomofo = [(re.compile('%s' % x[0], re.IGNORECASE), x[1]) for x in [
    ('a', 'ㄟˉ'),
    ('b', 'ㄅㄧˋ'),
    ('c', 'ㄙㄧˉ'),
    ('d', 'ㄉㄧˋ'),
    ('e', 'ㄧˋ'),
    ('f', 'ㄝˊㄈㄨˋ'),
    ('g', 'ㄐㄧˋ'),
    ('h', 'ㄝˇㄑㄩˋ'),
    ('i', 'ㄞˋ'),
    ('j', 'ㄐㄟˋ'),
    ('k', 'ㄎㄟˋ'),
    ('l', 'ㄝˊㄛˋ'),
    ('m', 'ㄝˊㄇㄨˋ'),
    ('n', 'ㄣˉ'),
    ('o', 'ㄡˉ'),
    ('p', 'ㄆㄧˉ'),
    ('q', 'ㄎㄧㄡˉ'),
    ('r', 'ㄚˋ'),
    ('s', 'ㄝˊㄙˋ'),
    ('t', 'ㄊㄧˋ'),
    ('u', 'ㄧㄡˉ'),
    ('v', 'ㄨㄧˉ'),
    ('w', 'ㄉㄚˋㄅㄨˋㄌㄧㄡˋ'),
    ('x', 'ㄝˉㄎㄨˋㄙˋ'),
    ('y', 'ㄨㄞˋ'),
    ('z', 'ㄗㄟˋ')
]]

# List of (bopomofo, romaji) pairs:
_bopomofo_to_romaji = [(re.compile('%s' % x[0]), x[1]) for x in [
    ('ㄅㄛ', 'p⁼wo'),
    ('ㄆㄛ', 'pʰwo'),
    ('ㄇㄛ', 'mwo'),
    ('ㄈㄛ', 'fwo'),
    ('ㄅ', 'p⁼'),
    ('ㄆ', 'pʰ'),
    ('ㄇ', 'm'),
    ('ㄈ', 'f'),
    ('ㄉ', 't⁼'),
    ('ㄊ', 'tʰ'),
    ('ㄋ', 'n'),
    ('ㄌ', 'l'),
    ('ㄍ', 'k⁼'),
    ('ㄎ', 'kʰ'),
    ('ㄏ', 'h'),
    ('ㄐ', 'ʧ⁼'),
    ('ㄑ', 'ʧʰ'),
    ('ㄒ', 'ʃ'),
    ('ㄓ', 'ʦ`⁼'),
    ('ㄔ', 'ʦ`ʰ'),
    ('ㄕ', 's`'),
    ('ㄖ', 'ɹ`'),
    ('ㄗ', 'ʦ⁼'),
    ('ㄘ', 'ʦʰ'),
    ('ㄙ', 's'),
    ('ㄚ', 'a'),
    ('ㄛ', 'o'),
    ('ㄜ', 'ə'),
    ('ㄝ', 'e'),
    ('ㄞ', 'ai'),
    ('ㄟ', 'ei'),
    ('ㄠ', 'au'),
    ('ㄡ', 'ou'),
    ('ㄧㄢ', 'yeNN'),
    ('ㄢ', 'aNN'),
    ('ㄧㄣ', 'iNN'),
    ('ㄣ', 'əNN'),
    ('ㄤ', 'aNg'),
    ('ㄧㄥ', 'iNg'),
    ('ㄨㄥ', 'uNg'),
    ('ㄩㄥ', 'yuNg'),
    ('ㄥ', 'əNg'),
    ('ㄦ', 'əɻ'),
    ('ㄧ', 'i'),
    ('ㄨ', 'u'),
    ('ㄩ', 'ɥ'),
    ('ˉ', '→'),
    ('ˊ', '↑'),
    ('ˇ', '↓↑'),
    ('ˋ', '↓'),
    ('˙', ''),
    ('，', ','),
    ('。', '.'),
    ('！', '!'),
    ('？', '?'),
    ('—', '-')
]]

# List of (romaji, ipa) pairs:
_romaji_to_ipa = [(re.compile('%s' % x[0], re.IGNORECASE), x[1]) for x in [
    ('ʃy', 'ʃ'),
    ('ʧʰy', 'ʧʰ'),
    ('ʧ⁼y', 'ʧ⁼'),
    ('NN', 'n'),
    ('Ng', 'ŋ'),
    ('y', 'j'),
    ('h', 'x')
]]

# List of (bopomofo, ipa) pairs:
_bopomofo_to_ipa = [(re.compile('%s' % x[0]), x[1]) for x in [
    ('ㄅㄛ', 'p⁼wo'),
    ('ㄆㄛ', 'pʰwo'),
    ('ㄇㄛ', 'mwo'),
    ('ㄈㄛ', 'fwo'),
    ('ㄧㄢ', 'jɛn'),
    ('ㄩㄢ', 'ɥæn'),
    ('ㄧㄣ', 'in'),
    ('ㄩㄣ', 'ɥn'),
    ('ㄧㄥ', 'iŋ'),
    ('ㄨㄥ', 'ʊŋ'),
    ('ㄩㄥ', 'jʊŋ'),
    # Add
    ('ㄧㄚ', 'ia'),
    ('ㄧㄝ', 'iɛ'),
    ('ㄧㄠ', 'iɑʊ'),
    ('ㄧㄡ', 'ioʊ'),
    ('ㄧㄤ', 'iɑŋ'),
    ('ㄨㄚ', 'ua'),
    ('ㄨㄛ', 'uo'),
    ('ㄨㄞ', 'uaɪ'),
    ('ㄨㄟ', 'ueɪ'),
    ('ㄨㄢ', 'uan'),
    ('ㄨㄣ', 'uən'),
    ('ㄨㄤ', 'uɑŋ'),
    ('ㄩㄝ', 'ɥɛ'),
    # End
    ('ㄅ', 'p⁼'),
    ('ㄆ', 'pʰ'),
    ('ㄇ', 'm'),
    ('ㄈ', 'f'),
    ('ㄉ', 't⁼'),
    ('ㄊ', 'tʰ'),
    ('ㄋ', 'n'),
    ('ㄌ', 'l'),
    ('ㄍ', 'k⁼'),
    ('ㄎ', 'kʰ'),
    ('ㄏ', 'x'),
    ('ㄐ', 'tʃ⁼'),
    ('ㄑ', 'tʃʰ'),
    ('ㄒ', 'ʃ'),
    ('ㄓ', 'ts`⁼'),
    ('ㄔ', 'ts`ʰ'),
    ('ㄕ', 's`'),
    ('ㄖ', 'ɹ`'),
    ('ㄗ', 'ts⁼'),
    ('ㄘ', 'tsʰ'),
    ('ㄙ', 's'),
    ('ㄚ', 'a'),
    ('ㄛ', 'o'),
    ('ㄜ', 'ə'),
    ('ㄝ', 'ɛ'),
    ('ㄞ', 'aɪ'),
    ('ㄟ', 'eɪ'),
    ('ㄠ', 'ɑʊ'),
    ('ㄡ', 'oʊ'),
    ('ㄢ', 'an'),
    ('ㄣ', 'ən'),
    ('ㄤ', 'ɑŋ'),
    ('ㄥ', 'əŋ'),
    ('ㄦ', 'əɻ'),
    ('ㄧ', 'i'),
    ('ㄨ', 'u'),
    ('ㄩ', 'ɥ'),
    ('ˉ', '→'),
    ('ˊ', '↑'),
    ('ˇ', '↓↑'),
    ('ˋ', '↓'),
    ('˙', ''),
    ('，', ','),
    ('。', '.'),
    ('！', '!'),
    ('？', '?'),
    ('—', '-'),
    ('《', '<'),
    ('》', '>'),
]]

finals_list = [
    [32, 66, 37, 31],           # "p⁼wo"
    [32, 61, 37, 31],           # "pʰwo"
    [29, 37, 31],               # "mwo"
    [22, 37, 31],               # "fwo"
    [26, 49, 30],               # "jɛn"
    [50, 42, 30],               # "ɥæn"
    [25, 30],                   # "in"
    [50, 30],                   # "ɥn"
    [25, 45],                   # "iŋ"
    [26, 58, 45],               # "jʊŋ"
    [58, 45],                   # "ʊŋ"
    [26, 18],                   # "ja"
    [25, 49],                   # "iɛ"
    [25, 46, 58],               # "iɑʊ"
    [26, 31, 58],               # "joʊ"
    [25, 46, 45],               # "iɑŋ"
    [37, 18, 51],               # "waɪ"
    [37, 21, 51],               # "weɪ"
    [37, 18, 30],               # "wan"
    [37, 48, 30],               # "wən"
    [35, 46, 45],               # "uɑŋ"
    [37, 18],                   # "wa"
    [37, 31],                   # "wo"
    [50, 49],                   # "ɥɛ"
    [32, 66],                   # "p⁼"
    [32, 61],                   # "pʰ"
    [34, 66],                   # "t⁼"
    [34, 61],                   # "tʰ"
    [27, 66],                   # "k⁼"
    [27, 61],                   # "kʰ"
    [34, 57, 66],               # "tʃ⁼"
    [34, 57, 61],               # "tʃʰ"
    [34, 33, 17, 66, 55, 17],   # "ts`⁼ɹ`"
    [34, 33, 17, 66],           # "ts`⁼"
    [34, 33, 17, 61, 55, 17],   # "ts`ʰɹ`"
    [34, 33, 17, 61],           # "ts`ʰ"
    [33, 17, 55, 17],           # "s`ɹ`"
    [33, 17],                   # "s`"
    [55, 17, 55, 17],           # "ɹ`ɹ`"
    [48, 55, 17],               # "əɹ`"
    [55, 17],                   # "ɹ`"
    [34, 33, 66, 55],           # "ts⁼ɹ"
    [34, 33, 66],               # "ts⁼"
    [34, 33, 61, 55],           # "tsʰɹ"
    [34, 33, 61],               # "tsʰ"
    [33, 55],                   # "sɹ"
    [18, 51],                   # "aɪ"
    [21, 51],                   # "eɪ"
    [46, 58],                   # "ɑʊ"
    [31, 58],                   # "oʊ"
    [18, 30],                   # "an"
    [48, 30],                   # "ən"
    [46, 45],                   # "ɑŋ"
    [48, 45],                   # "əŋ"
]

finals_result = [
    70,             # "p⁼wo"
    71,             # "pʰwo"
    72,             # "mwo"
    73,             # "fwo"
    101,            # "jɛn"
    103,            # "ɥæn"
    105,            # "in"
    106,            # "ɥn"
    109,            # "iŋ"
    111,            # "jʊŋ"
    110,            # "ʊŋ"
    114,            # "ja"
    115,            # "iɛ"
    116,            # "iɑʊ"
    117,            # "joʊ"
    118,            # "iɑŋ"
    120,            # "waɪ"
    121,            # "weɪ"
    122,            # "wan"
    123,            # "wən"
    124,            # "uɑŋ"
    119,            # "wa"
    67,             # "wo"
    125,            # "ɥɛ"
    68,             # "p⁼"
    69,             # "pʰ"
    74,             # "t⁼"
    75,             # "tʰ"
    76,             # "k⁼"
    77,             # "kʰ"
    79,             # "tʃ⁼"
    80,             # "tʃʰ"
    91,             # "ts`⁼ɹ`"
    85,             # "ts`⁼"
    92,             # "ts`ʰɹ`"
    86,             # "ts`ʰ"
    89,             # "s`ɹ`"
    87,             # "s`"
    90,             # "ɹ`ɹ`"
    113,            # "əɹ`"
    88,             # "ɹ`"
    93,             # "ts⁼ɹ"
    82,             # "ts⁼"
    94,             # "tsʰɹ"
    84,             # "tsʰ"
    95,             # "sɹ"
    96,             # "aɪ"
    97,             # "eɪ"
    98,             # "ɑʊ"
    99,             # "oʊ"
    104,            # "an"
    107,            # "ən"
    108,            # "ɑŋ"
    112,            # "əŋ"
]

# Convert numbers to Chinese pronunciation
def number_to_chinese(text):
    numbers = re.findall(r'\d+(?:\.?\d+)?', text)
    for number in numbers:
        text = text.replace(number, cn2an.an2cn(number), 1)
    return text

# Word Segmentation, and convert Chinese pronunciation to pinyin (bopomofo)
def chinese_to_bopomofo(text):
    from pypinyin import lazy_pinyin, BOPOMOFO
    text = text.replace('、', '，').replace('；', '，').replace('：', '，')
    words = jieba.lcut(text, cut_all=False)
    text = re.sub(r"\s+", "", text)
    text = ''
    for word in words:
        bopomofos = lazy_pinyin(word, BOPOMOFO)
        if not re.search('[\u4e00-\u9fff]', word):
            text += word
            continue
        for i in range(len(bopomofos)):
            bopomofos[i] = re.sub(r'([\u3105-\u3129])$', r'\1ˉ', bopomofos[i])
        if text != '':
            text += ' '
        text += ' '.join(bopomofos)
    return text

# Convert latin pronunciation to pinyin (bopomofo)
def latin_to_bopomofo(text):
    for regex, replacement in _latin_to_bopomofo:
        text = re.sub(regex, replacement, text)
    return text

# Convert pinyin (bopomofo) to Romaji (not used)
def bopomofo_to_romaji(text):
    for regex, replacement in _bopomofo_to_romaji:
        text = re.sub(regex, replacement, text)
    return text

# Convert pinyin (bopomofo) to IPA
def bopomofo_to_ipa(text):
    for regex, replacement in _bopomofo_to_ipa:
        text = re.sub(regex, replacement, text)
    return text

# Convert Chinese to Romaji (not used)
def chinese_to_romaji(text):
    text = number_to_chinese(text)
    text = chinese_to_bopomofo(text)
    text = latin_to_bopomofo(text)
    text = bopomofo_to_romaji(text)
    text = re.sub('i([aoe])', r'y\1', text)
    text = re.sub('u([aoəe])', r'w\1', text)
    text = re.sub('([ʦsɹ]`[⁼ʰ]?)([→↓↑ ]+|$)',
                  r'\1ɹ`\2', text).replace('ɻ', 'ɹ`')
    text = re.sub('([ʦs][⁼ʰ]?)([→↓↑ ]+|$)', r'\1ɹ\2', text)
    return text

# Convert Chinese to IPA
def chinese_to_ipa(text):
    text = number_to_chinese(text)
    text = chinese_to_bopomofo(text)
    text = latin_to_bopomofo(text)
    text = bopomofo_to_ipa(text)
    text = re.sub('i([aoe])', r'j\1', text)
    text = re.sub('u([aoəe])', r'w\1', text)
    text = re.sub('([sɹ]`[⁼ʰ]?)([→↓↑ ]+|$)',
                  r'\1ɹ`\2', text).replace('ɻ', 'ɹ`')
    text = re.sub('([s][⁼ʰ]?)([→↓↑ ]+|$)', r'\1ɹ\2', text)
    return text

'''
    Phoneme merge time
'''

def _connect_phone(phoneme_tokens, vocab):
    
    for i in range(len(finals_list)):
        for j in range(len(phoneme_tokens)):
            if phoneme_tokens[j:j+len(finals_list[i])] == finals_list[i]:
                phoneme_tokens[j] = finals_result[i]
                del phoneme_tokens[j+1:j+len(finals_list[i])]
    return phoneme_tokens

def _connect_tone(phoneme_tokens, vocab):

    tone_list = ["→", "↑", "↓↑", "↓"]
    tone_token = []
    last_single_token = 0
    base = 0
    pattern = r"\[[^\[\]]*\]"  # Exclude "[" and "]"
    for tone, idx in vocab.items():
        if re.match(pattern, tone):
            base = idx + 1
        if tone in tone_list:
            tone_token.append(idx)
            last_single_token = idx

    pre_token = None
    cur_token = None
    res_token = []
    for t in phoneme_tokens:
        cur_token = t
        if t in tone_token:
            cur_token = last_single_token + (pre_token - base) * len(tone_list) + tone_token.index(t) + 1
            res_token.pop()
        res_token.append(cur_token)
        pre_token = t

    return res_token

# Convert Chinese IPA to token
def chinese_merge_phoneme(phoneme_tokens, vocab):

    phoneme_tokens = _connect_phone(phoneme_tokens, vocab)
    # print('merge phoneme: ', phoneme_tokens)

    phoneme_tokens = _connect_tone(phoneme_tokens, vocab)
    # print('merge tones: ', phoneme_tokens)

    return phoneme_tokens


