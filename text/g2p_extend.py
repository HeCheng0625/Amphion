import re
import unicodedata
from g2p_en import G2p
from g2p_en.expand import normalize_numbers
import copy

PHONE_SET = [
    "!",
    ",",
    ".",
    ".B",
    ":",
    "<BOS>",
    "<EOS>",
    "<PAD>",
    "<UNK>",
    "?",
    "AA0B",
    "AA0E",
    "AA0I",
    "AA1B",
    "AA1E",
    "AA1I",
    "AA2B",
    "AA2E",
    "AA2I",
    "AE0B",
    "AE0E",
    "AE0I",
    "AE1B",
    "AE1E",
    "AE1I",
    "AE2B",
    "AE2E",
    "AE2I",
    "AH0B",
    "AH0E",
    "AH0I",
    "AH1B",
    "AH1E",
    "AH1I",
    "AH2B",
    "AH2E",
    "AH2I",
    "AO0B",
    "AO0E",
    "AO0I",
    "AO1",
    "AO1B",
    "AO1E",
    "AO1I",
    "AO2B",
    "AO2E",
    "AO2I",
    "AW0B",
    "AW0E",
    "AW0I",
    "AW1B",
    "AW1E",
    "AW1I",
    "AW2B",
    "AW2E",
    "AW2I",
    "AY0B",
    "AY0E",
    "AY0I",
    "AY1B",
    "AY1E",
    "AY1I",
    "AY2B",
    "AY2E",
    "AY2I",
    "BB",
    "BE",
    "BI",
    "CHB",
    "CHE",
    "CHI",
    "DB",
    "DE",
    "DHB",
    "DHE",
    "DHI",
    "DI",
    "EH0B",
    "EH0E",
    "EH0I",
    "EH1B",
    "EH1E",
    "EH1I",
    "EH2B",
    "EH2E",
    "EH2I",
    "ER0B",
    "ER0E",
    "ER0I",
    "ER1B",
    "ER1E",
    "ER1I",
    "ER2B",
    "ER2E",
    "ER2I",
    "EY0B",
    "EY0E",
    "EY0I",
    "EY1B",
    "EY1E",
    "EY1I",
    "EY2B",
    "EY2E",
    "EY2I",
    "FB",
    "FE",
    "FI",
    "GB",
    "GE",
    "GI",
    "HHB",
    "HHE",
    "HHI",
    "IH0B",
    "IH0E",
    "IH0I",
    "IH1B",
    "IH1E",
    "IH1I",
    "IH2B",
    "IH2E",
    "IH2I",
    "IY0B",
    "IY0E",
    "IY0I",
    "IY1B",
    "IY1E",
    "IY1I",
    "IY2B",
    "IY2E",
    "IY2I",
    "JHB",
    "JHE",
    "JHI",
    "KB",
    "KE",
    "KI",
    "L",
    "LB",
    "LE",
    "LI",
    "MB",
    "ME",
    "MI",
    "NB",
    "NE",
    "NGB",
    "NGE",
    "NGI",
    "NI",
    "OW0B",
    "OW0E",
    "OW0I",
    "OW1B",
    "OW1E",
    "OW1I",
    "OW2B",
    "OW2E",
    "OW2I",
    "OY0B",
    "OY0E",
    "OY0I",
    "OY1B",
    "OY1E",
    "OY1I",
    "OY2B",
    "OY2E",
    "OY2I",
    "PB",
    "PE",
    "PI",
    "RB",
    "RE",
    "RI",
    "SB",
    "SE",
    "SHB",
    "SHE",
    "SHI",
    "SI",
    "TB",
    "TE",
    "THB",
    "THE",
    "THI",
    "TI",
    "UH0B",
    "UH0E",
    "UH0I",
    "UH1B",
    "UH1E",
    "UH1I",
    "UH2E",
    "UH2I",
    "UW0B",
    "UW0E",
    "UW0I",
    "UW1B",
    "UW1E",
    "UW1I",
    "UW2B",
    "UW2E",
    "UW2I",
    "VB",
    "VE",
    "VI",
    "WB",
    "WE",
    "WI",
    "YB",
    "YE",
    "YI",
    "ZB",
    "ZE",
    "ZHB",
    "ZHE",
    "ZHI",
    "ZI",
    "|",
]
PHPONE2ID = {PHONE_SET[i]: i for i in range(len(PHONE_SET))}

PUNCS = "!,.?;:"


def is_sil_phoneme(p):
    return p == "" or not p[0].isalpha()


def add_bdr(txt_struct):
    txt_struct_ = []
    for i, ts in enumerate(txt_struct):
        txt_struct_.append(ts)
        if (
            i != len(txt_struct) - 1
            and not is_sil_phoneme(txt_struct[i][0])
            and not is_sil_phoneme(txt_struct[i + 1][0])
        ):
            txt_struct_.append(["|", ["|"]])
    return txt_struct_


def preprocess_text(text):
    text = normalize_numbers(text)
    text = "".join(
        char
        for char in unicodedata.normalize("NFD", text)
        if unicodedata.category(char) != "Mn"
    )  # Strip accents
    text = text.lower()
    text = re.sub("['\"()]+", "", text)
    text = re.sub("[-]+", " ", text)
    text = re.sub(f"[^ a-z{PUNCS}]", "", text)
    text = re.sub(f" ?([{PUNCS}]) ?", r"\1", text)  # !! -> !
    text = re.sub(f"([{PUNCS}])+", r"\1", text)  # !! -> !
    text = text.replace("i.e.", "that is")
    text = text.replace("i.e.", "that is")
    text = text.replace("etc.", "etc")
    text = re.sub(f"([{PUNCS}])", r" \1 ", text)
    text = re.sub(rf"\s+", r" ", text)
    return text


def postprocess(txt_struct):
    while len(txt_struct) > 0 and is_sil_phoneme(txt_struct[0][0]):
        txt_struct = txt_struct[1:]
    while len(txt_struct) > 0 and is_sil_phoneme(txt_struct[-1][0]):
        txt_struct = txt_struct[:-1]
    txt_struct = add_bdr(txt_struct)
    txt_struct = txt_struct + [["<EOS>", ["<EOS>"]]]
    return txt_struct


def process(txt, g2p):
    txt = preprocess_text(txt).strip()
    phs = g2p(txt)
    txt_struct = [[w, []] for w in txt.split(" ")]
    i_word = 0
    for p in phs:
        if p == " ":
            i_word += 1
        else:
            txt_struct[i_word][1].append(p)

    txt_struct_ret = copy.deepcopy(txt_struct)

    for i_word in range(len(txt_struct)):
        if not is_sil_phoneme(txt_struct[i_word][0]):
            if len(txt_struct[i_word][1]) > 1:
                txt_struct_ret[i_word][1][0] += "B"
                for i in range(1, len(txt_struct[i_word][1]) - 1):
                    txt_struct_ret[i_word][1][i] += "I"
                txt_struct_ret[i_word][1][-1] += "E"
            else:
                txt_struct_ret[i_word][1][0] += "B"

    txt_struct_ret = postprocess(txt_struct_ret)

    return txt_struct_ret, txt
