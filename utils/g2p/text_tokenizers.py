import re
import os
from typing import List, Pattern, Union
from phonemizer.utils import list2str, str2list
from phonemizer.backend import EspeakBackend
from phonemizer.backend.espeak.language_switch import LanguageSwitch
from phonemizer.backend.espeak.words_mismatch import WordMismatch
from phonemizer.punctuation import Punctuation
from phonemizer.separator import Separator



class TextTokenizer:
    """Phonemize Text."""

    def __init__(
        self,
        language="en-us",
        backend="espeak",
        separator=Separator(word="|_|", syllable="-", phone="|"),
        preserve_punctuation=False,
        punctuation_marks: Union[str, Pattern] = Punctuation.default_marks(),
        with_stress: bool = False,
        tie: Union[bool, str] = False,
        language_switch: LanguageSwitch = "remove-flags",
        words_mismatch: WordMismatch = "ignore",
    ) -> None:
        self.backend = EspeakBackend(
            language,
            punctuation_marks=punctuation_marks,
            preserve_punctuation=preserve_punctuation,
            with_stress=with_stress,
            tie=tie,
            language_switch=language_switch,
            words_mismatch=words_mismatch,
        )
        
        self.separator = separator

    def __call__(self, text, strip=True) -> List[str]:

        text_type = type(text)
        text = [re.sub(r'[^\w\s_,\.\?!\|\']', '', line.strip()) for line in str2list(text)]
        phonemized = self.backend.phonemize(
            text, separator=self.separator, strip=strip, njobs=1
        )
        if text_type == str:
            return list2str(phonemized)
        return phonemized