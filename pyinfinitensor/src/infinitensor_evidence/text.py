"""Literal-preserving lexical terms and source offsets, without model imports."""
import re
import unicodedata


def positive(value, name, maximum=None):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(name+' must be a positive integer')
    if maximum is not None and value > maximum:
        raise ValueError(name+' exceeds '+str(maximum))
    return value


def terms(text):
    value=unicodedata.normalize('NFKC',text).casefold()
    words=re.findall(r'[a-z0-9_]+(?:[./:+-][a-z0-9_]+)*',value)
    result=[]
    for word in words:
        result.append(word)
        result.extend(part for part in re.split(r'[._/:+-]',word) if part and part!=word)
    for run in re.findall(r'[\u3400-\u9fff]+',value):
        result.extend(run)
        result.extend(run[i:i+2] for i in range(len(run)-1))
    # Encode literals so FTS does not reinterpret punctuation or operators.
    return ['t'+word.encode('utf-8').hex() for word in result]


def character_chunks(text, size=700, overlap=80):
    """CPU lexical fallback. Its units are Unicode code points, not tokens."""
    positive(size,'size')
    if not 0 <= overlap < size:
        raise ValueError('overlap must be smaller than size')
    result=[];start=0
    while start<len(text):
        end=min(start+size,len(text))
        if text[start:end].strip():result.append((start,end))
        if end==len(text):break
        start=end-overlap
    return result
