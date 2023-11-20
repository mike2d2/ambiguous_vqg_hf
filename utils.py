import spacy
import re
spacy_tagger = spacy.load("en_core_web_sm")

def get_nps(question):
    doc = spacy_tagger(question)
    nps = []
    noun_gex = re.compile(r"^(NOUN ?)+$") 
    for span_start in range(0, len(doc)-1):
        for span_len in range(0, len(doc)-span_start-1, 1):
            span_end = span_start + span_len
            span = doc[span_start:span_end]
            tag_span = " ".join([tok.pos_ for tok in span])
            text_span = " ".join([tok.text for tok in span]) 
            text_span_is_subset = any([text_span in x for x in nps])
            if noun_gex.match(tag_span) is not None and text_span not in nps and not text_span_is_subset: 
                nps.append(text_span)
    final_nps = []
    # remove duplicates
    nps = list(set(nps))
    # remove subsets: 
    for i, text_span_a in enumerate(nps): 
        skip = False
        for j, text_span_b in enumerate(nps): 
            if i == j:
                continue
            if text_span_a in text_span_b:
                skip = True
        if not skip:
            final_nps.append(text_span_a)

    return final_nps 
