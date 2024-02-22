# Explore how U.S. Congress have spoken about 'workers'
# 11k docs

import textacy.datasets
from textacy import extract
from textacy import preprocessing as preproc

dataset = textacy.datasets.CapitolWords()
dataset.info
{'name': 'capitol_words',
 'site_url': 'http://sunlightlabs.github.io/Capitol-Words/',
 'description': 'Collection of ~11k speeches in the Congressional Record given by notable U.S. politicians between Jan 1996 and Jun 2016.'}
dataset.download()

# Each record contains full text of speech and basic metadata
record = next(dataset.records(limit=1))
record

# Extract only specific parts of interest
textacy.set_doc_extensions("extract")  # just setting these now -- we'll use them later!

# As a first step, inspect our keywords in context
list(extract.keyword_in_context(record.text, "work(ing|ers?)", window_width=35))

# preprocess the text to get rid of potential data quality issues and other distractions that may affect our analysis
preprocessor = preproc.make_pipeline(
    preproc.normalize.unicode,
    preproc.normalize.quotation_marks,
    preproc.normalize.whitespace,
)
preproc_text = preprocessor(record.text)
preproc_text[:200]

# changes are “destructive” — can’t reconstruct the original without keeping a copy around or re-loading it from disk


# make a spaCy Doc by applying a language-specific model pipeline to the text
doc = textacy.make_spacy_doc((preproc_text, record.meta), lang="en_core_web_sm")
doc._.preview
doc._.meta


# get a sense of how 'workers' are described using annotated part-of-speech tags
# extract just the adjectives and determinants immediately preceding our keyword
patterns = [
    {
        "POS": {
            "IN": ["ADJ", "DET"]
            },
        "OP": "+"
    },
    {
        "ORTH": {
            "REGEX": "workers?"
            }
    }
]
token_matches = extract.token_matches(doc, patterns)
list(token_matches)

# examples aren’t very interesting. would like results aggregated over all speeches: skilled workers, American workers, young workers...

# To accomplish this, load many records into a textacy.Corpus

records = dataset.records(limit=500)
preproc_records = ((preprocessor(text), meta) for text, meta in records)
corpus = textacy.Corpus("en_core_web_sm", data=preproc_records)
print(corpus)

# get a better sense of what’s in our corpus by leveraging the documents’ metadata
import collections

date = corpus.agg_metadata("date", min), corpus.agg_metadata("date", max)
speaker_name = corpus.agg_metadata("speaker_name", collections.Counter)

print(date)
print(speaker_name)

# extract matches from each processed document

import itertools

matches = itertools.chain.from_iterable(extract.token_matches(doc, patterns) for doc in corpus)

# lemmatize their texts for consistency
# inspect the most common descriptions of workers
collections.Counter(match.lemma_ for match in matches).most_common(20)

# To better understand the context of these mentions, extract keyterms (the most important or “key” terms)

corpus[0]._.extract_keyterms("textrank", normalize="lemma", window_size=10, edge_weighting="count", topn=10)

# Now, select the subset of speeches in which “worker(s)” were mentioned
docs_mentioning_workers = corpus.get(lambda doc: any(doc._.extract_regex_matches("workers?")))

# extract the keyterms from each and aggregaate
kt_weights = collections.Counter()

for doc in docs_mentioning_workers:
  keyterms = doc._.extract_keyterms(
      "textrank", normalize="lemma",
      window_size=10,
      edge_weighting="count",
      topn=10
  )
  kt_weights.update(dict(keyterms))

# rank the results
kt_weights.most_common(20)

# we can see from the list that 'workers' are brought up in discussion of jobs, the minimum wage, and trust funds. Makes sense!