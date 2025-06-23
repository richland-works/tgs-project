from tgs_project.backend.solr import SolrClient, SolrSchema
import json
import os
from tqdm import tqdm
from tgs_project.utils import read_jsonl, get_jsonl_n_count
from tgs_project.pipeline.pipeline import DataBatch

try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    print("dotenv not installed, skipping environment variable loading.")

# Pick up the NER results from the environment variable
# or use the default path
NER_RESULTS_PATH = os.getenv("NER_RESULTS_PATH", "ner_results.json")
if not os.path.exists(NER_RESULTS_PATH):
    raise FileNotFoundError(f"File {NER_RESULTS_PATH} not found.")
COREF_RESULTS_PATH = os.getenv("COREF_RESULTS_PATH", "coref_results.jsonl")
if not os.path.exists(COREF_RESULTS_PATH):
    raise FileNotFoundError(f"File {COREF_RESULTS_PATH} not found.")
SENTENCE_RESULTS_PATH = os.getenv("SENTENCE_RESULTS_PATH", "sentence_results.jsonl")
if not os.path.exists(SENTENCE_RESULTS_PATH):
    raise FileNotFoundError(f"File {SENTENCE_RESULTS_PATH} not found.")

# Set the page size for Solr
SOLR_INDEXING_PAGE_SIZE = int(os.getenv("SOLR_INDEXING_PAGE_SIZE", 1000))

# Pick up the Solr URL from the environment variable
# or use the default path
SOLR_URL = os.getenv("SOLR_URL", "http://localhost:8983/solr")
if not SOLR_URL:
    raise ValueError("SOLR_URL environment variable not set. Please set it to the Solr URL.")

# Initialize Solr client and schema
client = SolrClient(SOLR_URL)
collection = client.get_or_create_core("articles")
collection.delete_all_documents()

schema = SolrSchema(collection)

# We want to be able to do n-gram processing, so we need to create the type:
text_n_gram_type = {
        "name": "text_word_ngrams",
        "class": "solr.TextField",
        "analyzer": {
            "tokenizer": { "class": "solr.StandardTokenizerFactory" },
            "filters": [
                { "class": "solr.LowerCaseFilterFactory" },
                {
                    "class": "solr.ShingleFilterFactory",
                    "minShingleSize": "3",
                    "maxShingleSize": "5",
                    "outputUnigrams": "false"
                }
            ]
        }
    }
text_no_stop_words = {
    "name": "text_no_stop_words",
    "class": "solr.TextField",
    # Optional but nice: keeps phrase queries from leaking across sentence / value boundaries
    "positionIncrementGap": "100",
    "analyzer": {
        "tokenizer": { "class": "solr.StandardTokenizerFactory" },
        "filters": [
            { "class": "solr.LowerCaseFilterFactory" },
            {                       # -- word diet starts here
                "class": "solr.StopFilterFactory",
                "ignoreCase": "true"  # case-insensitive
                # Drop “words” and “format” to use Solr’s built-in list at lang/stopwords_en.txt.
                # Add them back if you need a custom file or Snowball format.
            }
        ]
    }
}


# Add the n-gram type to the schema
schema.create_custom_field_type(text_n_gram_type, raise_if_exists=False)
schema.create_custom_field_type(text_no_stop_words, raise_if_exists=False)

schema.add_field({"name": "id", "type": "string", "stored": True, "required": True}, raise_if_exists=False)
schema.add_field({"name": "article", "type": "text_general", "stored": True}, raise_if_exists=False)
schema.add_field({"name": "article_ngrams", "type": "text_word_ngrams", "stored": True}, raise_if_exists=False)
schema.add_field({"name": "article_no_stop_words", "type": "text_no_stop_words", "stored": True}, raise_if_exists=False)
schema.add_field({"name": "resolved_text", "type": "text_general", "stored": True}, raise_if_exists=False)
schema.add_field({"name": "sentences", "type": "text_general", "multiValued": True, "stored": True}, raise_if_exists=False)
schema.add_field({"name": "sentences_str", "type": "string", "multiValued": True, "stored": True}, raise_if_exists=False)
schema.add_field({"name": "highlights", "type": "text_general", "multiValued": False, "stored": True}, raise_if_exists=False)
schema.add_field({"name": "ner_PERSON", "type": "strings", "multiValued": True, "stored": True}, raise_if_exists=False)
schema.add_field({"name": "ner_ORG", "type": "strings", "multiValued": True, "stored": True}, raise_if_exists=False)
schema.add_field({"name": "ner_GPE", "type": "strings", "multiValued": True, "stored": True}, raise_if_exists=False)
schema.add_field({"name": "ner_DATE", "type": "strings", "multiValued": True, "stored": True}, raise_if_exists=False)
schema.add_field({"name": "ner_MONEY", "type": "strings", "multiValued": True, "stored": True}, raise_if_exists=False)
schema.add_field({"name": "ner_QUANTITY", "type": "strings", "multiValued": True, "stored": True}, raise_if_exists=False)
schema.add_field({"name": "ner_ORDINAL", "type": "strings", "multiValued": True, "stored": True}, raise_if_exists=False)
schema.add_field({"name": "ner_CARDINAL", "type": "strings", "multiValued": True, "stored": True}, raise_if_exists=False)
schema.add_field({"name": "ner_WORK_OF_ART", "type": "strings", "multiValued": True, "stored": True}, raise_if_exists=False)
schema.add_field({"name": "ner_EVENT", "type": "strings", "multiValued": True, "stored": True}, raise_if_exists=False)
schema.add_field({"name": "ner_LAW", "type": "strings", "multiValued": True, "stored": True}, raise_if_exists=False)
schema.create_copy_fields(copy_field_pairs=[
        ("article", "article_ngrams"),
        ("sentences", "sentences_str"),
    ],
    raise_if_exists=False

)


# First add the main docs with the NERs
with open(NER_RESULTS_PATH, "r") as f:
    data = json.load(f)

all_docs = list(data.items())

for i in tqdm(range(0, len(all_docs), SOLR_INDEXING_PAGE_SIZE), desc="Uploading to Solr"):
    docs = []
    for doc_id, item in all_docs[i:i+SOLR_INDEXING_PAGE_SIZE]:
        docs.append({
            "id": doc_id,
            "article": item["article"],
            "resolved_text": item.get("resolved_text", ""),
            "ner_PERSON": [ _[0] for _ in item.get('entities', []) if _[1] == "PERSON"],
            "ner_ORG": [ _[0] for _ in item.get('entities', []) if _[1] == "ORG"],
            "ner_GPE": [ _[0] for _ in item.get('entities', []) if _[1] == "GPE"],
            "ner_DATE": [ _[0] for _ in item.get('entities', []) if _[1] == "DATE"],
            "ner_MONEY": [ _[0] for _ in item.get('entities', []) if _[1] == "MONEY"],
            "ner_QUANTITY": [ _[0] for _ in item.get('entities', []) if _[1] == "QUANTITY"],
            "ner_ORDINAL": [ _[0] for _ in item.get('entities', []) if _[1] == "ORDINAL"],
            "ner_CARDINAL": [ _[0] for _ in item.get('entities', []) if _[1] == "CARDINAL"],
            "ner_WORK_OF_ART": [ _[0] for _ in item.get('entities', []) if _[1] == "WORK_OF_ART"],
            "ner_EVENT": [ _[0] for _ in item.get('entities', []) if _[1] == "EVENT"],
            "ner_LAW": [ _[0] for _ in item.get('entities', []) if _[1] == "LAW"],
            "sentences": item.get("sentences", []),
            "highlights": item.get("highlights","")
        })
    collection.add_documents(docs)

# Now add the coref results
all_docs = DataBatch(
    data=read_jsonl(COREF_RESULTS_PATH),
    count=get_jsonl_n_count(COREF_RESULTS_PATH) // SOLR_INDEXING_PAGE_SIZE,
    processed_ids=set()
)

missing_coref = {}
for batch in tqdm(all_docs.batched(SOLR_INDEXING_PAGE_SIZE), total=all_docs.count, desc="Uploading coref results to Solr"):
    docs = []
    for item in batch:
        if not item.get("resolved_text"):
            missing_coref[item["id"]] = item
            continue
        docs.append({
            "id": item.get("id"),
            "resolved_text": item.get("resolved_text", [])
        })
    collection.add_documents(docs, atomic=True)
if missing_coref:
    print(f"{len(missing_coref)} docs are missing coref results")

# Now add the sentence results
all_docs = DataBatch(
    data=read_jsonl(SENTENCE_RESULTS_PATH),
    count=get_jsonl_n_count(SENTENCE_RESULTS_PATH) // SOLR_INDEXING_PAGE_SIZE,
    processed_ids=set()
)

missing_sentences = {}
for batch in tqdm(all_docs.batched(SOLR_INDEXING_PAGE_SIZE), total=all_docs.count, desc="Uploading sentence results to Solr"):
    docs = []
    for item in batch:
        if not item.get("sentences"):
            missing_sentences[item["id"]] = item
            continue
        docs.append({
            "id": item.get("id"),
            "sentences": item.get("sentences", [])
        })
    collection.add_documents(docs, atomic=True)

if missing_sentences:
    print(f"{len(missing_sentences)} docs are missing sentences")
