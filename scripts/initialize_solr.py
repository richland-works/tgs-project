from tgs_project.backend.solr import SolrClient, SolrSchema
import json
import os
from tqdm import tqdm
from tgs_project.utils import read_jsonl, get_jsonl_n_count
from tgs_project.pipeline.pipeline import DataBatch
from tgs_project.utils import context_windows, tokenize_merge_phrases
import uuid

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
context_windows_collection = client.get_or_create_core("context_windows")
collection.delete_all_documents()
context_windows_collection.delete_all_documents()

schema = SolrSchema(collection)
context_windows_schema = SolrSchema(context_windows_collection)

schema.add_field({"name": "id", "type": "string", "stored": True, "required": True}, raise_if_exists=False)
schema.add_field({"name": "article", "type": "text_general", "stored": True, "indexed": True}, raise_if_exists=False)
schema.add_field({"name": "article_no_stop_words", "type": "text_general", "indexed": True, "stored": True}, raise_if_exists=False)
schema.add_field({"name": "resolved_text", "type": "text_general", "stored": True, "indexed": True}, raise_if_exists=False)
schema.add_field({"name": "sentences", "type": "text_general", "multiValued": True, "stored": True, "indexed": True}, raise_if_exists=False)
schema.add_field({"name": "sentences_str", "type": "string", "multiValued": True, "stored": True}, raise_if_exists=False)
schema.add_field({"name": "highlights", "type": "text_general", "multiValued": False, "stored": True, "indexed": True}, raise_if_exists=False)
schema.add_field({"name": "ner_PERSON", "type": "strings", "multiValued": True, "stored": True, "indexed": True}, raise_if_exists=False)
schema.add_field({"name": "ner_ORG", "type": "strings", "multiValued": True, "stored": True, "indexed": True}, raise_if_exists=False)
schema.add_field({"name": "ner_GPE", "type": "strings", "multiValued": True, "stored": True, "indexed": True}, raise_if_exists=False)
schema.add_field({"name": "ner_DATE", "type": "strings", "multiValued": True, "stored": True, "indexed": True}, raise_if_exists=False)
schema.add_field({"name": "ner_MONEY", "type": "strings", "multiValued": True, "stored": True, "indexed": True}, raise_if_exists=False)
schema.add_field({"name": "ner_QUANTITY", "type": "strings", "multiValued": True, "stored": True, "indexed": True}, raise_if_exists=False)
schema.add_field({"name": "ner_ORDINAL", "type": "strings", "multiValued": True, "stored": True, "indexed": True}, raise_if_exists=False)
schema.add_field({"name": "ner_CARDINAL", "type": "strings", "multiValued": True, "stored": True, "indexed": True}, raise_if_exists=False)
schema.add_field({"name": "ner_WORK_OF_ART", "type": "strings", "multiValued": True, "stored": True, "indexed": True}, raise_if_exists=False)
schema.add_field({"name": "ner_EVENT", "type": "strings", "multiValued": True, "stored": True, "indexed": True}, raise_if_exists=False)
schema.add_field({"name": "ner_LAW", "type": "strings", "multiValued": True, "stored": True, "indexed": True}, raise_if_exists=False)
schema.create_copy_fields(copy_field_pairs=[
        ("sentences", "sentences_str"),
    ],
    raise_if_exists=False

)

# Now add the n-grams collection
context_windows_schema.add_field({"name": "article_context_windows", "type": "text_general", "indexed": True, "stored": True}, raise_if_exists=False)
context_windows_schema.add_field({"name": "article_context_windows_str", "type": "string", "indexed": True, "stored": True}, raise_if_exists=False)
context_windows_schema.add_field({"name": "resolved_context_windows", "type": "text_general", "indexed": True, "stored": True}, raise_if_exists=False)
context_windows_schema.add_field({"name": "resolved_context_windows_str", "type": "string", "indexed": True, "stored": True}, raise_if_exists=False)
context_windows_schema.add_field({"name": "article_id", "type": "string", "indexed": True, "stored": True}, raise_if_exists=False)
context_windows_schema.create_copy_fields(copy_field_pairs=[
        ("article_context_windows", "article_context_windows_str"),
        ("resolved_context_windows", "resolved_context_windows_str")
    ],
    raise_if_exists=False
)


# First add the main docs with the NERs
# Now add the coref results
all_docs = DataBatch(
    data=read_jsonl(NER_RESULTS_PATH),
    count=get_jsonl_n_count(NER_RESULTS_PATH) // SOLR_INDEXING_PAGE_SIZE,
    processed_ids=set()
)

missing_coref = {}
for batch in tqdm(all_docs.batched(SOLR_INDEXING_PAGE_SIZE), total=all_docs.count, desc="Uploading coref results to Solr"):
    docs = []
    context_windows_docs = []
    for item in batch:
        docs.append({
            "id": item.get("id"),
            "article": item["article"],
            "article_no_stop_words": " ".join(
                tokenize_merge_phrases(
                    item["article"],
                    remove_stop_words=True,
                    remove_punct=False,
                    merge_punct=True)
                ),
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

        for context_window in context_windows(item["article"], ner_token_list=[_[0] for _ in item.get("entities", [])], window=2):
            context_windows_docs.append({
                "article_context_windows": context_window,
                "article_id": item.get("id"),
                "id": str(uuid.uuid5(uuid.NAMESPACE_DNS, item.get("id") + context_window))
            })
        for context_window in context_windows(item.get("resolved_text", ""), ner_token_list=[_[0] for _ in item.get("entities", [])], window=2):
            context_windows_docs.append({
                "resolved_context_windows": context_window,
                "article_id": item.get("id"),
                "id": str(uuid.uuid5(uuid.NAMESPACE_DNS, item.get("id") + context_window))
            })
    collection.add_documents(docs)
    context_windows_collection.add_documents(context_windows_docs)

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
            "resolved_text": item.get("resolved_text", ""),
        })
    collection.add_documents(docs, atomic=True)
    context_windows_collection.add_documents(context_windows_docs)

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
