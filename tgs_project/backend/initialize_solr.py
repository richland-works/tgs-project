from tgs_project.backend.solr import SolrClient, SolrSchema
import json
from tqdm import tqdm

client = SolrClient("http://localhost:8983/solr")
collection = client.get_core("articles")

schema = SolrSchema(collection)

schema.add_field({"name": "id", "type": "string", "stored": True, "required": True})
schema.add_field({"name": "article", "type": "text_general", "stored": True})
schema.add_field({"name": "resolved_text", "type": "text_general", "stored": True})
schema.add_field({"name": "ner_PERSON", "type": "strings", "multiValued": True, "stored": True})
schema.add_field({"name": "ner_ORG", "type": "strings", "multiValued": True, "stored": True})
schema.add_field({"name": "ner_GPE", "type": "strings", "multiValued": True, "stored": True})
schema.add_field({"name": "ner_DATE", "type": "strings", "multiValued": True, "stored": True})
schema.add_field({"name": "sentences", "type": "text_general", "multiValued": True, "stored": True})
schema.add_field({"name": "highlights", "type": "text_general", "multiValued": False, "stored": True})


# Load the data
with open("C:\\Users\\richa\\Documents\\tgs-project\\notebooks\\ner_results.json", "r") as f:
    data = json.load(f)

all_docs = list(data.items())
page_size = 1000

for i in tqdm(range(0, len(all_docs), page_size), desc="Uploading to Solr"):
    docs = []
    for doc_id, item in all_docs[i:i+page_size]:
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