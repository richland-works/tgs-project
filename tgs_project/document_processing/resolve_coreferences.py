import os
import spacy
from tqdm import tqdm
from fastcoref import spacy_component
from tgs_project.utils import write_jsonl, read_jsonl

try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    print("dotenv not installed, skipping environment variable loading.")

class CorefResolver:
    def __init__(
        self,
        model_name: str = "en_core_web_trf",
        batch_size: int = 128,
        results_path: str = os.getenv("COREF_RESULTS_PATH", "coref_results.json"),
        flush_interval: int = 512,
        use_gpu: bool = True,
    ):
        if use_gpu:
            spacy.require_gpu()  # type: ignore
        self.nlp = spacy.load(model_name)
        self.nlp.add_pipe("fastcoref")  # type: ignore
        self.batch_size = batch_size
        self.results_path = results_path
        self.flush_interval = flush_interval

        if not os.path.exists(self.results_path):
            with open(self.results_path, "w") as f:
                pass  # Create empty file

    @staticmethod
    def resolve_corefs(text, clusters):
        replacements = []
        for cluster in clusters:
            if not cluster:
                continue
            head = text[cluster[0][0]:cluster[0][1]]
            for mention in cluster[1:]:
                if mention is None:
                    continue
                start, end = mention
                replacements.append((start, end, head))
        replacements.sort(reverse=True, key=lambda x: x[0])
        for start, end, head in replacements:
            text = text[:start] + head + text[end:]
        return text

    def process(self, articles):
        processed_ids = {item["id"] for item in read_jsonl(self.results_path)}
        articles_to_process = [item for item in articles if item["id"] not in processed_ids]

        print(f"Already processed {len(processed_ids)} articles... skipping them.")
        print(f"Processing {len(articles_to_process)} articles...")

        text_item_stream = ((item["article"],item) for item in articles_to_process)
        results = []
        stream_w_progress = tqdm(text_item_stream, desc="Coref", total=len(articles_to_process))

        # Process articles in batches
        for doc, item in self.nlp.pipe(stream_w_progress, as_tuples=True, batch_size=self.batch_size, n_process=1):
            clusters = doc._.coref_clusters
            resolved_text = self.resolve_corefs(doc.text, clusters)
            results.append({
                "id": item["id"],
                "resolved_text": resolved_text,
                "article": item["article"],
                "highlights": item["highlights"],
            })
            if len(results) >= self.flush_interval:
                write_jsonl(self.results_path, results)
                results = []
        if results:
            write_jsonl(self.results_path, results)
