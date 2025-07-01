import spacy
from tqdm.auto import tqdm
from tgs_project.utils import write_jsonl, read_jsonl
import os

try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    print("dotenv not installed, skipping environment variable loading.")

class ArticleEntityProcessor:
    """
    A class to process articles and extract named entities using spaCy.
    Attributes:
        model_name (str): The name of the spaCy model to use.
        batch_size (int): The batch size for processing.
        use_gpu (bool): Whether to use GPU for processing.
        results_path (str): The path to save the results.
        flush_interval (int): The interval at which to flush results to disk.
    TO DO:
        - Add support for multiple models (e.g., en_core_web_sm, en_core_web_md, etc.)
        - Add support for custom models.
        - Create n grams while extracting entities.
    """
    def __init__(
        self,
        model_name="en_core_web_trf",
        batch_size: int=128,
        use_gpu: bool=True,
        results_path: str = os.getenv("NER_RESULTS_PATH", "ner_results.jsonl"),
        flush_interval: int = 512,
    ):
        if use_gpu:
            spacy.require_gpu() # type: ignore
        self.nlp = spacy.load(model_name)
        self.batch_size = batch_size
        self.results_path = results_path
        self.flush_interval = flush_interval

    def ents_of(self, doc):
        return [(e.text, e.label_) for e in doc.ents]

    def process(self, articles) -> None:
        """
        Process a list of articles and extract entities.
        Args:
            articles (list): List of articles to process.
        All articles should be in the format:
        {
            "id": <article_id>,
            "article": <article_text>,
            "highlights": <highlights>
        }
        Returns:
            None
        Writes the results to a JSONL file.
        Each line in the file will be a JSON object with the following keys:
            - id: The article ID
            - entities: A list of tuples (entity_text, entity_label)
            - highlights: The highlights of the article
            - article: The original article text
        """
        # Check if results already exist
        if os.path.exists(self.results_path):
            processed_ids = {item["id"] for item in read_jsonl(self.results_path)}
            print(f"Already processed {len(processed_ids)} articles... skipping them.")
            articles = [item for item in articles if item["id"] not in processed_ids]
        text_items_stream = ((item["article"], item) for item in articles)
        stream_w_progress = tqdm(text_items_stream, desc="NER", total=len(articles))
        print(f"Processing {len(articles)} articles...")
        results = []
        for doc, item in self.nlp.pipe(stream_w_progress, as_tuples=True, batch_size=self.batch_size):
            results.append({
                "id": item["id"],
                "entities": self.ents_of(doc),
                "highlights": item["highlights"],
                "article": item["article"],
            })
            if len(results) >= self.flush_interval:
                write_jsonl(self.results_path, results)
                results = []
        if results:
            write_jsonl(self.results_path, results)
