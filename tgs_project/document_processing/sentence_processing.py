import spacy
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from tgs_project.utils import write_jsonl, load_cnn_dailymail
from tgs_project.pipeline.pipeline import Parallel, Stage

# ── spaCy worker init (called once per process) ───────────────────────────────
def _init_worker():
    global nlp
    import spacy
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")

# ── pipeline stages ───────────────────────────────────────────────────────────
def _ensure_dict(x):
    return {"text":x} if isinstance(x,str) else x

def _tokenize(rec):
    d=_ensure_dict(rec)
    d["doc"]=nlp(d["text"]) # type: ignore
    return d
tokenize = Stage(_tokenize)

def _sent_cnt(rec):
    rec["sentences"]=list(rec["doc"].sents)
    return rec
count_sentences = Stage(_sent_cnt)

def _drop_doc(rec):
    rec.pop("doc", None)
    return rec
drop_doc = Stage(_drop_doc)

sentence_count_pipeline = tokenize | count_sentences | drop_doc          # composed pipeline

# ── main driver ───────────────────────────────────────────────────────────────
def main():
    import datasets

    dataset = load_cnn_dailymail() # type: ignore
    texts = dataset['train'][:10]['article']

    with open("sentence_counts.jsonl", "w") as f:
        results = texts | Parallel(
            sentence_count_pipeline,
            initializer=_init_worker,
            chunksize=100
        )
        write_jsonl(f, results)  # type: ignore

if __name__ == "__main__":
    mp.freeze_support()
    main()