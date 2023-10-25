import contextlib
import io
import sys

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider

@contextlib.contextmanager
def suppress_stdout():
    """Suppress stdout context manager."""
    current_stdout = sys.stdout
    sys.stdout = io.StringIO()  # Temporary redirect stdout to a fake stream
    try:
        yield
    finally:
        sys.stdout = current_stdout

def evaluate_score(pred_text, actual_text):
    """
    Compute CIDEr score using the Cider class from pycocoevalcap.
    
    Args:
        pred_text (list of str): List of predicted captions.
        actual_text (list of str): List of actual captions.
        
    Returns:
        float: CIDEr score.
    """
    # Ensure the predicted captions and actual captions are lists
    if not isinstance(pred_text, list):
        pred_text = [pred_text]
    if not isinstance(actual_text, list):
        actual_text = [actual_text]

    # Format predictions and actuals for CIDEr evaluation
    gts = {}
    res = {}
    for i, (pred, actual) in enumerate(zip(pred_text, actual_text)):
        # 文字列に変換
        pred = str(pred)
        actual = str(actual)
        gts[i] = [actual]
        res[i] = [pred]

    # Instantiate the CIDEr evaluator object
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Cider(), "CIDEr")
    ]

    final_scores = {}
    with suppress_stdout():
        for scorer, metric in scorers:
            score, scores = scorer.compute_score(gts, res)
            if type(score) == list:
                for m, s in zip(metric, score):
                    final_scores[m] = s
            else:
                final_scores[metric] = score

    return final_scores['CIDEr'], final_scores['Bleu_4']