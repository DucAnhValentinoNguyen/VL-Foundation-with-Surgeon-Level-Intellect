"""
prompts.py
==========

Class-label vocabulary and zero-shot prompt templates for evaluating
SigLIP2 on the HyperKvasir (hkv_subsample_p2) gastrointestinal endoscopy
dataset.

Why this file exists
--------------------
SigLIP2 is a contrastive vision-language model: its zero-shot accuracy
depends *strongly* on how class names are phrased. Raw folder names like
``bbps-2-3`` or ``esophagitis-b-d`` are not in the model's pretraining
vocabulary, so we map every folder name to (a) a clinically meaningful
natural-language label, and (b) a list of prompt templates that get
ensembled (averaged in embedding space) at evaluation time. This is the
standard CLIP/SigLIP "prompt ensembling" recipe (Radford et al., 2021;
Tschannen et al., 2025 -- arXiv:2502.14786).

Public API
----------
* ``CLASS_LABELS`` -- canonical mapping folder_name -> human description
* ``TEMPLATES_GENERIC`` -- domain-agnostic prompt templates
* ``TEMPLATES_ENDOSCOPY`` -- domain-conditioned prompt templates
* ``build_prompts(label, templates)`` -- expand a label through templates
"""

from __future__ import annotations
from typing import Dict, List

# ---------------------------------------------------------------------------
# Canonical natural-language labels per folder name.
#
# Names are deliberately *descriptive* rather than abbreviation-laden so the
# text encoder has the best chance of grounding them. Where the folder name
# encodes an ordinal grading (BBPS, UC), the textual label includes the grade
# in words.
# ---------------------------------------------------------------------------
CLASS_LABELS: Dict[str, str] = {
    # --- lower GI / anatomical landmarks ------------------------------------
    "cecum":                       "the cecum, the first part of the large intestine",
    "ileum":                       "the ileum, the final section of the small intestine",
    "retroflex-rectum":            "a retroflex view of the rectum",
    # --- lower GI / pathological findings -----------------------------------
    "hemorrhoids":                 "hemorrhoids in the rectum",
    "polyps":                      "a colon polyp",
    "ulcerative-colitis-grade-0-1": "ulcerative colitis of Mayo grade 0 to 1",
    "ulcerative-colitis-grade-1":   "ulcerative colitis of Mayo grade 1",
    "ulcerative-colitis-grade-1-2": "ulcerative colitis of Mayo grade 1 to 2",
    "ulcerative-colitis-grade-2":   "ulcerative colitis of Mayo grade 2",
    "ulcerative-colitis-grade-2-3": "ulcerative colitis of Mayo grade 2 to 3",
    "ulcerative-colitis-grade-3":   "ulcerative colitis of Mayo grade 3, severe inflammation",
    # --- lower GI / mucosal-view quality ------------------------------------
    "bbps-0-1":      "poorly cleaned colon mucosa, Boston Bowel Preparation Scale 0 to 1",
    "bbps-2-3":      "well cleaned colon mucosa, Boston Bowel Preparation Scale 2 to 3",
    "impacted-stool": "impacted stool obscuring the colon mucosa",
    # --- lower GI / therapeutic interventions -------------------------------
    "dyed-lifted-polyps":     "a colon polyp lifted with submucosal blue dye injection",
    "dyed-resection-margins": "blue dyed margins after endoscopic polyp resection",
    # --- upper GI / anatomical landmarks ------------------------------------
    "pylorus":            "the pylorus, the opening from the stomach into the duodenum",
    "retroflex-stomach":  "a retroflex view of the stomach interior",
    "z-line":             "the Z-line, the gastroesophageal junction",
    # --- upper GI / pathological findings -----------------------------------
    "barretts":               "Barrett's esophagus",
    "barretts-short-segment": "short-segment Barrett's esophagus",
    "esophagitis-a":          "esophagitis Los Angeles grade A",
    "esophagitis-b-d":        "esophagitis Los Angeles grade B to D",
    # --- additional classes only found in labeled-videos --------------------
    "BBPS-0-1":                       "poorly cleaned colon mucosa, Boston Bowel Preparation Scale 0 to 1",
    "BBPS-2-3":                       "well cleaned colon mucosa, Boston Bowel Preparation Scale 2 to 3",
    "anastomotic-leakage":            "an anastomotic leakage in the bowel",
    "colitis":                        "colitis, inflammation of the colon",
    "parasites":                      "intestinal parasites visible in the bowel",
    "self-expanding-stents":          "a self-expanding stent placed in the bowel",
    "snare-resection":                "endoscopic snare resection of a polyp",
    "cancer":                         "gastrointestinal cancer tissue",
    "esophagitis":                    "esophagitis, inflammation of the esophagus",
    "gastric-antral-vascular-ectasia": "gastric antral vascular ectasia, watermelon stomach",
    "ulcer":                          "a gastrointestinal ulcer",
}

# ---------------------------------------------------------------------------
# Prompt templates.
#
# Following Radford et al. (CLIP, 2021), we use ensembles of templates and
# average the resulting *normalized* text embeddings. Tschannen et al. (2025)
# show SigLIP2 benefits similarly from ensembling.
# ---------------------------------------------------------------------------
TEMPLATES_GENERIC: List[str] = [
    "a photo of {}.",
    "an image of {}.",
    "a picture of {}.",
    "this is {}.",
]

TEMPLATES_ENDOSCOPY: List[str] = [
    "an endoscopic image of {}.",
    "a gastrointestinal endoscopy photograph showing {}.",
    "a colonoscopy image showing {}.",
    "an endoscopy frame showing {}.",
    "a medical endoscopy photograph of {}.",
    "a video frame from a gastrointestinal endoscopy showing {}.",
    "a high-definition endoscopy image of {}.",
]


def build_prompts(label: str, templates: List[str]) -> List[str]:
    """Apply each template to a canonical label.

    Parameters
    ----------
    label
        Natural-language label, e.g. ``"a colon polyp"``.
    templates
        List of format strings each containing a single ``{}`` placeholder.
    """
    return [t.format(label) for t in templates]


# ---------------------------------------------------------------------------
# Convenience: groupings used by the EDA / benchmark scripts
# ---------------------------------------------------------------------------
# Closed-set image classes (the ones the labeled-images split actually
# provides). Useful when running zero-shot classification *without* the
# extra video-only classes acting as distractors.
IMAGE_CLASSES = [
    "cecum", "ileum", "retroflex-rectum",
    "hemorrhoids", "polyps",
    "ulcerative-colitis-grade-0-1", "ulcerative-colitis-grade-1",
    "ulcerative-colitis-grade-1-2", "ulcerative-colitis-grade-2",
    "ulcerative-colitis-grade-2-3", "ulcerative-colitis-grade-3",
    "bbps-0-1", "bbps-2-3", "impacted-stool",
    "dyed-lifted-polyps", "dyed-resection-margins",
    "pylorus", "retroflex-stomach", "z-line",
    "barretts", "barretts-short-segment",
    "esophagitis-a", "esophagitis-b-d",
]

# Classes which only appear in videos -- used for the open-set zero-shot
# stress test.
VIDEO_ONLY_CLASSES = [
    "anastomotic-leakage", "colitis", "parasites",
    "self-expanding-stents", "snare-resection",
    "cancer", "esophagitis", "gastric-antral-vascular-ectasia", "ulcer",
]
