import pytest
from lib.utils.clean_text import clean_corpus_text

def test_space_padded_dehyphenation():
    # Tests same-line hyphens with spaces around them: word - word, word -word
    assert clean_corpus_text("The study of con -form behavior.") == "The study of conform behavior."
    assert clean_corpus_text("The motiva -tion of human behavior.") == "The motivation of human behavior."
    assert clean_corpus_text("We are evalu -ating the tech -nological progress.") == "We are evaluating the technological progress."
    assert clean_corpus_text("Detailed anal -ysis of the dataset.") == "Detailed analysis of the dataset."
    assert clean_corpus_text("Classical condition -ing experiment.") == "Classical conditioning experiment."


def test_control_character_stripping():
    # Tests non-printable ASCII control characters (\x03, \x04, \x1f, \x05, \xad, \ufeff)
    text = "Neuroscience\x03 is \x04the \x1fstudy \x05of \xadthe \ufeffbrain."
    cleaned = clean_corpus_text(text)
    assert cleaned == "Neuroscience is the study of the brain."
    assert "\x03" not in cleaned
    assert "\x04" not in cleaned
    assert "\x1f" not in cleaned
    assert "\xfeff" not in cleaned


def test_formula_placeholder_stripping():
    # Tests HTML comments and formula placeholders
    text = "The membrane potential is calculated as <!-- formula-not-decoded --> where V is voltage."
    cleaned = clean_corpus_text(text)
    assert cleaned == "The membrane potential is calculated as where V is voltage."
    assert "<!-- formula-not-decoded -->" not in cleaned


def test_pua_unicode_normalization():
    # Tests Private Use Area unicode replacement
    text = "Symbol \uf6ae parentheses \uf6af and unmapped \uf8e5 character."
    cleaned = clean_corpus_text(text)
    assert "(parentheses)" in cleaned
    assert "\uf8e5" not in cleaned


def test_markdown_table_stripping():
    # Tests raw markdown table line stripping
    table_text = (
        "| About the authors | Preface | Acknowledgement |\n"
        "|---|---|---|\n"
        "| Author 1 | Pref 1 | Ack 1 |\n"
        "| Author 2 | Pref 2 | Ack 2 |\n"
    )
    # High proportion of table lines -> should drop chunk
    assert clean_corpus_text(table_text, "Book.pdf") == ""


def test_publisher_metadata_stripping():
    # Tests ISBN, DOI, and typesetter metadata lines
    text = (
        "ISBN: 978-0-367-74647-6 (hbk)\n"
        "DOI: 10.4324/9781003158899\n"
        "Typeset in Goudy by codeMantra\n"
        "This is valid neuroscience narrative prose explaining cortical functions."
    )
    cleaned = clean_corpus_text(text)
    assert "ISBN:" not in cleaned
    assert "DOI:" not in cleaned
    assert "codeMantra" not in cleaned
    assert "This is valid neuroscience narrative prose" in cleaned


def test_isolated_diagram_callout_filtering():
    # Tests isolated short callout lines
    text = (
        "This is narrative paragraph text describing the hippocampal circuit.\n"
        "M\n"
        "1.0\n"
        "=\n"
        "The dentate gyrus projects via mossy fibers to CA3 pyramidal neurons."
    )
    cleaned = clean_corpus_text(text)
    assert "M\n" not in cleaned
    assert "1.0\n" not in cleaned
    assert "=\n" not in cleaned
    assert "This is narrative paragraph text describing the hippocampal circuit." in cleaned
    assert "The dentate gyrus projects via mossy fibers to CA3 pyramidal neurons." in cleaned


def test_figure_caption_line_stripping():
    text = (
        "This is narrative text about motor pathways.\n"
        "FIGURE 7.1. Cross section of the spinal cord showing fissures and sulci.\n"
        "The corticospinal tract descends directly to lower motor neurons."
    )
    cleaned = clean_corpus_text(text)
    assert "FIGURE 7.1" not in cleaned
    assert "This is narrative text about motor pathways." in cleaned
    assert "The corticospinal tract descends directly to lower motor neurons." in cleaned


def test_inline_figure_reference_stripping():
    text = "The motor cortex regulates movement (see Figure 4-2) via pyramidal tracts (Fig. 3B)."
    cleaned = clean_corpus_text(text)
    assert "(see Figure 4-2)" not in cleaned
    assert "(Fig. 3B)" not in cleaned
    assert "The motor cortex regulates movement via pyramidal tracts." in cleaned


def test_toc_and_unheadinged_citation_filtering():
    toc_chunk = (
        "## 26. Memory and Amnesia 329\n"
        "Introduction 329 The Case of H.M. 329 Anatomy of Medial Temporal Amnesia 331 Pathologies Causing Medial Temporal Amnesia 332\n"
        "Material-Specific Anterograde Amnesia 332 Remote Memory and Retrograde Amnesia 332\n"
    )
    # High proportion of TOC lines -> should return empty
    assert clean_corpus_text(toc_chunk, "Book.pdf") == ""

    citation_chunk = (
        "- Yoon, D. Y., Gause, C. D., & Singer, H. S. (2007). Frontal dopaminergic abnormality. Journal of Neurological Sciences, 255(1), 50-56.\n"
        "- Thomas, N. J. T. (1989) Experience and Theory. American Journal of Psychology 102: 395-412.\n"
        "- Smith, A. (2012) Memory consolidation. Brain Research Press, Vol. 14, pp. 20-30.\n"
    )
    # Unheadinged citation list -> should drop chunk
    assert clean_corpus_text(citation_chunk, "Book.pdf") == ""
