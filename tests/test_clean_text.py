from lib.utils.clean_text import clean_corpus_text

def test_clean_ligatures():
    # Test th / Th splits at the start of words
    assert clean_corpus_text("Th e brain is complex.") == "The brain is complex."
    assert clean_corpus_text("th is is a test.") == "this is a test."
    assert clean_corpus_text("Th ey are studying neuroscience.") == "They are studying neuroscience."
    assert clean_corpus_text("Th ree methods were used.") == "Three methods were used."
    
    # Test th splits inside words
    assert clean_corpus_text("The o th er pathway.") == "The other pathway."
    assert clean_corpus_text("Whether or not.") == "Whether or not."
    
    # Test ff splits
    assert clean_corpus_text("We need to study the eff ect of neurotransmitters.") == "We need to study the effect of neurotransmitters."
    assert clean_corpus_text("This is an eff ective model.") == "This is an effective model."
    assert clean_corpus_text("The results diff er.") == "The results differ."
    assert clean_corpus_text("They suff er from deficits.") == "They suffer from deficits."
    
    # Test fl splits
    assert clean_corpus_text("It will refl ect the basic plan.") == "It will reflect the basic plan."
    assert clean_corpus_text("The refl ex arc is simple.") == "The reflex arc is simple."
    assert clean_corpus_text("Under the infl uence of drugs.") == "Under the influence of drugs."
    
    # Test fi splits
    assert clean_corpus_text("High affi nity binding.") == "High affinity binding."
    assert clean_corpus_text("The offi ce of research.") == "The office of research."
    assert clean_corpus_text("A specifi c region of the brain.") == "A specific region of the brain."
    assert clean_corpus_text("A new model is defi ned.") == "A new model is defined."
    assert clean_corpus_text("In the fi rst chapter.") == "In the first chapter."
    assert clean_corpus_text("They fi nd new evidence.") == "They find new evidence."
    
    # Test physiological splits
    assert clean_corpus_text("This is a physi ological response.") == "This is a physiological response."


def test_remove_metadata_and_layout():
    text = (
        "This is valid content.\n"
        "00-Swanson\\_FM.indd vi\n"
        "This is also valid content.\n"
        "5/28/2011 9:40:34 AM\n"
        "This page intentionally left blank\n"
        "ix\n"
        "125\n"
        "Final valid sentence."
    )
    
    cleaned = clean_corpus_text(text)
    
    expected = (
        "This is valid content.\n"
        "This is also valid content.\n"
        "Final valid sentence."
    )
    assert cleaned == expected


def test_unescape_and_strip_placeholders():
    text = (
        "The brain &amp; the mind.\n"
        "<!-- image -->\n"
        "Some details here.\n"
        "```\n"
        "stray fence line\n"
        "```\n"
    )
    cleaned = clean_corpus_text(text)
    assert "&amp;" not in cleaned
    assert "The brain & the mind." in cleaned
    assert "<!-- image -->" not in cleaned
    assert "```" not in cleaned
    assert "stray fence line" in cleaned


def test_remove_inline_references():
    from lib.utils.clean_text import remove_inline_references
    
    text = (
        "This is narrative text about the brain.\n"
        "## Selected Reading\n"
        "- Kandel E. R. (2013). Principles of Neural Science."
    )
    
    # Target book should slice
    pns_cleaned = remove_inline_references(text, "Principles of Neural Science.pdf")
    assert pns_cleaned == "This is narrative text about the brain."
    
    # Non-target book should NOT slice
    non_target_cleaned = remove_inline_references(text, "Other Book.pdf")
    assert "## Selected Reading" in non_target_cleaned


def test_is_standalone_index_or_bibliography():
    from lib.utils.clean_text import is_standalone_index_or_bibliography
    
    # Standalone index chunk
    index_text = (
        "## INDEX\n"
        "Abducens nerve, 143, 146\n"
        "Accessory nerve, 147\n"
    )
    assert is_standalone_index_or_bibliography(index_text, "Book.pdf") is True
    
    # Standalone bibliography chunk
    bib_text = (
        "Sanes JR, Jessell TM. 2013. Synapse formation. In: Kandel ER (eds). Principles of Neural Science. New York: McGraw-Hill.\n"
        "Young et al. 2001. Distribution of vasopressin. Academic Press."
    )
    assert is_standalone_index_or_bibliography(bib_text, "Book.pdf") is True
    
    # Normal prose
    prose_text = (
        "The brain contains billions of neurons that communicate via synapses.\n"
        "This is an introductory chapter on neuroscience."
    )
    assert is_standalone_index_or_bibliography(prose_text, "Book.pdf") is False

