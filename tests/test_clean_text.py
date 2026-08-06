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
    
    # Test classification and modification splits
    assert clean_corpus_text("High classifi cation performance.") == "High classification performance."
    assert clean_corpus_text("A modifi cation of the setup.") == "A modification of the setup."
    assert clean_corpus_text("A new model is defi ning the layer.") == "A new model is defining the layer."


def test_remove_metadata_and_layout():
    text = (
        "This is valid content.\n"
        "00-Swanson\\_FM.indd vi\n"
        "This is also valid content.\n"
        "5/28/2011 9:40:34 AM\n"
        "2026-07-20 23:30:00\n"
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


def test_remove_indesign_proof_metadata():
    # Swanson_Ch-08.indd 177 contains internal hyphens and should be stripped
    text = (
        "This is valid content.\n"
        "08-Swanson\\_Ch-08.indd 177\n"
        "Final valid sentence."
    )
    cleaned = clean_corpus_text(text)
    expected = "This is valid content.\nFinal valid sentence."
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
    
    # PDF source should slice generically
    pns_cleaned = remove_inline_references(text, "Principles of Neural Science.pdf")
    assert pns_cleaned == "This is narrative text about the brain."
    
    other_cleaned = remove_inline_references(text, "Cognitive Psychology A Student's Handbook.pdf")
    assert other_cleaned == "This is narrative text about the brain."
    
    # fineweb-edu source should NOT slice
    fineweb_cleaned = remove_inline_references(text, "fineweb-edu")
    assert "## Selected Reading" in fineweb_cleaned


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
    
    # fineweb-edu source should bypass and return False
    assert is_standalone_index_or_bibliography(bib_text, "fineweb-edu") is False


def test_copyright_and_front_matter():
    copyright_page = (
        "Copyright © 2013 by Swanson LLC. All rights reserved.\n"
        "No part of this publication may be reproduced, stored in a retrieval system, "
        "or transmitted in any form or by any means."
    )
    # Should be skipped entirely (return empty string)
    assert clean_corpus_text(copyright_page, "Book.pdf") == ""


def test_dehyphenation():
    # Line-wrap dehyphenation of dictionary words should join without hyphen
    assert clean_corpus_text("sup- ported") == "supported"
    assert clean_corpus_text("defend-\n\ning") == "defending"
    assert clean_corpus_text("organi-\n\nzation") == "organization"
    assert clean_corpus_text("interventricu-\n\nlar") == "interventricular"
    
    # Genuine compound terms should keep hyphen (even when split across line-break or with same-line space)
    assert clean_corpus_text("corticotropin-\nreleasing") == "corticotropin-releasing"
    assert clean_corpus_text("corticotropin- releasing") == "corticotropin-releasing"
    assert clean_corpus_text("striatal- ventrolateral") == "striatal-ventrolateral"
    assert clean_corpus_text("cholecystokinin- tetrapeptide") == "cholecystokinin-tetrapeptide"
    assert clean_corpus_text("GABA- ergic") == "GABA-ergic"


def test_control_characters():
    assert clean_corpus_text("approximately \x01 70 mV") == "approximately -70 mV"
    assert clean_corpus_text("2 \x01 2 factorial design") == "2 x 2 factorial design"
    assert clean_corpus_text("Kluver \x01 Bucy syndrome") == "Kluver-Bucy syndrome"


def test_spaced_caps_and_pua():
    assert clean_corpus_text("S A N T I A G O   R A M Ó N   Y   C A J A L \uf6ae 1909 \uf6af") == "SANTIAGO RAMÓN Y CAJAL (1909)"



def test_spaced_digits():
    assert clean_corpus_text("The year (1 8 4 3) and (1 9 0 9).") == "The year (1843) and (1909)."


def test_dict_ligatures():
    # Suffix splits
    assert clean_corpus_text("identify ing") == "identifying"
    assert clean_corpus_text("see ing") == "seeing"
    assert clean_corpus_text("identifi able") == "identifiable"
    assert clean_corpus_text("sett ing") == "setting"
    assert clean_corpus_text("gett ing") == "getting"
    
    # Scientific prefix splits
    assert clean_corpus_text("neuro science") == "neuroscience"
    assert clean_corpus_text("opto genetics") == "optogenetics"
    assert clean_corpus_text("bio acoustics") == "bioacoustics"


def test_cross_page_metadata_dehyphenation():
    # Intervening metadata/page number should be removed first, then dehyphenated
    text = (
        "interventricu-\n"
        "08-Swanson\\_Ch-08.indd 177\n"
        "177\n"
        "lar"
    )
    # Should first remove Swanson and 177, then join interventricu-\nlar -> interventricular
    assert clean_corpus_text(text, "Book.pdf") == "interventricular"


def test_clean_corpus_text_integrated():
    text = (
        "This is narrative text about the brain.\n"
        "## Selected Reading\n"
        "- Kandel E. R. (2013). Principles of Neural Science."
    )
    # With PDF source -> should remove references
    assert clean_corpus_text(text, "Principles of Neural Science.pdf") == "This is narrative text about the brain."
    
    # Standalone index text
    index_text = (
        "## INDEX\n"
        "Abducens nerve, 143, 146\n"
        "Accessory nerve, 147\n"
    )
    # Should return empty string for standalone index
    assert clean_corpus_text(index_text, "Book.pdf") == ""
