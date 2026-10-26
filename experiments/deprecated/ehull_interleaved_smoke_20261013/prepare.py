"""Ehull-conditioned generation and factual reading examples for one smoke run."""

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
START, END, MASK, PAD = ("<CRYSTAL_START>", "<CRYSTAL_END>", "<CRYSTAL_MASK>", "<CRYSTAL_PAD>")
EOS = "<|im_end|>"
K, STEPS, MAXLEN = 544, 32, 2048
PREDICTION_OFFSET = 0
READ_QUESTION = "Read the current crystal and report its formula, space group number, cell atom count, and independent orbit count."


def tasks(row):
    label = row["property_labels"]
    prefix = ("User: Generate a crystal conditioned on a target energy above hull of "
              f"{label['energy_above_hull']} {label['unit']}.\nAssistant:")
    lead = " " + row["structure_description"].strip() + "\n" + START
    crystal = row["plaid_text"]
    facts = row["core_facts"]
    reading_target = (
        f"The formula is {facts['cell_formula']}. "
        f"The space group number is {facts['space_group_number']}. "
        f"The unit cell contains {facts['cell_atom_count']} atoms and "
        f"{facts['independent_orbit_count']} independent orbits." + EOS
    )
    history = prefix + lead + crystal + END
    return [
        {"task": "G", "prefix": prefix, "lead": lead, "crystal": crystal},
        {"task": "Q", "prefix": history + "\nUser: " + READ_QUESTION + "\nAssistant:",
         "question": READ_QUESTION, "target": reading_target},
    ]
