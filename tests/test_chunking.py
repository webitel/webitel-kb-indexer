import json
import pathlib

import pytest

from kb_indexer import chunking

TESTDATA = pathlib.Path(__file__).parent / "testdata" / "chunking"

# Document-level cases.
CASES = [
    ("faq-rich", "Оплата і тарифи"),
    ("password-reset", "Скидання паролю"),
    ("hostile-author-text", None),
    ("long-article", "Умови користування"),
    ("code-runbook", "Ранбук індексації"),
    ("faq-single", "Чи є знижки при оплаті за рік?"),
]


def source_of(case):
    return (TESTDATA / case / "input.md").read_text(encoding="utf-8")


def recorded(case, produced, update):
    """The chunks recorded for a case, rewritten first when asked."""
    path = TESTDATA / case / "expected.json"
    if update:
        path.write_text(json.dumps(produced, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if not path.exists():
        pytest.fail(f"{path} is missing; run pytest --update-golden")

    return json.loads(path.read_text(encoding="utf-8"))


def body_of(source):
    """The source without its heading lines: those travel in the crumbs."""
    kept = []
    fence = ""
    for line in source.splitlines():
        fence = chunking._fence_state(fence, line)
        if fence or not chunking._HEADING.match(line):
            kept.append(line)

    return "".join(kept)


def squeezed(text):
    return "".join(text.split())


def is_subsequence(inner, outer):
    stream = iter(outer)

    return all(character in stream for character in inner)


def shared_edge(before, after):
    """The tail of one chunk that the next one repeats."""
    for size in range(min(len(before), len(after)), 0, -1):
        if before.endswith(after[:size]):
            return after[:size]

    return ""


@pytest.mark.parametrize(("case", "title"), CASES)
def test_a_document_splits_as_recorded(case, title, update_golden):
    produced = chunking.split(source_of(case), title=title)

    assert produced == recorded(case, produced, update_golden)


def test_every_recorded_case_is_exercised():
    """A fixture nobody splits proves nothing."""
    recorded_cases = sorted(path.name for path in TESTDATA.iterdir() if path.is_dir())

    assert recorded_cases == sorted(case for case, _ in CASES)


@pytest.mark.parametrize(("case", "title"), CASES)
def test_splitting_the_same_body_twice_gives_the_same_text(case, title):
    """Re-indexing a version must write the same text under the same chunk index."""
    source = source_of(case)

    assert chunking.split(source, title=title) == chunking.split(source, title=title)


@pytest.mark.parametrize(("case", "title"), CASES)
def test_no_chunk_is_larger_than_the_policy(case, title):
    for chunk in chunking.split(source_of(case), title=title):
        assert len(chunk) <= chunking.DEFAULT.max_chars


@pytest.mark.parametrize(("case", "title"), CASES)
def test_no_chunk_is_blank_or_padded(case, title):
    for chunk in chunking.split(source_of(case), title=title):
        assert chunk == chunk.strip()
        assert chunk


@pytest.mark.parametrize(("case", "title"), CASES)
def test_no_text_of_the_body_is_lost(case, title):
    source = source_of(case)
    produced = "".join(chunking.split(source, title=title))

    assert is_subsequence(squeezed(body_of(source)), squeezed(produced))


def test_the_strategy_identifier_is_pinned():
    """Matches the default of `kb.space.chunking_strategy`; the pipeline refuses anything else."""
    assert chunking.STRATEGY == "recursive_markdown"


def test_the_default_policy_is_pinned():
    """The size of a chunk is a retrieval decision, not a detail to drift."""
    assert (chunking.DEFAULT.max_chars, chunking.DEFAULT.overlap_chars) == (1000, 150)


@pytest.mark.parametrize(
    "policy",
    [
        {"max_chars": 0},
        {"max_chars": -10},
        {"overlap_chars": -1},
        {"overlap_chars": 1000},
        {"max_chars": 100, "overlap_chars": 100},
    ],
)
def test_an_unusable_policy_is_refused(policy):
    with pytest.raises(ValueError, match="chars"):
        chunking.Policy(**policy)


@pytest.mark.parametrize("text", ["", "   ", "\n\n\t\n"])
def test_a_body_without_text_gives_no_chunks(text):
    assert chunking.split(text, title="Стаття") == []


def test_the_article_subject_opens_a_body_without_headings():
    """A FAQ keeps its question in the subject; the answer alone embeds poorly."""
    assert chunking.split("Знижка становить 15%.", title="Чи є знижки?") == ["Чи є знижки?\n\nЗнижка становить 15%."]


def test_a_body_without_a_subject_carries_only_its_headings():
    assert chunking.split("## Тарифи\n\nтекст") == ["Тарифи\n\nтекст"]


def test_sibling_headings_do_not_nest():
    chunks = chunking.split("## Перший\n\nтекст один\n\n## Другий\n\nтекст два")

    assert [chunk.splitlines()[0] for chunk in chunks] == ["Перший", "Другий"]


def test_a_nested_heading_keeps_the_headings_above_it():
    chunks = chunking.split("# Доступ\n\n## Ролі\n\n### Оператор\n\nтекст", title="Довідка")

    assert chunks == ["Довідка / Доступ / Ролі / Оператор\n\nтекст"]


def test_a_heading_without_text_gives_no_chunk_but_stays_in_the_path():
    assert chunking.split("# Порожній\n\n## Далі\n\nтекст") == ["Порожній / Далі\n\nтекст"]


def test_a_hash_inside_a_code_fence_is_not_a_heading():
    chunks = chunking.split("## Реальний\n\n```md\n# не заголовок\n```\n\nхвіст")

    assert chunks == ["Реальний\n\n```md\n# не заголовок\n```\n\nхвіст"]


def test_a_tilde_fence_is_respected_too():
    chunks = chunking.split("~~~\n## не заголовок\n~~~")

    assert chunks == ["~~~\n## не заголовок\n~~~"]


def test_an_escaped_or_bare_hash_is_not_a_heading():
    """The Go side escapes author text that would otherwise read as markup."""
    assert chunking.split("\\## не заголовок\n\n#hashtag лишається") == ["\\## не заголовок\n\n#hashtag лишається"]


def test_a_blank_line_inside_a_fence_does_not_split_the_block():
    body = "```python\nfirst = 1\n\nsecond = 2\n```"

    assert chunking.split(body) == [body]


def test_prose_chunks_overlap_on_a_word_boundary():
    # Distinct sentences: repeated text would make any tail look like an overlap.
    chunks = chunking.split(" ".join(f"Речення номер {number} про тарифи." for number in range(120)))
    shared = shared_edge(chunks[0], chunks[1])

    assert len(chunks) > 1
    assert 0 < len(shared) <= chunking.DEFAULT.overlap_chars
    assert not shared[0].isspace()


def test_a_split_code_fence_is_reopened_and_never_repeats_a_line():
    """An overlap inside code would only duplicate lines and strand a fence mid-chunk."""
    body = "```bash\n" + "\n".join(f"echo рядок-{number}" for number in range(200)) + "\n```"
    chunks = chunking.split(body)
    lines = [line for chunk in chunks for line in chunk.splitlines() if line.startswith("echo")]

    assert len(chunks) > 1
    assert all(chunk.startswith("```bash") and chunk.endswith("```") for chunk in chunks)
    assert len(lines) == len(set(lines))


def test_a_word_longer_than_a_chunk_is_cut():
    chunks = chunking.split("х" * 2500)

    # 2500 characters in three chunks, the overlap of the second one repeated.
    assert [len(chunk) for chunk in chunks] == [1000, 1000, 650]


def test_a_deep_heading_path_still_leaves_room_for_the_text():
    """The path is context for the text, never the whole chunk."""
    headings = "\n\n".join(f"{'#' * level} {'Розділ довгої назви' * 3}" for level in range(1, 7))
    chunks = chunking.split(f"{headings}\n\n" + "слово " * 400, title="Дуже довга назва статті" * 4)
    text = chunks[0].split("\n\n", maxsplit=1)[1]

    assert len(text) >= chunking.DEFAULT.max_chars // chunking._PATH_SHARE


def test_a_subject_written_over_several_lines_stays_on_one():
    """The path opens the chunk; a line break in it would read as body text."""
    assert chunking.split("текст", title="Чи є\nзнижки?") == ["Чи є знижки?\n\nтекст"]


def test_an_overlap_never_reaches_back_into_code():
    """A tail that crossed a fence would open the next chunk with a lone closing fence."""
    # The fence nearly fills a chunk and the text after it is shorter than the
    # overlap, so a tail measured over the whole chunk would land inside the code.
    fence = "```bash\n" + "\n".join(f"echo рядок-{number} && sleep 1" for number in range(34)) + "\n```"
    sentence = "Довге речення про тарифи, оплату ліцензій і про умови користування сервісом підтримки. "
    chunks = chunking.split(f"{fence}\n\n" + sentence * 20)

    assert len(chunks) > 1
    for chunk in chunks:
        assert len([line for line in chunk.splitlines() if line.startswith("```")]) % 2 == 0


def test_a_fence_the_author_never_closed_is_closed_in_every_part():
    """An open fence would swallow whatever retrieval places after the chunk."""
    chunks = chunking.split("```bash\n" + "\n".join(f"echo рядок-{number}" for number in range(200)))

    assert len(chunks) > 1
    for chunk in chunks:
        assert len([line for line in chunk.splitlines() if line.startswith("```")]) % 2 == 0
