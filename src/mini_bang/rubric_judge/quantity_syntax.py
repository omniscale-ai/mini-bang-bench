"""Necessary local quantity syntax; never establishes factual incompatibility."""
import re
import unicodedata
from collections import Counter
from decimal import Decimal

# Aliased because the functions below use ``lexicon`` for their cardinal word table.
from mini_bang.rubric_judge import lexicon as vocabulary

EN_DASH = "\N{EN DASH}"
APPROX = "\N{ALMOST EQUAL TO}"
RIGHT_QUOTE = "\N{RIGHT SINGLE QUOTATION MARK}"
NUMERIC_LITERAL = r"[-+]?\d+(?:[.,]\d+)*(?:e[-+]?\d+)?"
NUMBER = NUMERIC_LITERAL + r"(?:\s*[" + EN_DASH + r"-]\s*\d+(?:[.,]\d+)*)?"
EXCLUSIVE_WORDS = r"only|solely|exclusively" + vocabulary.alternatives("exclusive_words")
POPULATION_EXCLUSIVE = (r"\b(?:" + EXCLUSIVE_WORDS + r")\b(?!\s+(?:after|before|when|then|during|at|in|by"
    + vocabulary.alternatives("exclusive_scope_followers") + r")\b)")
TIME = (r"(?:(?:time )?intervals?|durations?|steps?|timesteps?|seconds?|minutes?|hours?|frames?|moments?|ms"
    + vocabulary.alternatives("time_units") + ")")
NOUNS = {unit: english + vocabulary.alternatives("unit_nouns", unit) for unit, english in {
    "coherent_organisms": r"organisms?|body|bodies|cells?|entities|entity|symbionts?",
    "organism_parts": r"lobes?|parts?|layers?|gaps?",
    "groups": r"groups?|clusters?|communit(?:y|ies)|collections?|flocks?",
    "population_types": r"types?|kinds?",
    "visual_structures_unspecified": r"structures?|blobs?|shapes?",
    "activity_events": r"events?|flashes?|bursts?|collisions?|mergers?|splits?",
    "spatial_patterns": r"patterns?|stripes?|bands?|rings?",
    "numerical_channel_values": r"values?|channels?|mass|density|amplitude|intensity",
    "time": TIME,
    "other_explicit_unit": r"directions?|sides?|places?|corners?|points?|ends?|widgets?|units?|elements?",
}.items()}
NOMINAL_STOP = set(("a an the of to and or but if then than through from between until before after "
    "at in on by with without as for is are was were be been being have has had do does did "
    "remain remains survive survives exist exists longer more less further another "
    "change unchanged changes times").split()) | set(vocabulary.items("nominal_stop_words"))
INCIDENTAL_NOMINAL = {"time", "occasion", "occasions", "aspect", "aspects", "hand", "hands",
    *vocabulary.items("incidental_nominal_words")}
INCIDENTAL_INFLECTED = vocabulary.text("incidental_nominal_pattern")
CARDINAL_COMPONENTS = {value: vocabulary.words("cardinal_words", str(value), base=english) for value, english in {
    11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen", 15: "fifteen", 16: "sixteen",
    17: "seventeen", 18: "eighteen", 19: "nineteen", 20: "twenty", 30: "thirty", 40: "forty",
    50: "fifty", 60: "sixty", 70: "seventy", 80: "eighty", 90: "ninety",
    100: "hundred hundreds", 200: "", 300: "", 400: "", 500: "", 600: "", 700: "", 800: "", 900: "",
    1000: "thousand thousands", 10**6: "million millions", 10**9: "billion billions",
    10**12: "trillion trillions"}.items()}
CONJUNCTIONS = {"and", *vocabulary.items("conjunctions")}
PLURAL_SCALE_WORDS = tuple(vocabulary.items("plural_scale_words"))
PARTITIVE = r"(?:of\s+(?:the\s+)?" + vocabulary.alternatives("partitive_prepositions") + ")"
ADVERBIAL_QUANTITIES = set(vocabulary.items("adverbial_quantity", "words"))


def cardinal_lexicon(number_words):
    return {vocabulary.fold(word): value for value, words in {**number_words, **CARDINAL_COMPONENTS}.items()
        for word in words.split()}


def cardinal_pattern(number_words):
    atoms = set(cardinal_lexicon(number_words)) - {'a_single'}
    atom = r"(?:" + '|'.join(re.escape(w) for w in sorted(atoms, key=len, reverse=True)) + r")\b"
    return atom + r"(?:[\s-]+(?:(?:and" + vocabulary.alternatives("conjunctions") + r")\s+)?" + atom + r")*"


def cardinal_value(words, lexicon):
    """Compose conventional cardinals; ambiguous/noncanonical scope stays lexical."""
    total = group = 0
    last_scale = float('inf')
    previous = 0
    for word in words:
        if word in CONJUNCTIONS:
            if previous < 20:
                return None
            continue
        value = lexicon[word]
        if value >= 100 and not group and (word.endswith('s') or word.endswith(PLURAL_SCALE_WORDS)):
            return None
        if value >= 1000:
            if value >= last_scale:
                return None
            total += (group or 1) * value
            group, last_scale = 0, value
        elif word in {'hundred', 'hundreds'}:
            if group >= 100:
                return None
            group = (group or 1) * 100
        elif value >= 100:
            if group:
                return None
            group = value
        elif value >= 20:
            if group % 100:
                return None
            group += value
        elif group % 100 == 0 or (value < 10 and group % 10 == 0 and group % 100 >= 20):
            group += value
        else:
            return None
        previous = value
    return total + group


FRACTION_DENOMINATORS = vocabulary.mapping("word_fractions", "denominators")


def numeric_expression_pattern(number_words, captures=False):
    """One expression grammar for numeric evidence and bound scalar witnesses."""
    words = r'(?:a_single|' + cardinal_pattern(number_words) + ')'
    # Word fractions ("<integer> <integer-part word> [and] <fraction> <denominator>") need the lexicon.
    fraction = (r'|(?:' + '|'.join(vocabulary.items("word_fractions", "integer_part_words")) + r')\s+(?:'
        + vocabulary.text("word_fractions", "conjunction") + r'\s+)?(?P<fraction>' + words
        + r')\s+(?P<denominator>' + '|'.join(FRACTION_DENOMINATORS) + r')') if FRACTION_DENOMINATORS else ''
    decimal_words = (r'(?P<integer>' + words + r')\s+(?:(?:point' + vocabulary.alternatives("decimal_point_words")
        + r')\s+(?P<decimal>' + words + r')' + fraction + r')')
    pattern = r'(?<!\w)(?:' + decimal_words + r'|' + words + r')(?!\w)|(?<![\d.])' + NUMERIC_LITERAL
    return pattern if captures else re.sub(r'\(\?P<\w+>', '(?:', pattern)


def numeric_projection(text, number_words):
    """Value-preserving numeric evidence shared with the witness vocabulary.

    Exact integers/decimals normalize across languages/spellings. Noncanonical
    or vague numeric scope retains components instead of inventing a value.
    """
    text = re.sub(r'\ba single\b', 'a_single', normalized(text))
    lexicon, result = cardinal_lexicon(number_words), Counter()
    digits = r'(?<![\d.])' + NUMERIC_LITERAL
    pattern = numeric_expression_pattern(number_words, captures=True)
    def split_words(value):
        return re.findall(r'[\w]+', value)
    def emit(value):
        value = Decimal(value)
        sign, coefficient, exponent = value.as_tuple()
        coefficient = list(coefficient)
        while len(coefficient) > 1 and coefficient[-1] == 0:
            coefficient.pop()
            exponent += 1
        if not any(coefficient):
            sign, exponent = 0, 0
        result[str(Decimal((sign, tuple(coefficient), exponent)))] += 1
    for match in re.finditer(pattern, text):
        raw = match.group()
        if match['integer']:
            integer = cardinal_value(split_words(match['integer']), lexicon)
            fraction_words = split_words(match['decimal'] or match.groupdict().get('fraction'))
            fraction = cardinal_value(fraction_words, lexicon)
            if match['decimal'] and all(w in lexicon and lexicon[w] < 10 for w in fraction_words):
                literal = ''.join(str(lexicon[w]) for w in fraction_words)
                if integer is not None:
                    emit(str(integer) + '.' + literal)
                    continue
            else:
                named = match.groupdict().get('denominator')
                denominator = FRACTION_DENOMINATORS[named] if named else 10**len(str(fraction))
                if integer is not None and fraction is not None and fraction < denominator:
                    emit(str(integer) + '.' + str(fraction).zfill(len(str(denominator)) - 1))
                    continue
        if re.fullmatch(digits, raw):
            value = raw.replace(',', '') if re.fullmatch(r'[-+]?[1-9]\d{0,2}(?:,\d{3})+', raw) else raw.replace(',', '.')
            try:
                emit(value)
            except ArithmeticError:
                result['numeric-literal:' + raw] += 1
            continue
        pending = []
        for word in split_words(raw):
            if word not in lexicon and word not in CONJUNCTIONS:
                result['numeric-scope:' + word] += 1
                continue
            trial = pending + [word]
            if word in CONJUNCTIONS and cardinal_value(trial, lexicon) is None:
                if pending:
                    value = cardinal_value(pending, lexicon)
                    emit(value) if value is not None else result.update('numeric-scope:' + w for w in pending)
                pending = []
                continue
            if cardinal_value(trial, lexicon) is None and word not in CONJUNCTIONS:
                if pending:
                    value = cardinal_value(pending, lexicon)
                    emit(value) if value is not None else result.update('numeric-scope:' + w for w in pending)
                pending = [word]
            else:
                pending = trial
        if pending:
            value = cardinal_value(pending, lexicon)
            emit(value) if value is not None else result.update('numeric-scope:' + w for w in pending)
    return result


def normalized(text):
    return vocabulary.fold(unicodedata.normalize("NFKC", text).casefold()).replace(RIGHT_QUOTE, "'")


def noun_units(word):
    return {unit for unit, pattern in NOUNS.items() if re.fullmatch(pattern, word)}


def nominal_unit(tail, unit):
    """Bind quantity to its own noun phrase, stopping before another clause.

    Known heads must agree with the proposed unit. Unknown nominal vocabulary
    remains possible; the semantic judge still owns its meaning and identity.
    """
    tail = re.sub(r"^\s+" + PARTITIVE, " ", tail)
    phrase = re.match(r"\s+([^\W\d_][\w-]*(?:\s+[^\W\d_][\w-]*){0,6})", tail)
    if not phrase:
        return False
    words = phrase[1].split()
    if words[0] in NOMINAL_STOP:
        return False
    for word in words:
        if (word in {"change", "changes", "unchanged"} or word in INCIDENTAL_NOMINAL
                or INCIDENTAL_INFLECTED and re.fullmatch(INCIDENTAL_INFLECTED, word)):
            return False
        if word in NOMINAL_STOP:
            break
        categories = noun_units(word)
        if categories:
            # 'population types' and 'channel values' are headed by types/values.
            if word in {"population", "channel"}:
                continue
            return unit in categories
    return unit not in {"time", "numerical_channel_values"}


def quantity_pattern(number_words):
    # Cardinal atoms compose across tens, hundreds and arbitrarily many scales;
    # complete quantities are not enumerated or capped at a particular value.
    cardinal = cardinal_pattern(number_words)
    forms = [NUMBER, cardinal, r"a\s+single", r"a\s+pair", r"both", *vocabulary.items("quantity_words", "cardinal_forms"),
        r"no", r"none", r"several", r"multiple", r"few", r"many", *vocabulary.items("quantity_words", "vague_or_negative")]
    return r"(?<!\w)(?:" + "|".join(forms) + r")(?!\w)"


def temporal_spans(text):
    approximate = (r"(?:(?:roughly|approximately|about" + vocabulary.alternatives("approximation_words")
        + r")\s+|[~" + APPROX + r"]\s*)?")
    connector = r"(?:through|to|and|until" + vocabulary.alternatives("range_connectors") + r"|[" + EN_DASH + r"-])"
    patterns = [
        r"\b(?:t\s*[:=" + APPROX + r"~]|(?:steps?|timesteps?|frames?" + vocabulary.alternatives("step_nouns")
            + r")\s*[:=]?)\s*" + approximate + NUMBER + r"(?:\s*" + connector + r"\s*" + NUMBER + r")?",
        r"\btime\s*[:=]\s*" + approximate + NUMBER,
        r"\b(?:from|between|after|before|during|since|until|at|in" + vocabulary.alternatives("range_prepositions") + r")\s+"
            + approximate + NUMBER + r"\s*" + connector + r"\s*" + NUMBER,
        r"\b(?:at|before|after|during|since|until|on|in" + vocabulary.alternatives("time_prepositions") + r")\s+"
            + approximate + NUMBER,
        r"\b\d{1,2}:\d{2}(?::\d{2})?\b", NUMBER + r"\s*" + TIME + r"\b",
    ]
    return [m.span() for pattern in patterns for m in re.finditer(pattern, text)]


def explicit_quantity_witness(text, unit, number_words):
    text = normalized(text)
    quantity = quantity_pattern(number_words)
    if unit == "time":
        return bool(re.search(quantity + r"\s+(?:[\w-]+\s+){0,2}" + TIME + r"\b", text)
            or re.search(r"\b(?:at\s+" + vocabulary.alternatives("step_prepositions") + r")?(?:step|timestep|frame"
                         + vocabulary.alternatives("step_nouns") + r")\s*[:=]?\s*" + NUMBER + r"(?!\w)", text)
            or re.search(r"\bt\s*[:=" + APPROX + r"~]\s*" + NUMBER, text))
    if unit == "numerical_channel_values":
        value = r'(?:' + numeric_expression_pattern(number_words) + r')'
        extremum = r"peaks?|maximum|minimum" + vocabulary.alternatives("extremum_words")
        channel = NOUNS[unit] + r"|" + extremum
        # Bind the numeric value to its measurement predicate, rather than any
        # number elsewhere in a clause that happens to mention a channel.
        for clause in re.split(r"[;\n!?]|(?<!\d)\.(?!\d)", text):
            relation = r"(?:reach\w*|peak\w*|equal\w*|is|was" + vocabulary.alternatives("measure_relations") + r"|[:=])"
            connector = r"\s*(?:(?:at|of" + vocabulary.alternatives("measure_connectors") + r")\s+){0,2}"
            scalar = r"\b(?:" + channel + r")\b(?:\s+[\w-]+){0,3}?\s*" + relation + connector + value
            peak = r"\b(?:" + extremum + r")" + connector + value
            direct = r"\b(?:" + channel + r")\b\s*(?:[:=]\s*)?(?P<direct_value>" + value + r")"
            for match in re.finditer("(?:" + scalar + "|" + peak + "|" + direct + ")", clause):
                tail = clause[match.end():]
                if re.match(r'\s+(?:of' + vocabulary.alternatives("partitive_words") + r')\b', tail):
                    continue
                direct_value = match.groupdict().get('direct_value')
                if direct_value is not None and not re.fullmatch(NUMERIC_LITERAL, direct_value):
                    if not re.search(r'\b(?:have|has|had' + vocabulary.alternatives("possession_verbs") + r')\s*$',
                                     clause[:match.start()]):
                        continue
                wrong_unit = False
                for word in re.findall(r"[\w-]+", tail)[:7]:
                    if word in NOMINAL_STOP:
                        break
                    categories = noun_units(word)
                    if categories:
                        wrong_unit = unit not in categories
                        break
                if not wrong_unit:
                    return True
    if unit == "activity_events":
        event = (r"(?:flash\w*|occur\w*|happen\w*|repeat\w*|pulse\w*|collid\w*|merg\w*|split\w*"
            + vocabulary.alternatives("event_verbs") + ")")
        frequency = (r"(?:once|twice" + vocabulary.alternatives("frequency_words") + r"|(?:one|two) times?"
            + vocabulary.alternatives("frequency_phrases") + ")")
        for clause in re.split(r"[.;\n!?]", text):
            clause = re.sub(r"\bat once\b|\bonce (?:again|more)\b", 'time-adverb', clause)
            ratio = (r"\b" + frequency + r"\s+(?:(?:as|the|its|that|initial)\b"
                + vocabulary.alternatives("ratio_comparatives") + ")")
            if not re.search(ratio, clause) and re.search(r"\b" + event + r"\b(?:\s+[\w-]+){0,2}\s+" + frequency + r"\b", clause):
                return True
    temporal = temporal_spans(text) + [m.span() for m in re.finditer(r"\bno\s+one\b", text)]
    for match in re.finditer(quantity, text):
        if any(start < match.end() and match.start() < end for start, end in temporal):
            continue
        tail = text[match.end():]
        # An adverbial quantity followed by a verb quantifies the action, not a noun.
        if match.group() in ADVERBIAL_QUANTITIES:
            verb = re.match(vocabulary.text("adverbial_quantity", "verb_pattern"), tail)
            if verb and not noun_units(verb[1]):
                continue
        if nominal_unit(tail, unit):
            return True
    return False


def same_quantity_kind(candidate, reference, unit, number_words):
    """A count of intervals is not a duration or an ordinal simulation step."""
    if unit != "time":
        return True
    interval = r"(?:time\s+intervals?" + vocabulary.alternatives("interval_phrases") + r")\b"
    pattern = quantity_pattern(number_words) + r"\s+" + interval
    def interval_count(text):
        text = normalized(text)
        return any(not any(a < match.start() < b for a, b in temporal_spans(text))
            for match in re.finditer(pattern, text))
    return interval_count(candidate) == interval_count(reference)


INFLECTION = vocabulary.mapping("noun_inflection")
INFLECTED_LETTER = INFLECTION.get("letter", "")
TRUNCATED_SINGULAR_ENDINGS = tuple(INFLECTION.get("truncated_singular_endings", ()))
FLEETING_VOWEL_SINGULAR_ENDINGS = tuple(INFLECTION.get("fleeting_vowel_singular_endings", ()))
PLURAL_SINGULAR_ENDINGS = tuple(tuple(pair) for pair in INFLECTION.get("plural_singular_endings", ()))


def noun_pair(plural, singular):
    """Conservative local inflection comparison, not an ontology.

    English plurals are built in; other inflections come from the lexicon.
    """
    if plural == singular:
        return False
    inflected = bool(INFLECTED_LETTER) and re.search(INFLECTED_LETTER, plural)
    if inflected and singular.endswith(TRUNCATED_SINGULAR_ENDINGS) and plural == singular[:-1]:
        return True
    if inflected and singular.endswith(FLEETING_VOWEL_SINGULAR_ENDINGS):
        # Productive fleeting-vowel alternation in genitive plurals: the final
        # consonant cluster may insert a fleeting vowel.
        cluster = INFLECTION["consonant"]
        reduced = re.sub(r"(?<=" + cluster + r")" + INFLECTION["fleeting_vowel"] + r"(?=" + cluster + r"$)", '', plural)
        if reduced == singular[:-1]:
            return True
    if plural in {singular + "s", singular + "es"} or (singular.endswith("y") and plural == singular[:-1] + "ies"):
        return True
    if any(singular.endswith(end) and plural == singular[:-len(end)] + replacement
            for end, replacement in (("us", "i"), ("ex", "ices"), ("ix", "ices"), ("is", "es"), ("um", "a"), ("on", "a"))):
        return True
    return any(plural.endswith(p) and singular.endswith(s) and plural[:-len(p)] == (singular[:-len(s)] if s else singular)
        for p, s in PLURAL_SINGULAR_ENDINGS)


def existence_readings(text, forms, number_words):
    """Decompose affirmative existence, retaining its literal context/complement.

    NP-before-verb and existential inversion share one representation. Context
    is compared, never discarded to manufacture identity or phase agreement.
    """
    text, context = normalized(text).strip(), ""
    context_start = r"(?:at|by|in|on|near|after|before|during" + vocabulary.alternatives("context_prepositions") + r")\b"
    if ',' in text:
        before, after = text.split(',', 1)
        if re.match(context_start, before) and after.strip() and any(
                re.search(r'\b(?:' + form + r')\b', after) for form in forms.values()):
            context, text = before.strip(), after.strip()
    text = re.sub(r"^(?:at least|at most|no fewer than|no more than" + vocabulary.alternatives("leading_bounds")
                  + r")\s+", "", text)
    if re.search(r"\b(?:not|no|never" + vocabulary.alternatives("negation_particles") + r")\b", text):
        return []
    readings = []
    for lemma, pattern in forms.items():
        match = re.fullmatch(r"(?P<before>[^,.!?;:\n]*?)\b(?:" + pattern + r")\b(?P<after>[^.!?;:\n]*)[.!?\s]*", text)
        if not match:
            continue
        before, after = match['before'].strip(), match['after'].strip()
        options = [(before, after, context)] if before else []
        if after and (not before or before == 'there' or re.match(context_start, before)):
            inverted_context = ' '.join(v for v in (context, before if before != 'there' else '') if v)
            delimiter = re.search(r",\s*|\s+(?=(?:at|in|on|near|after|before|during"
                                  + vocabulary.alternatives("delimiter_prepositions") + r")\b)", after)
            options.append((after[:delimiter.start()] if delimiter else after,
                after[delimiter.start():].strip() if delimiter else '', inverted_context))
        for subject, tail, reading_context in options:
            if not subject:
                continue
            residual_subject = re.sub(quantity_pattern(number_words), 'quantity', subject)
            if re.search(r"\b(?:and|or|but" + vocabulary.alternatives("coordinators") + r")\b", residual_subject + ' ' + tail):
                continue
            readings.append((lemma, subject.split(), (' '.join(reading_context.split()), ' '.join(tail.split()))))
    return readings


def exclusive_existence_subject(text, forms, number_words):
    return any(re.search(POPULATION_EXCLUSIVE, ' '.join(parsed[1]))
        for parsed in existence_readings(text, forms, number_words))


def exclusive_subject_quantity(text, forms, number_words):
    subjects = [' '.join(parsed[1]) for parsed in existence_readings(text, forms, number_words)]
    subject = next((value for value in subjects if re.search(POPULATION_EXCLUSIVE, value)), '')
    operator = re.search(POPULATION_EXCLUSIVE, subject)
    subject = subject[operator.end():] if operator else subject
    return bool(re.search(quantity_pattern(number_words), subject) or re.search(
        r"\b(?:all|most|some|almost|nearly|exactly|precisely" + vocabulary.alternatives("subject_quantifiers") + r")\b",
        subject))


def adjective_pair(a, b):
    """Two forms of one adjective that differ only in the lexicon's inflectional ending."""
    ending = vocabulary.text("adjective_ending")
    if not ending:
        return False
    return bool(re.search(ending, a) and re.search(ending, b)
        and re.sub(ending, '', a) == re.sub(ending, '', b))


def same_head_existence(candidate, reference, forms, number_words):
    """Same plural head versus one of that head, with unchanged predicate/tail.

    Unit tags and the model's count boolean cannot bypass this syntax guard.
    Different complements, negation and opposed properties remain semantic work.
    """
    return any(_same_head_reading(left, right, number_words)
        for left in existence_readings(candidate, forms, number_words)
        for right in existence_readings(reference, forms, number_words))


def _same_head_reading(left, right, number_words):
    left_tail, right_tail = left[2][1].split(), right[2][1].split()
    if left[0] != right[0] or left[2][0] != right[2][0] or len(left_tail) != len(right_tail) or not all(
            a == b or adjective_pair(a, b) for a, b in zip(left_tail, right_tail)):
        return False
    # The SAME canonical quantity grammar drives nominal witness detection and
    # subject normalization; adding a number word must not leave a second list.
    quantity = quantity_pattern(number_words)
    scope = (r"(?:the|these|those|a(?!\s+single\b)|an|only|solely|exclusively|exactly|precisely|just|all|most|some|almost|nearly"
        + vocabulary.alternatives("scope_words") + r")\b\s*")
    def scopes(text):
        return re.sub(r"^(?:" + scope + r")+", '', text).strip()
    one_words = [re.escape(word).replace("a_single", r"a\s+single") for word in number_words[1].split()]
    one = r"(?:" + "|".join(one_words) + r"|1(?:[.,]0+)?)\b\s+"
    ref_subject = scopes(" ".join(right[1]))
    matches = list(re.finditer(one, ref_subject))
    if not matches:
        return False
    match = matches[-1]
    leading_context = ref_subject[:match.start()].strip()
    leading_context = re.sub(r"\b(?:only|exactly|precisely|just" + vocabulary.alternatives("leading_scope_words")
                             + r")\s*$", '', leading_context).strip()
    ref_np = scopes(ref_subject[match.end():])
    ref_partitive = bool(re.match(PARTITIVE, ref_np))
    ref_np = re.sub(r"^" + PARTITIVE, '', ref_np)
    ref_words = ref_np.split()
    subject = " ".join(left[1])
    if leading_context:
        if not subject.startswith(leading_context + ' '):
            return False
        subject = subject[len(leading_context):].strip()
    prefix = r"(?:" + scope + r"|(?:" + quantity + r")\s*|" + PARTITIVE + ")"
    subject = re.sub(r"^(?:" + prefix + r")+", '', subject).strip()
    words = subject.split()
    if not ref_words or len(words) != len(ref_words):
        return False
    # Only recognized quantity/scope syntax is removed; substantive modifiers
    # and earlier propositions are never discarded to manufacture agreement.
    changed = [(a, b) for a, b in zip(words, ref_words) if a != b]
    return all(noun_pair(a, b) or adjective_pair(a, b) for a, b in changed) and (
        any(noun_pair(a, b) for a, b in changed) or ref_partitive)
