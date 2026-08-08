"""GraphRAG query intent and entity parsing helpers."""

from dataclasses import dataclass
import re

from rag.graph.models import GraphEntity
from rag.graph.schema import GraphEntityType, GraphRelationType
from rag.graph.triple_extractor import (
    PluginLookup,
    build_candidate_variants,
    resolve_plugin_id,
)


QUERY_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9+._-]*")
QUERY_WHITESPACE_PATTERN = re.compile(r"\s+")
QUERY_PUNCTUATION_PATTERN = re.compile(r"[?!,:;]+")
MAX_QUERY_ENTITY_TOKENS = 8

DEPENDENCY_QUERY_PATTERNS = (
    re.compile(r"\bwhat does .+ depend on\b", re.IGNORECASE),
    re.compile(r"\bdoes .+ depend on\b", re.IGNORECASE),
    re.compile(r"\bdepends on\b", re.IGNORECASE),
    re.compile(r"\bdependencies of\b", re.IGNORECASE),
    re.compile(r"\brequires?\b", re.IGNORECASE),
)
REVERSE_DEPENDENCY_QUERY_PATTERNS = (
    re.compile(r"\bwhat depends on\b", re.IGNORECASE),
    re.compile(r"\bwhich plugins depend on\b", re.IGNORECASE),
    re.compile(r"\bdepended on by\b", re.IGNORECASE),
    re.compile(r"\brequired by\b", re.IGNORECASE),
    re.compile(r"\bdepending on\b", re.IGNORECASE),
)
CONFLICT_QUERY_PATTERNS = (
    re.compile(r"\bconflicts? with\b", re.IGNORECASE),
    re.compile(r"\bincompatible with\b", re.IGNORECASE),
    re.compile(r"\bconflicts?\b", re.IGNORECASE),
    re.compile(r"\bincompatible\b", re.IGNORECASE),
)
MULTI_HOP_QUERY_PATTERNS = (
    re.compile(r"\bindirect(?:ly)?\b", re.IGNORECASE),
    re.compile(r"\btransitive(?:ly)?\b", re.IGNORECASE),
    re.compile(r"\bthrough\b", re.IGNORECASE),
    re.compile(r"\bchain\b", re.IGNORECASE),
)
BOOLEAN_QUERY_PATTERN = re.compile(
    r"^(?:does|is|are|can|could|will|would)\b",
    re.IGNORECASE,
)
COUNT_QUERY_PATTERN = re.compile(
    r"\b(?:how many|count|number of)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class GraphQueryIntent:
    """
    Parsed graph intent from a user query.

    Args:
        relation_types (tuple[str, ...]): Relation types requested by the query.
        direction (str): Traversal direction needed for the relation query.
        traversal_depth (int): Traversal depth requested by the query.
    """

    relation_types: tuple[str, ...]
    direction: str
    traversal_depth: int = 1


@dataclass(frozen=True)
class GraphQueryMatch:
    """
    Parsed graph query state used by graph traversal.

    Args:
        query (str): Original user query.
        query_entity (str): Raw entity text found in the query.
        matched_entity (GraphEntity): Canonical plugin entity matched from the query.
        intent (GraphQueryIntent): Parsed relation intent.
        plan (GraphQueryPlan): Structured query plan used by later graph stages.
        entities (tuple[ResolvedQueryEntity, ...]): All resolved plugin mentions.
    """

    query: str
    query_entity: str
    matched_entity: GraphEntity
    intent: GraphQueryIntent
    plan: "GraphQueryPlan"
    entities: tuple["ResolvedQueryEntity", ...]


@dataclass(frozen=True)
class ResolvedQueryEntity:
    """
    A canonical plugin entity together with its span in the raw query.

    Attributes:
        text (str): Exact entity text copied from the original query.
        entity (GraphEntity): Canonical graph entity resolved from the text.
        start (int): Inclusive character offset of the entity in the query.
        end (int): Exclusive character offset of the entity in the query.
    """

    text: str
    entity: GraphEntity
    start: int
    end: int


@dataclass(frozen=True)
class GraphQueryPlan:
    """
    Structured graph retrieval request derived from a user query.

    Attributes:
        relation_types (tuple[str, ...]): Graph relations to retrieve.
        direction (str): Traversal direction for the relation lookup.
        source_entity (GraphEntity | None): Known source when unambiguous.
        target_entity (GraphEntity | None): Known target when unambiguous.
        traversal_depth (int): Number of graph hops requested.
        answer_mode (str): Expected answer shape: list, boolean, or count.
        matched_rule (str): Coarse rule category used for diagnostics.
    """

    relation_types: tuple[str, ...]
    direction: str
    source_entity: GraphEntity | None
    target_entity: GraphEntity | None
    traversal_depth: int
    answer_mode: str
    matched_rule: str


def normalize_graph_query(query: str) -> str:
    """
    Normalize user wording for graph-intent matching.

    The original query remains unchanged for plugin entity resolution.

    Args:
        query (str): Raw user query.

    Returns:
        str: Normalized query text used by intent rules.
    """
    normalized_query = query.translate(
        str.maketrans(
            {
                "\u2018": "'",
                "\u2019": "'",
                "\u2010": "-",
                "\u2011": "-",
                "\u2013": "-",
                "\u2014": "-",
            }
        )
    ).lower()

    normalized_query = re.sub(
        r"\bplug[\s-]?ins\b",
        "plugins",
        normalized_query,
    )
    normalized_query = re.sub(
        r"\bplug[\s-]?in\b",
        "plugin",
        normalized_query,
    )
    normalized_query = QUERY_PUNCTUATION_PATTERN.sub(" ", normalized_query)
    return QUERY_WHITESPACE_PATTERN.sub(" ", normalized_query).strip()


def build_query_entity(plugin_id: str) -> GraphEntity:
    """
    Build a plugin graph entity from a canonical plugin ID.

    Args:
        plugin_id (str): Canonical plugin ID.

    Returns:
        GraphEntity: Plugin entity used by query parsing.
    """
    return GraphEntity(
        name=plugin_id,
        entity_type=GraphEntityType.PLUGIN.value,
        entity_id=plugin_id,
    )


def detect_graph_relation_types(query: str) -> tuple[str, ...] | None:
    """
    Detect the relationship family requested by a graph query.

    Args:
        query (str): User query text.

    Returns:
        tuple[str, ...] | None: Requested graph relation types, if recognized.
    """
    normalized_query = normalize_graph_query(query)
    return _detect_relation_types(normalized_query)


def detect_graph_query_direction(query: str) -> str | None:
    """
    Detect the traversal direction requested by a graph query.

    Args:
        query (str): User query text.

    Returns:
        str | None: Requested traversal direction, if recognized.
    """
    normalized_query = normalize_graph_query(query)
    return _detect_query_direction(normalized_query)


def _detect_relation_types(normalized_query: str) -> tuple[str, ...] | None:
    """
    Detect the relationship family without deciding traversal direction.

    Args:
        normalized_query (str): Query text after mechanical normalization.

    Returns:
        tuple[str, ...] | None: Relation types requested by the query, or None
        when no supported relationship wording is present.
    """
    if any(pattern.search(normalized_query) for pattern in CONFLICT_QUERY_PATTERNS):
        return (GraphRelationType.CONFLICTS_WITH.value,)

    if any(
        pattern.search(normalized_query)
        for pattern in (*DEPENDENCY_QUERY_PATTERNS, *REVERSE_DEPENDENCY_QUERY_PATTERNS)
    ):
        return (
            GraphRelationType.DEPENDS_ON.value,
            GraphRelationType.OPTIONAL_DEPENDS_ON.value,
        )

    return None


def _detect_query_direction(normalized_query: str) -> str | None:
    """
    Detect traversal direction independently from relation type.

    Args:
        normalized_query (str): Query text after mechanical normalization.

    Returns:
        str | None: ``outgoing`` for dependencies of the named plugin,
        ``incoming`` for plugins related to the named target, ``both`` for
        conflicts, or None when direction is not recognized.
    """
    if any(pattern.search(normalized_query) for pattern in CONFLICT_QUERY_PATTERNS):
        return "both"

    if any(
        pattern.search(normalized_query)
        for pattern in REVERSE_DEPENDENCY_QUERY_PATTERNS
    ):
        return "incoming"

    if any(pattern.search(normalized_query) for pattern in DEPENDENCY_QUERY_PATTERNS):
        return "outgoing"

    return None


def detect_graph_query_intent(query: str) -> GraphQueryIntent | None:
    """
    Detect relation intent from a user query.

    Args:
        query (str): User query text.

    Returns:
        GraphQueryIntent | None: Parsed graph intent, if the query is relational.
    """
    normalized_query = normalize_graph_query(query)
    traversal_depth = (
        2
        if any(pattern.search(normalized_query) for pattern in MULTI_HOP_QUERY_PATTERNS)
        else 1
    )

    relation_types = _detect_relation_types(normalized_query)
    direction = _detect_query_direction(normalized_query)
    if relation_types is None or direction is None:
        return None

    return GraphQueryIntent(
        relation_types=relation_types,
        direction=direction,
        traversal_depth=traversal_depth,
    )


def build_graph_query_plan(
    query: str,
    intent: GraphQueryIntent,
    entities: tuple[ResolvedQueryEntity, ...],
) -> GraphQueryPlan:
    """
    Build a structured graph request from intent and resolved entities.

    Source and target fields are populated only when exactly one entity is
    present. Multi-entity role assignment remains the responsibility of the
    structural template parser.

    Args:
        query (str): Original user query.
        intent (GraphQueryIntent): Relation and traversal intent.
        entities (tuple[ResolvedQueryEntity, ...]): Resolved plugin mentions.

    Returns:
        GraphQueryPlan: Structured request for graph retrieval.
    """
    anchor_entity = entities[0].entity if len(entities) == 1 else None
    source_entity = (
        anchor_entity if anchor_entity and intent.direction == "outgoing" else None
    )
    target_entity = (
        anchor_entity if anchor_entity and intent.direction == "incoming" else None
    )
    if GraphRelationType.CONFLICTS_WITH.value in intent.relation_types:
        matched_rule = "conflict"
    elif intent.direction == "incoming":
        matched_rule = "incoming_dependency"
    else:
        matched_rule = "outgoing_dependency"

    normalized_query = normalize_graph_query(query)
    if COUNT_QUERY_PATTERN.search(normalized_query):
        answer_mode = "count"
    elif BOOLEAN_QUERY_PATTERN.search(normalized_query):
        answer_mode = "boolean"
    else:
        answer_mode = "list"

    return GraphQueryPlan(
        relation_types=intent.relation_types,
        direction=intent.direction,
        source_entity=source_entity,
        target_entity=target_entity,
        traversal_depth=intent.traversal_depth,
        answer_mode=answer_mode,
        matched_rule=matched_rule,
    )


def resolve_query_entity_text(
    text: str,
    plugin_lookup: PluginLookup,
) -> tuple[str, GraphEntity] | None:
    """
    Resolve a plugin entity from one text span.

    Args:
        text (str): Candidate query text span.
        plugin_lookup (PluginLookup): Canonical plugin lookup built from IDs.

    Returns:
        tuple[str, GraphEntity] | None: Matched text and canonical plugin entity.
    """
    entities = resolve_query_entities(text, plugin_lookup)
    if not entities:
        return None

    entity = entities[0]
    return entity.text, entity.entity


def resolve_query_entities(
    text: str,
    plugin_lookup: PluginLookup,
) -> tuple[ResolvedQueryEntity, ...]:
    """
    Resolve every known plugin mention and preserve its raw-query span.

    Args:
        text (str): Candidate query text.
        plugin_lookup (PluginLookup): Canonical plugin lookup built from IDs.

    Returns:
        tuple[ResolvedQueryEntity, ...]: Ordered, non-overlapping plugin mentions.
    """
    tokens = list(QUERY_TOKEN_PATTERN.finditer(text))
    resolved_entities: list[ResolvedQueryEntity] = []
    occupied_token_indexes: set[int] = set()

    for start_index, _token in enumerate(tokens):
        if start_index in occupied_token_indexes:
            continue

        for token_count in range(
            min(MAX_QUERY_ENTITY_TOKENS, len(tokens) - start_index),
            0,
            -1,
        ):
            token_indexes = range(start_index, start_index + token_count)
            if any(index in occupied_token_indexes for index in token_indexes):
                continue

            lookup_phrase = " ".join(
                tokens[index].group() for index in token_indexes
            )
            target_id = None
            for variant in build_candidate_variants(lookup_phrase):
                target_id = resolve_plugin_id(variant, plugin_lookup)
                if target_id:
                    break
            if target_id:
                end_token = tokens[start_index + token_count - 1]
                resolved_entities.append(
                    ResolvedQueryEntity(
                        text=text[_token.start() : end_token.end()],
                        entity=build_query_entity(target_id),
                        start=_token.start(),
                        end=end_token.end(),
                    )
                )
                occupied_token_indexes.update(token_indexes)
                break

    return tuple(resolved_entities)


def parse_graph_query(
    query: str,
    plugin_lookup: PluginLookup,
) -> GraphQueryMatch | None:
    """
    Parse a user query into graph intent and a canonical entity.

    Args:
        query (str): User query text.
        plugin_lookup (PluginLookup): Canonical plugin lookup built from IDs.

    Returns:
        GraphQueryMatch | None: Parsed graph query state when graph retrieval applies.
    """
    intent = detect_graph_query_intent(query)
    if not intent:
        return None

    entities = resolve_query_entities(query, plugin_lookup)
    if not entities:
        return None

    first_entity = entities[0]
    return GraphQueryMatch(
        query=query,
        query_entity=first_entity.text,
        matched_entity=first_entity.entity,
        intent=intent,
        plan=build_graph_query_plan(query, intent, entities),
        entities=entities,
    )
