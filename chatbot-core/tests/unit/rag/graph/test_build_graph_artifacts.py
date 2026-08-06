"""Unit tests for GraphRAG graph build entrypoint."""

import json
from unittest.mock import Mock

import pytest

from rag.graph.build_graph_artifacts import (
    get_update_center_dependencies,
    load_update_center_triples,
    load_plugin_chunks,
    load_plugin_ids,
    run_graph_build,
)
from rag.graph.graph_artifacts import GraphArtifactPaths
from rag.graph.graph_store import load_graph


def build_chunk(title: str, text: str) -> dict:
    """
    Build a plugin chunk for graph build tests.

    Args:
        title (str): Source plugin title.
        text (str): Chunk text.

    Returns:
        dict: Chunk payload matching chunks_plugin_docs.json.
    """
    return {
        "id": "chunk-1",
        "chunk_text": text,
        "metadata": {
            "title": title,
            "data_source": "jenkins_plugins_documentation",
        },
    }


def test_load_plugin_chunks_keeps_dict_records(tmp_path):
    """
    Verify plugin chunk loading skips malformed non-dict records.
    """
    chunks_path = tmp_path / "chunks_plugin_docs.json"
    chunks_path.write_text(
        json.dumps(
            [
                build_chunk("source-plugin", "valid text"),
                {"id": "chunk-2"},
                "bad-record",
                123,
            ]
        ),
        encoding="utf-8",
    )

    chunks = load_plugin_chunks(chunks_path)

    assert chunks == [build_chunk("source-plugin", "valid text")]


def test_load_plugin_ids_rejects_invalid_values(tmp_path):
    """
    Verify the canonical plugin index rejects malformed records.
    """
    plugin_names_path = tmp_path / "plugin_names.json"
    plugin_names_path.write_text(json.dumps(["source-plugin", 123]), encoding="utf-8")

    with pytest.raises(ValueError, match="invalid plugin ID"):
        load_plugin_ids(plugin_names_path)


def test_run_graph_build_writes_artifacts_from_fake_inputs(tmp_path):
    """
    Verify the build orchestration works with small fake input files.
    """
    mock_logger = Mock()
    plugin_names_path = tmp_path / "plugin_names.json"
    chunks_path = tmp_path / "chunks_plugin_docs.json"
    paths = GraphArtifactPaths(
        graph_path=tmp_path / "graph" / "plugin_graph.json",
        triples_path=tmp_path / "graph" / "triples.jsonl",
        report_path=tmp_path / "graph" / "extraction_report.json",
    )
    plugin_names_path.write_text(
        json.dumps(["source-plugin", "target-plugin"]),
        encoding="utf-8",
    )
    chunks_path.write_text(
        json.dumps(
            [
                build_chunk(
                    "source-plugin",
                    "This plugin depends on Target Plugin.",
                )
            ]
        ),
        encoding="utf-8",
    )

    report = run_graph_build(
        plugin_names_path=plugin_names_path,
        chunks_path=chunks_path,
        artifact_paths=paths,
        logger=mock_logger,
    )
    graph = load_graph(str(paths.graph_path), mock_logger)

    assert report["chunk_count"] == 1
    assert report["graph_source"] == "plugin_documentation"
    assert report["dependency_metadata"] == "not_used"
    assert report["triple_count"] == 1
    assert report["node_count"] == 2
    assert report["edge_count"] == 1
    assert paths.triples_path.exists()
    assert paths.report_path.exists()
    assert graph.number_of_edges("source-plugin", "target-plugin") == 1


def test_load_update_center_triples_preserves_optional_version(tmp_path):
    """Verify structured dependency metadata becomes a source-grounded triple."""
    snapshot_path = tmp_path / "update-center.actual.json"
    snapshot_path.write_text(
        json.dumps(
            {
                "plugins": {
                    "source-plugin": {
                        "dependencies": [
                            {
                                "name": "target-plugin",
                                "optional": True,
                                "version": "1.2.3",
                            }
                        ]
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    triples = load_update_center_triples(snapshot_path)

    assert len(triples) == 1
    assert triples[0].relation == "OPTIONAL_DEPENDS_ON"
    assert triples[0].evidence.source_data_source == "jenkins_update_center"
    assert "1.2.3" in triples[0].evidence.evidence


def test_get_update_center_dependencies_filters_by_plugin_id(tmp_path):
    """Verify direct lookup returns only one plugin's dependencies."""
    snapshot_path = tmp_path / "update-center.actual.json"
    snapshot_path.write_text(
        json.dumps(
            {
                "plugins": {
                    "source-plugin": {
                        "dependencies": [
                            {"name": "required-plugin"},
                            {"name": "optional-plugin", "optional": True},
                        ]
                    },
                    "other-plugin": {
                        "dependencies": [{"name": "unrelated-plugin"}]
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    dependencies = get_update_center_dependencies(snapshot_path, "source-plugin")

    assert [triple.target.entity_id for triple in dependencies] == [
        "required-plugin",
        "optional-plugin",
    ]
    assert get_update_center_dependencies(snapshot_path, "missing-plugin") == []

    with pytest.raises(ValueError, match="must not be empty"):
        get_update_center_dependencies(snapshot_path, " ")


def test_run_graph_build_merges_update_center_into_one_graph(tmp_path):
    """Verify documentation and metadata dependencies share one artifact."""
    mock_logger = Mock()
    plugin_names_path = tmp_path / "plugin_names.json"
    chunks_path = tmp_path / "chunks_plugin_docs.json"
    update_center_path = tmp_path / "update-center.actual.json"
    paths = GraphArtifactPaths(
        graph_path=tmp_path / "graph" / "plugin_graph.json",
        triples_path=tmp_path / "graph" / "triples.jsonl",
        report_path=tmp_path / "graph" / "extraction_report.json",
    )
    plugin_names_path.write_text(
        json.dumps(["source-plugin", "target-plugin"]),
        encoding="utf-8",
    )
    chunks_path.write_text(
        json.dumps(
            [
                build_chunk(
                    "source-plugin",
                    "This plugin depends on Target Plugin.",
                )
            ]
        ),
        encoding="utf-8",
    )
    update_center_path.write_text(
        json.dumps(
            {
                "plugins": {
                    "source-plugin": {
                        "dependencies": [
                            {"name": "target-plugin", "optional": False}
                        ]
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    report = run_graph_build(
        plugin_names_path=plugin_names_path,
        chunks_path=chunks_path,
        artifact_paths=paths,
        logger=mock_logger,
        update_center_path=update_center_path,
    )
    graph = load_graph(str(paths.graph_path), mock_logger)

    assert report["triple_count"] == 1
    assert graph.number_of_edges("source-plugin", "target-plugin") == 1
    edges = graph.get_edge_data("source-plugin", "target-plugin")
    edge = next(iter(edges.values()))
    assert edge["relation"] == "DEPENDS_ON"
    assert edge["source_data_source"] == "jenkins_plugins_documentation"


def test_run_graph_build_keeps_documentation_conflicts(tmp_path):
    """Verify dependency precedence does not remove other relation types."""
    mock_logger = Mock()
    plugin_names_path = tmp_path / "plugin_names.json"
    chunks_path = tmp_path / "chunks_plugin_docs.json"
    update_center_path = tmp_path / "update-center.actual.json"
    paths = GraphArtifactPaths(
        graph_path=tmp_path / "graph" / "plugin_graph.json",
        triples_path=tmp_path / "graph" / "triples.jsonl",
        report_path=tmp_path / "graph" / "extraction_report.json",
    )
    plugin_names_path.write_text(
        json.dumps(["source-plugin", "target-plugin"]),
        encoding="utf-8",
    )
    chunks_path.write_text(
        json.dumps(
            [
                build_chunk(
                    "source-plugin",
                    "This plugin depends on Target Plugin and conflicts with "
                    "Target Plugin.",
                )
            ]
        ),
        encoding="utf-8",
    )
    update_center_path.write_text(
        json.dumps(
            {
                "plugins": {
                    "source-plugin": {
                        "dependencies": [
                            {"name": "target-plugin", "optional": False}
                        ]
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    run_graph_build(
        plugin_names_path=plugin_names_path,
        chunks_path=chunks_path,
        artifact_paths=paths,
        logger=mock_logger,
        update_center_path=update_center_path,
    )
    graph = load_graph(str(paths.graph_path), mock_logger)

    assert graph.number_of_edges("source-plugin", "target-plugin") == 2
    assert {
        edge["relation"]
        for edge in graph.get_edge_data("source-plugin", "target-plugin").values()
    } == {"DEPENDS_ON", "CONFLICTS_WITH"}
