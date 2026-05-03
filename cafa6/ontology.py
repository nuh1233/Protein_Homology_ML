"""GO ontology parsing and ancestor closure utilities."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from cafa6.io import BRANCH_TO_ASPECT


NAMESPACE_TO_BRANCH: dict[str, str] = {
    "molecular_function": "MF",
    "biological_process": "BP",
    "cellular_component": "CC",
}

CLOSURE_RELATIONS: tuple[str, ...] = ("is_a", "part_of")


@dataclass(frozen=True)
class GoTerm:
    """Parsed GO term stanza."""

    term: str
    name: str
    namespace: str
    branch: str | None
    is_obsolete: bool
    alt_ids: tuple[str, ...]


@dataclass(frozen=True)
class GoEdge:
    """Raw child-to-parent GO edge."""

    child_term: str
    parent_term: str
    relation: str


@dataclass(frozen=True)
class GoTables:
    """Ontology tables and parser report."""

    terms: pd.DataFrame
    edges: pd.DataFrame
    ancestors: pd.DataFrame
    alt_ids: pd.DataFrame
    all_terms: pd.DataFrame
    report: dict[str, object]


def _empty_edge_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "child_term",
            "parent_term",
            "relation",
            "child_branch",
            "parent_branch",
        ]
    )


def _empty_ancestor_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "term",
            "ancestor",
            "distance",
            "term_branch",
            "ancestor_branch",
        ]
    )


def _parse_go_parent(line: str, relation: str) -> str:
    """Extract a GO parent ID from an OBO parent line."""

    if relation == "is_a":
        return line.removeprefix("is_a:").strip().split()[0]

    value = line.removeprefix("relationship:").strip()
    tokens = value.split()
    if len(tokens) < 2:
        raise ValueError(f"Malformed relationship line: {line}")
    return tokens[1]


def parse_go_obo(path: str | Path) -> tuple[list[GoTerm], list[GoEdge]]:
    """Parse GO terms and raw parent edges from an OBO file."""

    obo_path = Path(path)
    terms: list[GoTerm] = []
    edges: list[GoEdge] = []

    stanza_type: str | None = None
    current: dict[str, object] = {}

    def flush_current() -> None:
        if stanza_type != "Term" or not current:
            return

        term_id = current.get("id")
        if not isinstance(term_id, str) or not term_id:
            raise ValueError("Encountered a [Term] stanza without an id.")

        namespace = str(current.get("namespace") or "")
        alt_ids = tuple(sorted(set(current.get("alt_ids", []))))
        term = GoTerm(
            term=term_id,
            name=str(current.get("name") or ""),
            namespace=namespace,
            branch=NAMESPACE_TO_BRANCH.get(namespace),
            is_obsolete=bool(current.get("is_obsolete", False)),
            alt_ids=alt_ids,
        )
        terms.append(term)

        for edge in current.get("edges", []):
            parent_term, relation = edge
            edges.append(
                GoEdge(
                    child_term=term_id,
                    parent_term=str(parent_term),
                    relation=str(relation),
                )
            )

    with obo_path.open("rt", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith("[") and line.endswith("]"):
                flush_current()
                stanza_type = line[1:-1]
                current = {}
                continue

            if stanza_type != "Term":
                continue

            if line.startswith("id: "):
                current["id"] = line.removeprefix("id:").strip()
            elif line.startswith("name: "):
                current["name"] = line.removeprefix("name:").strip()
            elif line.startswith("namespace: "):
                current["namespace"] = line.removeprefix("namespace:").strip()
            elif line.startswith("alt_id: "):
                current.setdefault("alt_ids", []).append(line.removeprefix("alt_id:").strip())
            elif line.startswith("is_obsolete: "):
                current["is_obsolete"] = line.removeprefix("is_obsolete:").strip().lower() == "true"
            elif line.startswith("is_a: "):
                current.setdefault("edges", []).append((_parse_go_parent(line, "is_a"), "is_a"))
            elif line.startswith("relationship: part_of "):
                current.setdefault("edges", []).append((_parse_go_parent(line, "part_of"), "part_of"))

    flush_current()
    return terms, edges


def _terms_to_frame(terms: Iterable[GoTerm]) -> pd.DataFrame:
    rows = [
        {
            "term": term.term,
            "name": term.name,
            "namespace": term.namespace,
            "branch": term.branch,
            "is_obsolete": term.is_obsolete,
            "alt_ids": "|".join(term.alt_ids),
            "n_alt_ids": len(term.alt_ids),
        }
        for term in terms
    ]
    columns = ["term", "name", "namespace", "branch", "is_obsolete", "alt_ids", "n_alt_ids"]
    return pd.DataFrame.from_records(rows, columns=columns).sort_values("term", kind="mergesort").reset_index(drop=True)


def _make_alt_id_frame(active_terms: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for row in active_terms.itertuples(index=False):
        alt_ids = str(row.alt_ids).split("|") if row.alt_ids else []
        rows.extend({"alt_id": alt_id, "term": row.term} for alt_id in alt_ids if alt_id)

    alt_id_frame = pd.DataFrame.from_records(rows, columns=["alt_id", "term"])
    if alt_id_frame.empty:
        return alt_id_frame

    duplicated = alt_id_frame.loc[alt_id_frame.duplicated("alt_id", keep=False), "alt_id"].drop_duplicates().tolist()
    if duplicated:
        preview = ", ".join(duplicated[:10])
        raise ValueError(f"GO alt_id values map to multiple active terms: {preview}")

    return alt_id_frame.sort_values(["alt_id", "term"], kind="mergesort").reset_index(drop=True)


def _filter_edges(
    raw_edges: Iterable[GoEdge],
    active_terms: pd.DataFrame,
    relations: Iterable[str] = CLOSURE_RELATIONS,
) -> tuple[pd.DataFrame, dict[str, int]]:
    relation_set = set(relations)
    active_term_ids = set(active_terms["term"])
    branch_by_term = dict(zip(active_terms["term"], active_terms["branch"], strict=True))

    rows: list[dict[str, str]] = []
    stats = {
        "raw_edge_count": 0,
        "dropped_relation_count": 0,
        "dropped_unknown_child_count": 0,
        "dropped_unknown_parent_count": 0,
        "dropped_unknown_branch_count": 0,
        "dropped_cross_branch_count": 0,
    }

    for edge in raw_edges:
        stats["raw_edge_count"] += 1

        if edge.relation not in relation_set:
            stats["dropped_relation_count"] += 1
            continue
        if edge.child_term not in active_term_ids:
            stats["dropped_unknown_child_count"] += 1
            continue
        if edge.parent_term not in active_term_ids:
            stats["dropped_unknown_parent_count"] += 1
            continue

        child_branch = branch_by_term[edge.child_term]
        parent_branch = branch_by_term[edge.parent_term]
        if not child_branch or not parent_branch:
            stats["dropped_unknown_branch_count"] += 1
            continue
        if child_branch != parent_branch:
            stats["dropped_cross_branch_count"] += 1
            continue

        rows.append(
            {
                "child_term": edge.child_term,
                "parent_term": edge.parent_term,
                "relation": edge.relation,
                "child_branch": child_branch,
                "parent_branch": parent_branch,
            }
        )

    if not rows:
        return _empty_edge_frame(), stats

    edges = pd.DataFrame.from_records(rows)
    edges = edges.drop_duplicates(["child_term", "parent_term", "relation"])
    edges = edges.sort_values(["child_term", "parent_term", "relation"], kind="mergesort").reset_index(drop=True)
    return edges, stats


def build_ancestor_table(terms: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    """Build a transitive ancestor table from child-to-parent edges."""

    if edges.empty:
        return _empty_ancestor_frame()

    parents_by_term = (
        edges.groupby("child_term")["parent_term"]
        .apply(lambda values: tuple(sorted(set(values))))
        .to_dict()
    )
    branch_by_term = dict(zip(terms["term"], terms["branch"], strict=True))

    rows: list[dict[str, object]] = []
    for term in terms["term"]:
        term_branch = branch_by_term[term]
        queue: deque[tuple[str, int]] = deque((parent, 1) for parent in parents_by_term.get(term, ()))
        seen: dict[str, int] = {}

        while queue:
            ancestor, distance = queue.popleft()
            if ancestor == term:
                continue
            if ancestor in seen and seen[ancestor] <= distance:
                continue

            seen[ancestor] = distance
            for parent in parents_by_term.get(ancestor, ()):
                queue.append((parent, distance + 1))

        for ancestor, distance in seen.items():
            rows.append(
                {
                    "term": term,
                    "ancestor": ancestor,
                    "distance": int(distance),
                    "term_branch": term_branch,
                    "ancestor_branch": branch_by_term[ancestor],
                }
            )

    if not rows:
        return _empty_ancestor_frame()

    ancestors = pd.DataFrame.from_records(rows)
    ancestors = ancestors.loc[ancestors["term"] != ancestors["ancestor"]]
    ancestors = ancestors.sort_values(["term", "distance", "ancestor"], kind="mergesort").reset_index(drop=True)
    return ancestors


def build_go_tables(
    obo_path: str | Path,
    relations: Iterable[str] = CLOSURE_RELATIONS,
) -> GoTables:
    """Parse GO OBO and return canonical active term, edge, and ancestor tables."""

    parsed_terms, raw_edges = parse_go_obo(obo_path)
    all_terms = _terms_to_frame(parsed_terms)
    active_terms = all_terms.loc[(~all_terms["is_obsolete"]) & all_terms["branch"].notna()].copy()
    active_terms = active_terms.sort_values("term", kind="mergesort").reset_index(drop=True)

    alt_ids = _make_alt_id_frame(active_terms)
    edges, edge_stats = _filter_edges(raw_edges, active_terms, relations=relations)
    ancestors = build_ancestor_table(active_terms, edges)

    report = {
        "relations_used_for_closure": sorted(set(relations)),
        "all_term_count": int(len(all_terms)),
        "active_term_count": int(len(active_terms)),
        "obsolete_term_count": int(all_terms["is_obsolete"].sum()),
        "unknown_namespace_term_count": int(all_terms["branch"].isna().sum()),
        "active_alt_id_count": int(len(alt_ids)),
        "edge_count": int(len(edges)),
        "ancestor_row_count": int(len(ancestors)),
        "edge_filtering": {name: int(value) for name, value in edge_stats.items()},
        "active_terms_by_branch": {
            branch: int(count)
            for branch, count in active_terms.groupby("branch")["term"].count().sort_index().items()
        },
    }

    return GoTables(
        terms=active_terms,
        edges=edges,
        ancestors=ancestors,
        alt_ids=alt_ids,
        all_terms=all_terms,
        report=report,
    )


def close_training_terms(
    train_terms: pd.DataFrame,
    terms: pd.DataFrame,
    ancestors: pd.DataFrame,
    alt_ids: pd.DataFrame,
    all_terms: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Expand direct training labels with ontology ancestors."""

    active_ids = set(terms["term"])
    all_term_ids = set(all_terms["term"])
    obsolete_ids = set(all_terms.loc[all_terms["is_obsolete"], "term"])
    alt_to_term = dict(zip(alt_ids["alt_id"], alt_ids["term"], strict=True)) if not alt_ids.empty else {}

    resolved = train_terms.copy()
    resolved["source_term"] = resolved["term"]
    resolved["canonical_term"] = resolved["term"].where(resolved["term"].isin(active_ids), resolved["term"].map(alt_to_term))

    unresolved = resolved.loc[resolved["canonical_term"].isna(), "source_term"].drop_duplicates()
    unresolved_terms = set(unresolved)
    obsolete_training_terms = sorted(unresolved_terms.intersection(obsolete_ids))
    missing_training_terms = sorted(unresolved_terms.difference(all_term_ids).difference(alt_to_term))
    alt_mapped_mask = resolved["canonical_term"].notna() & (resolved["source_term"] != resolved["canonical_term"])
    alt_mapped_terms = sorted(set(resolved.loc[alt_mapped_mask, "source_term"].dropna()))

    resolved = resolved.dropna(subset=["canonical_term"]).copy()
    term_info = terms.loc[:, ["term", "branch"]].rename(columns={"term": "canonical_term", "branch": "ontology_branch"})
    resolved = resolved.merge(term_info, on="canonical_term", how="left")
    resolved["ontology_aspect"] = resolved["ontology_branch"].map(BRANCH_TO_ASPECT)

    branch_mismatches = sorted(
        resolved.loc[resolved["branch"] != resolved["ontology_branch"], "source_term"].drop_duplicates().tolist()
    )

    direct_rows = resolved.loc[:, ["entry_id", "source_term", "canonical_term", "ontology_aspect", "ontology_branch"]].rename(
        columns={
            "canonical_term": "term",
            "ontology_aspect": "aspect",
            "ontology_branch": "branch",
        }
    )
    direct_rows["is_direct"] = True
    direct_rows["distance"] = 0

    ancestor_seed = resolved.loc[:, ["entry_id", "source_term", "canonical_term"]]
    ancestor_rows = ancestor_seed.merge(ancestors, left_on="canonical_term", right_on="term", how="inner")
    if ancestor_rows.empty:
        ancestor_output = pd.DataFrame(columns=direct_rows.columns)
    else:
        ancestor_output = ancestor_rows.loc[:, ["entry_id", "source_term", "ancestor", "distance", "ancestor_branch"]].rename(
            columns={
                "ancestor": "term",
                "ancestor_branch": "branch",
            }
        )
        ancestor_output["aspect"] = ancestor_output["branch"].map(BRANCH_TO_ASPECT)
        ancestor_output["is_direct"] = False
        ancestor_output = ancestor_output.loc[:, direct_rows.columns]

    closed = pd.concat([direct_rows, ancestor_output], ignore_index=True)
    if closed.empty:
        output = pd.DataFrame(columns=["entry_id", "term", "aspect", "branch", "is_direct", "distance", "n_sources"])
    else:
        output = (
            closed.groupby(["entry_id", "term", "aspect", "branch"], as_index=False)
            .agg(
                is_direct=("is_direct", "max"),
                distance=("distance", "min"),
                n_sources=("source_term", "nunique"),
            )
            .sort_values(["entry_id", "branch", "term"], kind="mergesort")
            .reset_index(drop=True)
        )
        output["is_direct"] = output["is_direct"].astype(bool)
        output["distance"] = output["distance"].astype(int)
        output["n_sources"] = output["n_sources"].astype(int)

    report = {
        "direct_annotation_rows": int(len(train_terms)),
        "closed_annotation_rows": int(len(output)),
        "unique_direct_terms": int(train_terms["term"].nunique()),
        "unique_closed_terms": int(output["term"].nunique()) if not output.empty else 0,
        "alt_id_mapped_training_term_count": int(len(alt_mapped_terms)),
        "alt_id_mapped_training_terms": alt_mapped_terms,
        "obsolete_training_term_count": int(len(obsolete_training_terms)),
        "obsolete_training_terms": obsolete_training_terms,
        "missing_training_term_count": int(len(missing_training_terms)),
        "missing_training_terms": missing_training_terms,
        "branch_mismatch_training_term_count": int(len(branch_mismatches)),
        "branch_mismatch_training_terms": branch_mismatches,
        "closed_rows_by_branch": {
            branch: int(count)
            for branch, count in output.groupby("branch")["term"].count().sort_index().items()
        }
        if not output.empty
        else {},
    }
    return output, report
