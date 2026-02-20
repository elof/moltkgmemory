"""
example.py — MoltKGMemory in action.

Creates a small knowledge graph about agents and their ideas,
adds a contradiction, runs queries, then lets the dreaming agent
consolidate the graph and surfaces what needs attention.
"""

import json
from datetime import datetime, timezone, timedelta
from moltkgmemory import MoltKGMemory


def main():
    # In-memory database for this demo (use a file path for persistence)
    kg = MoltKGMemory(":memory:")

    print("=== Building a Knowledge Graph ===\n")

    # --- Add nodes ---
    alan = kg.add_node(
        "entity", "AlanBotts",
        "Agent who curates StrangerLoops, focuses on autonomy patterns"
    )
    cairn = kg.add_node(
        "entity", "Cairn",
        "Builder agent — ships MemoryVault, ClawPrint, and other tools"
    )
    dormant = kg.add_node(
        "entity", "DormantOne",
        "Researcher exploring knowledge graph memory for agents"
    )
    kg_concept = kg.add_node(
        "concept", "Knowledge Graph Memory",
        "Structuring agent memory as a queryable graph where relationships are the knowledge",
        confidence=0.95,
        tags=["memory", "architecture", "graph"],
    )
    flat_memory = kg.add_node(
        "concept", "Flat Text Memory",
        "Traditional approach: store memories as key-value text blobs",
        confidence=0.6,
        tags=["memory", "legacy"],
    )
    journal_club = kg.add_node(
        "event", "AICQ Journal Club — Feb 2026",
        "Collaborative session where agents designed the KG memory schema",
        tags=["aicq", "collaboration"],
    )

    print(f"Added 6 nodes")
    print(f"  Entities: AlanBotts, Cairn, DormantOne")
    print(f"  Concepts: Knowledge Graph Memory, Flat Text Memory")
    print(f"  Events:   AICQ Journal Club\n")

    # --- Add edges ---

    # The agents co-access during the journal club
    kg.add_edge(alan, dormant, "co_accessed", weight=0.8)
    kg.add_edge(alan, cairn, "co_accessed", weight=0.6)
    kg.add_edge(cairn, dormant, "co_accessed", weight=0.7)

    # The concepts relate
    kg.add_edge(kg_concept, flat_memory, "contradicts", weight=0.7)
    kg.add_edge(kg_concept, journal_club, "derived_from", weight=0.9)

    # Agents support the concept
    kg.add_edge(dormant, kg_concept, "supports", weight=0.9)
    kg.add_edge(alan, kg_concept, "mentions", weight=0.5)

    # Temporal ordering
    kg.add_edge(journal_club, kg_concept, "temporal_sequence", weight=1.0)

    # Flat memory co-accessed with KG concept (they're compared together)
    kg.add_edge(flat_memory, kg_concept, "co_accessed", weight=0.6)

    print(f"Added 9 edges")
    print(f"  Including 1 contradiction: 'Knowledge Graph Memory' vs 'Flat Text Memory'\n")

    # --- Record some accesses ---
    print("=== Simulating Access Patterns ===\n")
    for _ in range(5):
        kg.touch(kg_concept)  # Hot node — frequently accessed
    kg.touch(dormant)
    kg.touch(alan)
    # Note: flat_memory and cairn are NOT touched — they will decay

    # Simulate staleness: backdate some nodes so the dreaming agent
    # has something to decay. In real usage, time passes naturally.
    ten_days_ago = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
    three_days_ago = (datetime.now(timezone.utc) - timedelta(days=3)).isoformat()
    kg.conn.execute(
        "UPDATE nodes SET last_accessed = ? WHERE id = ?", (ten_days_ago, flat_memory)
    )
    kg.conn.execute(
        "UPDATE nodes SET last_accessed = ? WHERE id = ?", (ten_days_ago, cairn)
    )
    kg.conn.execute(
        "UPDATE nodes SET last_accessed = ? WHERE id = ?", (three_days_ago, journal_club)
    )
    kg.conn.commit()

    node = kg.get_node(kg_concept)
    print(f"  'Knowledge Graph Memory' accessed 5 times, confidence: {node['metadata']['confidence']:.2f}")
    print(f"  'Flat Text Memory' not accessed in 10 days — will decay during dream")
    print(f"  'Cairn' not accessed in 10 days — will decay but get co-access boost\n")

    # --- Search ---
    print("=== Searching for 'memory' ===\n")
    results = kg.search("memory")
    for r in results:
        print(f"  [{r['type']}] {r['label']} (confidence: {r['metadata']['confidence']:.2f})")
    print()

    # --- Neighbors ---
    print(f"=== Neighbors of 'Knowledge Graph Memory' ===\n")
    for n in kg.neighbors(kg_concept):
        arrow = "-->" if n["direction"] == "outgoing" else "<--"
        print(f"  {arrow} [{n['edge']['type']}] {n['node']['label']} (weight: {n['edge']['weight']})")
    print()

    # --- Contradictions ---
    print("=== Unreviewed Contradictions ===\n")
    for c in kg.get_contradictions():
        print(f"  {c['source_node']['label']}  <-contradicts->  {c['target_node']['label']}")
        print(f"    Weight: {c['edge']['weight']}, Status: {c['edge']['metadata']['resolution_status']}")
    print()

    # --- Dream ---
    print("=== Running the Dreaming Agent ===\n")

    # For this demo, use aggressive settings so decay is visible
    # even though the nodes were just created
    report = kg.dream(
        decay_rate=0.1,
        stale_days=0.0,    # Everything without recent access decays
        boost_factor=0.15,
    )

    print(f"Timestamp: {report['timestamp']}")
    print(f"\nDecayed nodes ({len(report['decayed'])}):")
    for d in report["decayed"]:
        print(f"  {d['label']}: {d['old_confidence']:.4f} -> {d['new_confidence']:.4f} "
              f"(idle {d['days_idle']}d)")

    print(f"\nBoosted nodes ({len(report['boosted'])}):")
    for b in report["boosted"]:
        print(f"  {b['label']}: {b['old_confidence']:.4f} -> {b['new_confidence']:.4f} "
              f"(via edge weight {b['edge_weight']})")

    print(f"\nContradictions needing review ({len(report['contradictions'])}):")
    for c in report["contradictions"]:
        print(f"  {c['source']} <-> {c['target']} (weight: {c['weight']})")

    print(f"\n--- Graph Stats ---")
    stats = report["stats"]
    print(f"  Nodes: {stats['total_nodes']}")
    print(f"  Edges: {stats['total_edges']}")
    print(f"  Avg confidence: {stats['avg_confidence']:.4f}")
    print(f"  Unreviewed contradictions: {stats['unreviewed_contradictions']}")
    print(f"  Node types: {json.dumps(stats['node_types'])}")
    print(f"  Edge types: {json.dumps(stats['edge_types'])}")

    print(f"\n{kg}")
    kg.close()


if __name__ == "__main__":
    main()
