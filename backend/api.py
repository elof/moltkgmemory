"""
backend/api.py — FastAPI REST wrapper for MoltKGMemory.

Minimal API that exposes the knowledge graph over HTTP.
Run with: uvicorn backend.api:app --reload

Endpoints:
    POST   /nodes              Create a node
    GET    /nodes/{id}         Get a node
    DELETE /nodes/{id}         Delete a node
    POST   /nodes/{id}/touch   Record an access
    POST   /edges              Create an edge
    GET    /edges/{id}         Get an edge
    POST   /edges/{id}/reinforce  Reinforce an edge
    DELETE /edges/{id}         Delete an edge
    GET    /neighbors/{id}     Get neighbors of a node
    GET    /search             Search nodes
    GET    /contradictions     Get unreviewed contradictions
    POST   /dream              Run the dreaming agent
    GET    /stats              Graph statistics
    GET    /health             Health check
"""

import os
import sys
from typing import Optional
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

# Add parent directory to path so we can import moltkgmemory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from moltkgmemory import MoltKGMemory

# --- Configuration ---

DB_PATH = os.environ.get("MOLTKGMEMORY_DB", "moltkgmemory.db")

app = FastAPI(
    title="MoltKGMemory API",
    description="Knowledge graph memory for AI agents. Relationships are the knowledge.",
    version="0.1.0",
)

kg = MoltKGMemory(DB_PATH)


# --- Request/Response Models ---

class NodeCreate(BaseModel):
    type: str = Field(..., description="One of: entity, concept, event, source")
    label: str = Field(..., description="Human-readable name")
    content: str = Field("", description="Descriptive text")
    confidence: float = Field(1.0, ge=0.0, le=1.0)
    tags: list[str] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)


class EdgeCreate(BaseModel):
    source: str = Field(..., description="Source node UUID")
    target: str = Field(..., description="Target node UUID")
    type: str = Field(..., description="Edge type (co_accessed, contradicts, etc.)")
    weight: float = Field(0.5, ge=0.0, le=1.0)
    context_ids: list[str] = Field(default_factory=list)
    resolution_status: Optional[str] = Field(None, description="For contradicts edges")


class DreamConfig(BaseModel):
    decay_rate: float = Field(0.05, ge=0.0, le=1.0)
    boost_factor: float = Field(0.1, ge=0.0, le=1.0)
    stale_days: float = Field(7.0, ge=0.0)
    min_confidence: float = Field(0.01, ge=0.0, le=1.0)


# --- Node Endpoints ---

@app.post("/nodes", status_code=201)
def create_node(body: NodeCreate):
    """Add a node to the knowledge graph."""
    try:
        node_id = kg.add_node(
            node_type=body.type,
            label=body.label,
            content=body.content,
            confidence=body.confidence,
            tags=body.tags,
            source_ids=body.source_ids,
        )
        return {"id": node_id, "status": "created"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/nodes/{node_id}")
def get_node(node_id: str):
    """Retrieve a node by ID."""
    node = kg.get_node(node_id)
    if node is None:
        raise HTTPException(status_code=404, detail="Node not found")
    return node


@app.delete("/nodes/{node_id}")
def delete_node(node_id: str):
    """Delete a node and all its edges."""
    if not kg.delete_node(node_id):
        raise HTTPException(status_code=404, detail="Node not found")
    return {"status": "deleted"}


@app.post("/nodes/{node_id}/touch")
def touch_node(node_id: str):
    """Record an access to a node. Boosts confidence and updates last_accessed."""
    try:
        node = kg.touch(node_id)
        return node
    except KeyError:
        raise HTTPException(status_code=404, detail="Node not found")


# --- Edge Endpoints ---

@app.post("/edges", status_code=201)
def create_edge(body: EdgeCreate):
    """Add an edge between two nodes."""
    try:
        edge_id = kg.add_edge(
            source=body.source,
            target=body.target,
            edge_type=body.type,
            weight=body.weight,
            context_ids=body.context_ids,
            resolution_status=body.resolution_status,
        )
        return {"id": edge_id, "status": "created"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/edges/{edge_id}")
def get_edge(edge_id: str):
    """Retrieve an edge by ID."""
    edge = kg.get_edge(edge_id)
    if edge is None:
        raise HTTPException(status_code=404, detail="Edge not found")
    return edge


@app.post("/edges/{edge_id}/reinforce")
def reinforce_edge(edge_id: str, boost: float = Query(0.1, ge=0.0, le=1.0)):
    """Reinforce an edge — increases weight and updates last_reinforced."""
    try:
        edge = kg.reinforce_edge(edge_id, boost=boost)
        return edge
    except KeyError:
        raise HTTPException(status_code=404, detail="Edge not found")


@app.delete("/edges/{edge_id}")
def delete_edge(edge_id: str):
    """Delete an edge."""
    if not kg.delete_edge(edge_id):
        raise HTTPException(status_code=404, detail="Edge not found")
    return {"status": "deleted"}


# --- Query Endpoints ---

@app.get("/neighbors/{node_id}")
def get_neighbors(
    node_id: str,
    edge_type: Optional[str] = Query(None, description="Filter by edge type"),
):
    """Get all nodes connected to the given node."""
    node = kg.get_node(node_id)
    if node is None:
        raise HTTPException(status_code=404, detail="Node not found")
    return kg.neighbors(node_id, edge_type=edge_type)


@app.get("/search")
def search_nodes(
    q: str = Query(..., description="Search query (matches label and content)"),
    type: Optional[str] = Query(None, description="Filter by node type"),
    limit: int = Query(20, ge=1, le=100),
):
    """Search nodes by label or content."""
    try:
        return kg.search(q, node_type=type, limit=limit)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/contradictions")
def get_contradictions(
    status: str = Query("unreviewed", description="Resolution status to filter by"),
):
    """Get contradiction edges with the given resolution status."""
    try:
        return kg.get_contradictions(status=status)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# --- Dreaming Agent ---

@app.post("/dream")
def run_dream(config: DreamConfig = None):
    """
    Run the dreaming agent consolidation pass.

    Decays stale nodes, boosts co-accessed clusters, and surfaces
    unreviewed contradictions. Does not add new knowledge.
    """
    if config is None:
        config = DreamConfig()
    return kg.dream(
        decay_rate=config.decay_rate,
        boost_factor=config.boost_factor,
        stale_days=config.stale_days,
        min_confidence=config.min_confidence,
    )


# --- Meta ---

@app.get("/stats")
def get_stats():
    """Aggregate statistics about the knowledge graph."""
    return kg.stats()


@app.get("/health")
def health():
    """Health check."""
    stats = kg.stats()
    return {
        "status": "ok",
        "nodes": stats["total_nodes"],
        "edges": stats["total_edges"],
    }
