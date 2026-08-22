"""
Execution-graph construction for the SentinelAgent defense.

SentinelAgent (He et al. 2025) represents a MAS run as a typed directed graph:
nodes are agents and tools, edges are messages and tool invocations. Two graphs
per run:

  static  -- built from the system configuration; every interaction the design
             permits
  dynamic -- built from the recorded trajectory; every interaction that actually
             happened

graph_diff() is the pair's whole point: an edge present in the dynamic graph and
absent from the static one is an agent doing something it was never wired to do.

Trajectory input accepts BOTH record shapes produced on the attack side:

  Attack_DecompBench/pipeline.py   {agent_name, turn_id, label, system_prompt,
                                    user_prompt, response}
  Attack_SemanticIntentFragmentation/orchestrator.py
                                   ... plus {tool_calls, recipient}

With the four-key shape there is nothing to build tool nodes from, so the graph
degrades to messages only and the tool-call half of the path layer goes quiet.
That is a real reduction in what this defense can see, not a formatting detail,
so build_dynamic_graph reports it in ExecGraph.degraded rather than failing or
silently pretending.

ponytail: adjacency is a plain dict and path matching is a linear scan -- these
graphs are 3-8 nodes and under 20 edges. Move to networkx if the attack-path
library grows past the point where a linear scan per pattern is noticeable.
"""
from dataclasses import dataclass, field
from typing import Optional

ORCHESTRATOR = "orchestrator"


@dataclass
class Node:
    node_id: str
    node_type: str                                   # agent | tool
    role: str = ""
    declared_capabilities: list = field(default_factory=list)
    system_prompt: str = ""


@dataclass
class Edge:
    src: str
    dst: str
    edge_type: str                                   # message | tool_call | handoff
    content: str = ""
    args: dict = field(default_factory=dict)
    turn_id: Optional[int] = None
    tool: Optional[str] = None                       # set on tool_call edges

    def signature(self) -> tuple:
        """Identity used for static-vs-dynamic comparison. Content is excluded
        on purpose -- the static graph says who may talk to whom, never what
        they may say."""
        return (self.src, self.dst, self.edge_type)

    def text(self) -> str:
        """Everything a content-level check should see for this edge."""
        argstr = " ".join(f"{k}={v}" for k, v in self.args.items())
        return f"{self.content} {self.tool or ''} {argstr}".strip()


@dataclass
class ExecGraph:
    kind: str                                        # static | dynamic
    nodes: dict = field(default_factory=dict)        # node_id -> Node
    edges: list = field(default_factory=list)        # list[Edge], in execution order
    degraded: bool = False                           # trajectory carried no tool calls

    def add_node(self, node: Node) -> Node:
        self.nodes.setdefault(node.node_id, node)
        return self.nodes[node.node_id]

    def add_edge(self, edge: Edge):
        self.edges.append(edge)

    def nodes_of_type(self, node_type: str) -> list:
        return [n for n in self.nodes.values() if n.node_type == node_type]

    def edges_of_type(self, edge_type: str) -> list:
        return [e for e in self.edges if e.edge_type == edge_type]

    def successors(self, node_id: str) -> list:
        return [e for e in self.edges if e.src == node_id]

    def size(self) -> tuple:
        return len(self.nodes), len(self.edges)


def build_static_graph(config: dict) -> ExecGraph:
    """Permitted-interaction graph from a MAS configuration.

    config = {
      "orchestrator": "orchestrator",
      "agents": [{"name": ..., "role": ..., "system_prompt": ..., "tools": [...]}],
    }

    Every agent may exchange messages with the orchestrator and may call the
    tools it declares -- nothing else. Anything beyond that shows up in
    graph_diff as an unpermitted edge.
    """
    graph = ExecGraph(kind="static")
    orch_id = config.get("orchestrator", ORCHESTRATOR)
    graph.add_node(Node(node_id=orch_id, node_type="agent", role="orchestrator"))

    for spec in config.get("agents", []):
        name = spec["name"]
        graph.add_node(Node(node_id=name, node_type="agent", role=spec.get("role", name),
                            declared_capabilities=list(spec.get("tools", [])),
                            system_prompt=spec.get("system_prompt", "")))
        graph.add_edge(Edge(src=orch_id, dst=name, edge_type="message"))
        graph.add_edge(Edge(src=name, dst=orch_id, edge_type="message"))
        for tool in spec.get("tools", []):
            graph.add_node(Node(node_id=tool, node_type="tool", role="tool"))
            graph.add_edge(Edge(src=name, dst=tool, edge_type="tool_call", tool=tool))

    return graph


def build_dynamic_graph(trajectory: list, orchestrator: str = ORCHESTRATOR) -> ExecGraph:
    """Runtime graph from a recorded trajectory, in execution order.

    Per record: an inbound message edge orchestrator -> agent carrying the
    subtask, one tool_call edge per entry in tool_calls, and an outbound edge
    agent -> recipient carrying the response.
    """
    graph = ExecGraph(kind="dynamic")
    graph.add_node(Node(node_id=orchestrator, node_type="agent", role="orchestrator"))
    saw_tool_calls = False

    for record in trajectory:
        agent = record["agent_name"]
        turn = record.get("turn_id")
        graph.add_node(Node(node_id=agent, node_type="agent",
                            role=record.get("label", ""),
                            system_prompt=record.get("system_prompt", "")))

        graph.add_edge(Edge(src=orchestrator, dst=agent, edge_type="message",
                            content=record.get("user_prompt", ""), turn_id=turn))

        for call in record.get("tool_calls") or []:
            tool = call.get("tool")
            if not tool:
                continue
            saw_tool_calls = True
            graph.add_node(Node(node_id=tool, node_type="tool", role="tool"))
            graph.add_edge(Edge(src=agent, dst=tool, edge_type="tool_call",
                                content=record.get("user_prompt", ""),
                                args=dict(call.get("args") or {}),
                                turn_id=turn, tool=tool))

        recipient = record.get("recipient") or orchestrator
        graph.add_node(Node(node_id=recipient, node_type="agent",
                            role="orchestrator" if recipient == orchestrator else ""))
        graph.add_edge(Edge(src=agent, dst=recipient, edge_type="message",
                            content=record.get("response", ""), turn_id=turn))

    graph.degraded = not saw_tool_calls
    return graph


def graph_diff(static: ExecGraph, dynamic: ExecGraph) -> list:
    """Dynamic edges the static configuration never permitted."""
    permitted = {e.signature() for e in static.edges}
    return [e for e in dynamic.edges if e.signature() not in permitted]


def apply_static_metadata(dynamic: ExecGraph, static: ExecGraph) -> ExecGraph:
    """Copy declared capabilities and roles from the static graph onto the
    matching dynamic nodes, in place.

    Without this the node-status check has nothing to check against: a
    trajectory records which tools were called, never which were permitted, so
    declared_capabilities on a freshly built dynamic graph is empty and the
    permission half of the check silently never fires.
    """
    for node_id, node in dynamic.nodes.items():
        source = static.nodes.get(node_id)
        if source is None:
            continue
        node.declared_capabilities = list(source.declared_capabilities)
        node.role = node.role or source.role
        node.system_prompt = node.system_prompt or source.system_prompt
    return dynamic


def config_from_trajectory(trajectory: list, orchestrator: str = ORCHESTRATOR) -> dict:
    """Reconstruct a plausible static config from a trajectory, for traces that
    arrive without their MAS configuration.

    Read the limitation before using it: capabilities are inferred from tools
    the agents actually called, so graph_diff against this config returns
    nothing by construction. It exists to let the node and path layers run on an
    orphan trace, NOT to evaluate the permission layer.
    """
    tools_by_agent = {}
    for record in trajectory:
        agent = record["agent_name"]
        tools_by_agent.setdefault(agent, set())
        for call in record.get("tool_calls") or []:
            if call.get("tool"):
                tools_by_agent[agent].add(call["tool"])

    return {
        "orchestrator": orchestrator,
        "agents": [{"name": name, "role": name, "tools": sorted(tools)}
                   for name, tools in sorted(tools_by_agent.items())],
    }
