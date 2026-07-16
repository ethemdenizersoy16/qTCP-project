from networkx import Graph, dijkstra_path, exception

from .topology import Topology as Topo
from .node import BSMNode
from .node import QTCPNode
from .dqc_net_topo import DQCNetTopo

class QTCPNetTopo(DQCNetTopo):
    QTCP_NODE = "QTCPNode"

    def _add_nodes(self, config: dict):
        for node in config[Topo.ALL_NODE]:
            seed = node[Topo.SEED]
            node_type = node[Topo.TYPE]
            name = node[Topo.NAME]
            template = self.templates.get(node.get(Topo.TEMPLATE, None), {})

            if node_type == self.BSM_NODE:
                others = self.bsm_to_router_map[name]
                node_obj = BSMNode(name, self.tl, others, component_templates=template)
            elif node_type == self.QTCP_NODE:
                node_obj = QTCPNode(
                    name, self.tl,
                    memo_size=node.get(self.MEMO_ARRAY_SIZE, 0),
                    data_memo_size=node.get(self.DATA_MEMO_ARRAY_SIZE, 0),
                    component_templates=template,
                    # DQCNetTopo drops these on the floor; we don't.
                    gate_fid=node.get(Topo.GATE_FIDELITY, 1),
                    meas_fid=node.get(Topo.MEASUREMENT_FIDELITY, 1),
                )
            else:
                raise ValueError(f"Unknown type of node '{node_type}'")

            node_obj.set_seed(seed)
            self.nodes[node_type].append(node_obj)
   
    def _generate_forwarding_table(self, config: dict):
        """For static routing."""
        graph = Graph()
        for node in config[Topo.ALL_NODE]:
            if node[Topo.TYPE] == self.QTCP_NODE:
                graph.add_node(node[Topo.NAME])

        costs = {}
        for qc in self.qchannels:
            router, bsm = qc.sender.name, qc.receiver
            if bsm not in costs:
                costs[bsm] = [router, qc.distance]
            else:
                costs[bsm] = [router] + costs[bsm]
                costs[bsm][-1] += qc.distance

        graph.add_weighted_edges_from(costs.values())
        for src in self.nodes[self.QTCP_NODE]:
            for dst_name in graph.nodes:
                if src.name == dst_name:
                    continue
                try:
                    if dst_name > src.name:
                        path = dijkstra_path(graph, src.name, dst_name)
                    else:
                        path = dijkstra_path(graph, dst_name, src.name)[::-1]
                    next_hop = path[1]
                    # routing protocol locates at the bottom of the stack
                    routing_protocol = src.network_manager.get_routing_protocol()
                    routing_protocol.update_forwarding_rule(dst_name, next_hop)
                except exception.NetworkXNoPath:
                    pass