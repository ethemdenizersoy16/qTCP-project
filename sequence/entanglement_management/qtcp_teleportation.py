from sequence.entanglement_management.teleportation import (
    TeleportProtocol,
    TeleportMessage,
    TeleportMsgType,
)
from sequence.utils import log
 
 
class QTCPTeleportMessage(TeleportMessage):
    """A teleport correction message that also carries the qTCP transfer id."""
 
    def __init__(self, msg_type: TeleportMsgType, bob_comm_memory_name: str,
                 x_flip: int, z_flip: int, reservation, transfer_id: int):
        super().__init__(
            msg_type,
            bob_comm_memory_name=bob_comm_memory_name,
            x_flip=x_flip,
            z_flip=z_flip,
            reservation=reservation,
        )
        self.transfer_id = transfer_id
 
 
class QTCPTeleportProtocol(TeleportProtocol):
    """TeleportProtocol that carries a qTCP transfer id end to end.
 
    Alice stamps the id onto the correction message; Bob reads it off and hands
    it to teleport_complete(), so the app never has to infer the transfer from
    a comm memory name.
    """
 
    def __init__(self, owner, alice: bool, transfer_id: int,
                 data_memory_index: int = None, remote_node_name: str = None):
        super().__init__(
            owner,
            alice=alice,
            data_memory_index=data_memory_index,
            remote_node_name=remote_node_name,
        )
        self.transfer_id = transfer_id
 
    def alice_bell_measurement(self, reservation):
        """Identical to the parent, except the correction message is a
        QTCPTeleportMessage carrying self.transfer_id."""
        comm_key = self.alice_comm_memory.qstate_key
        data_memory_array = self.owner.get_component_by_name(self.owner.data_memo_arr_name)
        data_key = data_memory_array[self.data_memory_index].qstate_key
        log.logger.debug(
            f"{self.name}: alice_bell_measure data_key={data_key}, comm_key={comm_key}"
        )
 
        rnd = self.owner.get_generator().random()
        meas = self.owner.timeline.quantum_manager.run_circuit(
            TeleportProtocol._bsm_circuit, [data_key, comm_key], rnd
        )
        z, x = meas[data_key], meas[comm_key]
        log.logger.info(
            f"{self.name} bell measurement results: x={x}, z={z}, "
            f"remote memory={self.bob_comm_memory_name}"
        )
 
        msg = QTCPTeleportMessage(
            TeleportMsgType.MEASUREMENT_RESULT,
            bob_comm_memory_name=self.bob_comm_memory_name,
            x_flip=x,
            z_flip=z,
            reservation=reservation,
            transfer_id=self.transfer_id,
        )
        self.owner.send_message(self.remote_node_name, msg)
 
    def bob_handle_correction(self, msg: QTCPTeleportMessage):
        """Identical to the parent, except teleport_complete() is called with
        the transfer id read off the message."""
        log.logger.debug(
            f"{self.name}: bob_handle_correction, memory={msg.bob_comm_memory_name}, "
            f"x_flip={msg.x_flip}, z_flip={msg.z_flip}"
        )
        bob_comm_memory_key = self.bob_comm_memory.qstate_key
 
        if msg.x_flip:
            rnd = self.owner.get_generator().random()
            self.owner.timeline.quantum_manager.run_circuit(
                TeleportProtocol._x_flip_circuit, [bob_comm_memory_key], rnd
            )
            log.logger.info(f"{self.name}: X-flip applied on memory {msg.bob_comm_memory_name}")
        if msg.z_flip:
            rnd = self.owner.get_generator().random()
            self.owner.timeline.quantum_manager.run_circuit(
                TeleportProtocol._z_flip_circuit, [bob_comm_memory_key], rnd
            )
            log.logger.info(f"{self.name}: Z-flip applied on memory {msg.bob_comm_memory_name}")
 
        self.owner.teleport_app.teleport_complete(bob_comm_memory_key, msg.transfer_id)