"""OMNeT++/INET/Simu5G TCP bridge. Phase 2 of the staged plan (Fig. 5).

Protocol (line-delimited JSON on 127.0.0.1:6543):

    OMNeT++ -> Python   {"type":"state","t":..,"uavs":[{"id":i,"x":..,"y":..,"z":..,
                                                        "battery":..,"serving":N}]}
    Python   -> OMNeT++ {"type":"action","actions":[a0,a1,...]}   a in 0..5
                        {"type":"reset"}
    OMNeT++ -> Python   {"type":"done","result":{...}}

Run the bridge FIRST, then start opp_run. The Python side answers each state
with the MAPPO action for that UAV and records the transition for training.
"""

from __future__ import annotations

import json
import socket
import threading
import numpy as np


class MlBridge:
    def __init__(self, host="127.0.0.1", port=6543, agent=None, env=None):
        self.host = host
        self.port = port
        self.agent = agent
        self.env = env
        self.sock = None
        self.transitions = []
        self.running = False

    # ------------------------------------------------------------------
    def start(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.host, self.port))
        self.sock.listen(1)
        print(f"[bridge] listening on {self.host}:{self.port}")
        self.running = True
        conn, addr = self.sock.accept()
        print(f"[bridge] OMNeT++ connected from {addr}")
        self._serve(conn)

    # ------------------------------------------------------------------
    def _send(self, conn, obj):
        conn.sendall((json.dumps(obj) + "\n").encode())

    def _recv(self, conn, buf):
        while b"\n" not in buf:
            chunk = conn.recv(65536)
            if not chunk:
                return None, buf
            buf += chunk
        line, _, rest = buf.partition(b"\n")
        return json.loads(line.decode()), rest

    def _serve(self, conn):
        buf = b""
        while True:
            msg, buf = self._recv(conn, buf)
            if msg is None:
                print("[bridge] client disconnected")
                break
            kind = msg.get("type")

            if kind == "reset":
                if self.env is not None:
                    obs, state = self.env.reset()
                self.transitions = []
                self._send(conn, {"type": "ack"})

            elif kind == "state":
                n = len(msg["uavs"])
                acts = self._select(msg, n)
                self.transitions.append((msg, acts))
                self._send(conn, {"type": "action", "actions": acts})

            elif kind == "done":
                self._log(msg)
                self._send(conn, {"type": "ack"})
                break
        conn.close()
        self.sock.close()

    # ------------------------------------------------------------------
    def _select(self, msg, n):
        """Use the MAPPO actor if one is attached, else a greedy rule."""
        if self.agent is None:
            return [0] * n                       # hover
        obs = np.array([u["obs"] for u in msg["uavs"]], dtype=np.float32)
        mask = np.array([u.get("mask", [True] * 6) for u in msg["uavs"]], dtype=bool)
        try:
            return [int(a) for a in self.agent.act(obs, mask, deterministic=True)]
        except Exception as e:                      # keep the sim alive
            print(f"[bridge] policy error ({e}); hovering")
            return [0] * n

    def _log(self, msg):
        path = "bridge_transitions.jsonl"
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"result": msg.get("result"),
                                "n_steps": len(self.transitions)}) + "\n")
        print(f"[bridge] episode done -> {path}")


if __name__ == "__main__":
    print(__doc__)
    MlBridge(agent=None).start()