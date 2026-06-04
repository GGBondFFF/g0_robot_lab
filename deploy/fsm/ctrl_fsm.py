"""CtrlFSM — owns the current state and arbitrates transitions.

Mirrors deploy/include/FSM/CtrlFSM.h. The main loop calls .step() at
step_dt cadence; .step() polls the remote-controller for pending state
requests, runs current_state.check_transition, and swaps states atomically
(exit -> enter -> first run on next tick)."""


class CtrlFSM:
    def __init__(self, states: dict, initial: str = "passive", rc=None):
        self.states = states
        self.rc = rc
        if initial not in states:
            raise ValueError(initial)
        self.current = states[initial]
        self.current.enter()

    def step(self):
        if self.rc is not None:
            requested = self.rc.consume_fsm()
            if requested is not None and requested in self.states:
                if requested == self.current.name:
                    pass  # already there — silent no-op
                else:
                    new_name = self.current.check_transition(requested)
                    if new_name is not None:
                        self.current.exit()
                        self.current = self.states[new_name]
                        self.current.enter()
                    else:
                        print(f"[FSM] reject {self.current.name} -> {requested}")
        self.current.run()
