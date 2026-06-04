"""Abstract FSMState base — mirrors deploy/include/FSM/FSMState.h."""


class FSMState:
    name: str = "base"

    def __init__(self, env, ort_runner=None, cfg=None, rc=None):
        self.env = env
        self.ort = ort_runner
        self.cfg = cfg or {}
        self.rc = rc

    def enter(self):
        pass

    def run(self):
        """Advance one policy step. Returns nothing."""
        raise NotImplementedError

    def exit(self):
        pass

    def check_transition(self, requested: str):
        """Return new state name (str) if a transition is allowed; else None.
        Subclasses encode the allowed source->target rules."""
        return None
