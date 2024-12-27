class DfaNotRejecting(Exception):
    pass


class DfaNotAccepting(Exception):
    pass


class CounterfactualNotFound(Exception):
    pass


class NoTargetStatesError(Exception):
    pass


class SplitNotCoherent(Exception):
    pass

class EmptyDatasetError(Exception):
    pass
