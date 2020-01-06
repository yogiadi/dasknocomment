from collections import deque
from dask.core import istask, subs
def head(task):
    
    if istask(task):
        return task[0]
    elif isinstance(task, list):
        return list
    else:
        return task
def args(task):
    
    if istask(task):
        return task[1:]
    elif isinstance(task, list):
        return task
    else:
        return ()
class Traverser(object):
    
    def __init__(self, term, stack=None):
        self.term = term
        if not stack:
            self._stack = deque([END])
        else:
            self._stack = stack
    def __iter__(self):
        while self.current is not END:
            yield self.current
            self.next()
    def copy(self):
        
        return Traverser(self.term, deque(self._stack))
    def next(self):
        
        subterms = args(self.term)
        if not subterms:
            # No subterms, pop off stack
            self.term = self._stack.pop()
        else:
            self.term = subterms[0]
            self._stack.extend(reversed(subterms[1:]))
    @property
    def current(self):
        return head(self.term)
    def skip(self):
        
        self.term = self._stack.pop()
class Token(object):
    
    def __init__(self, name):
        self.name = name
    def __repr__(self):
        return self.name
# A variable to represent *all* variables in a discrimination net
VAR = Token("?")
# Represents the end of the traversal of an expression. We can't use `None`,
# 'False', etc... here, as anything may be an argument to a function.
END = Token("end")
class Node(tuple):
    
    __slots__ = ()
    def __new__(cls, edges=None, patterns=None):
        edges = edges if edges else {}
        patterns = patterns if patterns else []
        return tuple.__new__(cls, (edges, patterns))
    @property
    def edges(self):
        
        return self[0]
    @property
    def patterns(self):
        
        return self[1]
class RewriteRule(object):
    
    def __init__(self, lhs, rhs, vars=()):
        if not isinstance(vars, tuple):
            raise TypeError("vars must be a tuple of variables")
        self.lhs = lhs
        if callable(rhs):
            self.subs = rhs
        else:
            self.subs = self._apply
        self.rhs = rhs
        self._varlist = [t for t in Traverser(lhs) if t in vars]
        # Reduce vars down to just variables found in lhs
        self.vars = tuple(sorted(set(self._varlist)))
    def _apply(self, sub_dict):
        term = self.rhs
        for key, val in sub_dict.items():
            term = subs(term, key, val)
        return term
    def __str__(self):
        return "RewriteRule({0}, {1}, {2})".format(self.lhs, self.rhs, self.vars)
    def __repr__(self):
        return str(self)
class RuleSet(object):
    
    def __init__(self, *rules):
        
        self._net = Node()
        self.rules = []
        for p in rules:
            self.add(p)
    def add(self, rule):
        
        if not isinstance(rule, RewriteRule):
            raise TypeError("rule must be instance of RewriteRule")
        vars = rule.vars
        curr_node = self._net
        ind = len(self.rules)
        # List of variables, in order they appear in the POT of the term
        for t in Traverser(rule.lhs):
            prev_node = curr_node
            if t in vars:
                t = VAR
            if t in curr_node.edges:
                curr_node = curr_node.edges[t]
            else:
                curr_node.edges[t] = Node()
                curr_node = curr_node.edges[t]
        # We've reached a leaf node. Add the term index to this leaf.
        prev_node.edges[t].patterns.append(ind)
        self.rules.append(rule)
    def iter_matches(self, term):
        
        S = Traverser(term)
        for m, syms in _match(S, self._net):
            for i in m:
                rule = self.rules[i]
                subs = _process_match(rule, syms)
                if subs is not None:
                    yield rule, subs
    def _rewrite(self, term):
        
        for rule, sd in self.iter_matches(term):
            # We use for (...) because it's fast in all cases for getting the
            # first element from the match iterator. As we only want that
            # element, we break here
            term = rule.subs(sd)
            break
        return term
    def rewrite(self, task, strategy="bottom_up"):
        
        return strategies[strategy](self, task)
def _top_level(net, term):
    return net._rewrite(term)
def _bottom_up(net, term):
    if istask(term):
        term = (head(term),) + tuple(_bottom_up(net, t) for t in args(term))
    elif isinstance(term, list):
        term = [_bottom_up(net, t) for t in args(term)]
    return net._rewrite(term)
strategies = {"top_level": _top_level, "bottom_up": _bottom_up}
def _match(S, N):
    
    stack = deque()
    restore_state_flag = False
    # matches are stored in a tuple, because all mutations result in a copy,
    # preventing operations from changing matches stored on the stack.
    matches = ()
    while True:
        if S.current is END:
            yield N.patterns, matches
        try:
            # This try-except block is to catch hashing errors from un-hashable
            # types. This allows for variables to be matched with un-hashable
            # objects.
            n = N.edges.get(S.current, None)
            if n and not restore_state_flag:
                stack.append((S.copy(), N, matches))
                N = n
                S.next()
                continue
        except TypeError:
            pass
        n = N.edges.get(VAR, None)
        if n:
            restore_state_flag = False
            matches = matches + (S.term,)
            S.skip()
            N = n
            continue
        try:
            # Backtrack here
            (S, N, matches) = stack.pop()
            restore_state_flag = True
        except Exception:
            return
def _process_match(rule, syms):
    
    subs = {}
    varlist = rule._varlist
    if not len(varlist) == len(syms):
        raise RuntimeError("length of varlist doesn't match length of syms.")
    for v, s in zip(varlist, syms):
        if v in subs and subs[v] != s:
            return None
        else:
            subs[v] = s
    return subs
