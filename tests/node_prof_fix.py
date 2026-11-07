#!/usr/bin/env python3
# Repair the broken per-graph print block in the local ggml-cpu node-prof patch
# (the previous heredoc turned '\n' inside string literals into real newlines).
import re
import sys

p = sys.argv[1]
s = open(p).read()

# Collapse real newlines that were meant to be \n escapes inside fprintf literals.
bad = [
    ('graph nodes=%d total=%.2f ms\n", cgraph->n_nodes, tot0);', None),
]
# Generic repair: any '"...ms\n\t\t\t", ' pattern where the newline is literal.
pat = re.compile(r'ms\n(\s*)", ')

def repl(m):
    return 'ms\\n", '

s2 = pat.sub(repl, s)
if s2 != s:
    open(p, 'w').write(s2)
    print("repaired")
else:
    print("nothing matched")
