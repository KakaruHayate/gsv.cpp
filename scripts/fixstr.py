import sys
p = sys.argv[1]
src = open(p, 'r', encoding='utf-8').read()
out = []
i = 0
n = len(src)
in_str = False
in_line_comment = False
fixed = 0
BS = chr(92)
while i < n:
    c = src[i]
    if in_line_comment:
        out.append(c)
        if c == '\n':
            in_line_comment = False
        i += 1
        continue
    if in_str:
        if c == BS and i + 1 < n:
            out.append(c); out.append(src[i + 1]); i += 2; continue
        if c == '"':
            in_str = False; out.append(c); i += 1; continue
        if c == '\n':
            out.append('"'); out.append(BS); out.append('n'); out.append('"')
            fixed += 1
            i += 1
            continue
        out.append(c); i += 1; continue
    if src.startswith('//', i):
        in_line_comment = True
        out.append('//'); i += 2; continue
    if c == '"':
        in_str = True; out.append(c); i += 1; continue
    out.append(c); i += 1
open(p, 'w', encoding='utf-8').write(''.join(out))
print('fixed', fixed, 'embedded newlines')
