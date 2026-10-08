import re, sys
p = sys.argv[1] if len(sys.argv) > 1 else 'src/gsv_hubert.cpp'
s = open(p, 'r', encoding='utf-8').read()
BS = chr(92)
bad = '"' + BS + 'n""'
n = s.count(bad)
s = s.replace(bad, BS + 'n"')
open(p, 'w', encoding='utf-8').write(s)
print('collapsed', n, 'bad sequences in', p)
