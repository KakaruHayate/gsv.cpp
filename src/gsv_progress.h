#pragma once
// Coarse synthesis progress shared with the HTTP layer (/progress). Cheap atomics, no locks.
#include <atomic>

namespace gsvprog {
// 0 idle, 1 text frontend, 2 AR, 3 DiT/CFM, 4 vocoder, 5 done
inline std::atomic<int> stage{0};
inline std::atomic<int> cur{0};
inline std::atomic<int> total{0};

inline void reset() {
    stage.store(0);
    cur.store(0);
    total.store(0);
}
inline void begin(int st, int t = 0) {
    cur.store(0);
    total.store(t);
    stage.store(st);
}
inline void tick() { cur.fetch_add(1); }
inline void done() {
    stage.store(5);
    cur.store(1);
    total.store(1);
}
}  // namespace gsvprog
