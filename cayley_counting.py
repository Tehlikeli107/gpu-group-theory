"""
Cayley Graph Counting Invariants for Group Classification
==========================================================
The Thompson Conjecture (element order histograms) fails at order 16:
  Z8xZ2 ~ M16 (same order histogram, NOT isomorphic)
  Z4xZ4 ~ Q8xZ2 ~ Z4xZ4 (collisions in element orders)

Question: Do Cayley graph K4/K5 counting invariants resolve these?

A Cayley graph Cay(G, S) for a group G and generating set S has:
  - Vertices: elements of G
  - Edges: g ~ h iff g^{-1}h in S (S symmetric, identity-free)

For classification we want invariants independent of the generating set S.
Use the CANONICAL generating set: S = all non-identity elements (this gives
the complete graph, useless). Better: S = all involutions (order-2 elements).
Or: try all possible generating sets, take the multiset of K4 counts.

Actually, the BEST approach: use the CAYLEY GRAPH POLYNOMIAL — compute
K4 count for EVERY possible generating set S, return the multiset (histogram).
This is a complete group invariant if K4 counts vary with S.

For order 16: 15 non-identity elements, 2^15 possible generating sets. Too many.
Better: use S = {all elements of order d} for each divisor d of |G|.

This gives |divisors(n)|-1 Cayley graphs per group (one per non-trivial order).
Compute K4/K5/K3 counts on each. Much richer invariant than element order histogram alone.

GPU: batch all group elements across all order-16 groups simultaneously.
"""

import torch
import numpy as np
from itertools import combinations
from collections import Counter
import time

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ============================================================
# Group constructors (same as group_counting_revolution.py)
# ============================================================

def make_cyclic(n):
    return np.array([[(a+b)%n for b in range(n)] for a in range(n)], dtype=np.int32)

def make_direct_product(T1, T2):
    n1, n2 = len(T1), len(T2)
    n = n1*n2
    T = np.zeros((n,n), dtype=np.int32)
    for a in range(n):
        for b in range(n):
            a1,a2 = divmod(a,n2); b1,b2 = divmod(b,n2)
            T[a,b] = T1[a1,b1]*n2 + T2[a2,b2]
    return T

def make_dihedral(n):
    N=2*n; T=np.zeros((N,N),dtype=np.int32)
    for a in range(N):
        for b in range(N):
            ar=a%n; as_=a//n; br=b%n; bs=b//n
            if as_==0 and bs==0: T[a,b]=(ar+br)%n
            elif as_==0 and bs==1: T[a,b]=n+(br-ar)%n
            elif as_==1 and bs==0: T[a,b]=n+(ar+br)%n
            else: T[a,b]=(br-ar)%n
    return T

def make_quaternion():
    # Q8 = {1,-1,i,-i,j,-j,k,-k}, encoded 0..7
    # Multiplication table for Q8
    Q = [
        [0,1,2,3,4,5,6,7],
        [1,0,3,2,5,4,7,6],
        [2,3,1,0,6,7,4,5],
        [3,2,0,1,7,6,5,4],
        [4,5,7,6,0,1,3,2],
        [5,4,6,7,1,0,2,3],
        [6,7,5,4,2,3,1,0],
        [7,6,4,5,3,2,0,1],
    ]
    return np.array(Q, dtype=np.int32)

def make_modular16():
    """M16 = Z8 semidirect Z2, where Z2 acts on Z8 by x -> x+4.
    Elements: (a,b), a in Z8, b in Z2, coded as a + 8*b.
    Multiplication: (a,b)*(c,d) = (a + c*(-1)^b, b+d) but standard M16:
    M16 = <a,b | a^8 = b^2 = 1, bab = a^5>
    """
    # Brute force: compute from generators
    n = 16
    # a = generator of order 8, b = generator of order 2
    # Represent elements as words; use GAP-compatible multiplication
    # M16: elements indexed 0..15, multiplication from presentation
    # Let a=0..7 rotations (a^k = k), b=8..15 (b*a^k = 8+k)
    T = np.zeros((n, n), dtype=np.int32)
    for x in range(n):
        for y in range(n):
            # x = a^i * b^j (j=0 or 1), i in 0..7
            xi = x % 8; xj = x // 8
            yi = y % 8; yj = y // 8
            # (a^xi b^xj)(a^yi b^yj) = a^xi (b^xj a^yi b^xj) b^yj
            # b a^yi b^{-1} = a^{5*yi mod 8}  (since bab^{-1} = a^5)
            # b^xj a^yi b^{-xj} = a^{5^xj * yi mod 8}
            factor = (5 ** xj) % 8
            new_i = (xi + factor * yi) % 8
            new_j = (xj + yj) % 2
            T[x, y] = new_i + 8 * new_j
    # Verify it's a group
    return T

def make_z4xz4():
    """Z4 x Z4, elements (a,b) with a,b in Z4, coded as a+4*b."""
    T = np.zeros((16,16), dtype=np.int32)
    for x in range(16):
        for y in range(16):
            x1,x2 = x%4,x//4; y1,y2 = y%4,y//4
            T[x,y] = ((x1+y1)%4) + 4*((x2+y2)%4)
    return T

def make_z4xz2xz2():
    """Z4 x Z2 x Z2."""
    T4 = make_cyclic(4); T2 = make_cyclic(2)
    return make_direct_product(make_direct_product(T4, T2), T2)

def make_q8xz2():
    """Q8 x Z2."""
    T2 = make_cyclic(2)
    return make_direct_product(make_quaternion(), T2)

def make_z4semidirz4():
    """Z4 semidirect Z4 where action is by inversion: <a,b | a^4=b^4=1, bab^-1=a^-1>"""
    T = np.zeros((16,16), dtype=np.int32)
    for x in range(16):
        for y in range(16):
            xi = x%4; xj = x//4
            yi = y%4; yj = y//4
            # (a^xi b^xj)(a^yi b^yj) = a^xi b^xj a^yi b^{-xj} b^{xj+yj}
            # b^xj a^yi b^{-xj} = a^{(-1)^xj * yi mod 4}
            factor = (-1)**xj
            new_i = (xi + factor * yi) % 4
            new_j = (xj + yj) % 4
            T[x,y] = new_i + 4*new_j
    return T


# ============================================================
# Core invariants
# ============================================================

def element_orders(T):
    """Compute order of each element."""
    n = len(T)
    identity = None
    for i in range(n):
        if np.array_equal(T[i], np.arange(n)):
            identity = i; break
    if identity is None:
        identity = 0

    orders = []
    for g in range(n):
        x = g; order = 1
        while x != identity:
            x = T[x, g]; order += 1
            if order > n: break
        orders.append(order)
    return orders, identity


def build_cayley_graph(T, generating_set, identity):
    """Build adjacency matrix of Cayley graph Cay(G, S)."""
    n = len(T)
    # Inverse lookup
    inv = [None] * n
    for g in range(n):
        for h in range(n):
            if T[g, h] == identity:
                inv[g] = h; break

    A = np.zeros((n, n), dtype=np.int8)
    for g in range(n):
        for s in generating_set:
            h = T[g, s]  # g * s
            A[g, h] = 1
    # Also add edges for s^{-1}: ensure symmetry
    S_inv = [inv[s] for s in generating_set if inv[s] is not None]
    for g in range(n):
        for s in S_inv:
            h = T[g, s]
            A[g, h] = 1
    np.fill_diagonal(A, 0)
    return A


def count_cliques_fast(A, k):
    """Count k-cliques via brute force (small n)."""
    n = len(A)
    count = 0
    for verts in combinations(range(n), k):
        if all(A[verts[i], verts[j]] == 1
               for i in range(k) for j in range(i+1, k)):
            count += 1
    return count


def cayley_invariants(T, gen_set_name, gen_set):
    """Compute K3, K4 counts for a Cayley graph."""
    n = len(T)
    _, identity = element_orders(T)
    A = build_cayley_graph(T, gen_set, identity)
    degree = A.sum(0).mean()
    k3 = count_cliques_fast(A, 3)
    k4 = count_cliques_fast(A, 4)
    return {
        'gen_set': gen_set_name,
        'degree': float(degree),
        'K3': k3,
        'K4': k4,
    }


# ============================================================
# Build all order-16 groups
# ============================================================

def build_order16_groups():
    """Build all 14 non-isomorphic groups of order 16."""
    T2 = make_cyclic(2)
    T4 = make_cyclic(4)
    T8 = make_cyclic(8)
    T16 = make_cyclic(16)

    groups = {
        'Z16':         T16,
        'Z8xZ2':       make_direct_product(T8, T2),
        'Z4xZ4':       make_z4xz4(),
        'Z4xZ2xZ2':    make_z4xz2xz2(),
        'Z2^4':        make_direct_product(make_direct_product(T2,T2), make_direct_product(T2,T2)),
        'D8':          make_dihedral(8),
        'Q16':         None,   # Generalized quaternion, skip for now
        'QD16':        None,   # Quasidihedral, skip
        'M16':         make_modular16(),
        'D4xZ2':       make_direct_product(make_dihedral(4), T2),
        'Q8xZ2':       make_q8xz2(),
        'Z4semidirZ4': make_z4semidirz4(),
        # Two more: modular max, etc.
    }
    # Filter None
    return {k: v for k, v in groups.items() if v is not None}


# ============================================================
# Main analysis
# ============================================================

if __name__ == "__main__":
    print("="*65)
    print("CAYLEY GRAPH COUNTING INVARIANTS FOR ORDER-16 GROUPS")
    print("="*65)
    print()

    groups = build_order16_groups()
    print(f"Groups loaded: {len(groups)}")
    print(f"Groups: {list(groups.keys())}")
    print()

    # First: check Thompson Conjecture (element order histograms)
    print("--- ELEMENT ORDER HISTOGRAMS (Thompson Invariant) ---")
    order_sigs = {}
    for name, T in groups.items():
        orders, _ = element_orders(T)
        hist = dict(sorted(Counter(orders).items()))
        order_sigs[name] = str(hist)
        print(f"  {name:<20}: {hist}")

    # Find collisions
    hist_to_groups = {}
    for name, sig in order_sigs.items():
        hist_to_groups.setdefault(sig, []).append(name)
    print("\nThompson collisions (same order histogram, may be non-isomorphic):")
    for sig, names in hist_to_groups.items():
        if len(names) > 1:
            print(f"  {names} -> {sig}")

    print()
    print("--- CAYLEY GRAPH COUNTING INVARIANTS ---")
    print("(Using involution generating set: S = all elements of order 2)")
    print()

    cayley_sigs = {}
    for name, T in groups.items():
        orders, identity = element_orders(T)
        n = len(T)

        # Involution generating set (elements of order 2)
        involutions = [g for g in range(n) if g != identity and orders[g] == 2]

        t0 = time.time()
        inv_result = cayley_invariants(T, 'involutions', involutions)
        t1 = time.time()

        # Also try all-elements generating set (complete graph less identity)
        # For small n, this might be uniform across groups
        # Try order-4 elements instead
        order4 = [g for g in range(n) if g != identity and orders[g] == 4]
        order4_result = cayley_invariants(T, 'order4', order4) if order4 else None

        sig = (inv_result['K3'], inv_result['K4'])
        cayley_sigs[name] = sig

        print(f"  {name:<20}: invol K3={inv_result['K3']:4d}, K4={inv_result['K4']:4d}", end="")
        if order4_result:
            print(f" | order4 K3={order4_result['K3']:4d}, K4={order4_result['K4']:4d}", end="")
        print(f"  ({(t1-t0):.1f}s)")

    print()
    print("--- COLLISION RESOLUTION ---")
    print("\nGroups colliding in Thompson invariant:")
    for sig, names in hist_to_groups.items():
        if len(names) > 1:
            cay_sigs = {n: cayley_sigs[n] for n in names}
            all_distinct = len(set(str(s) for s in cay_sigs.values())) == len(names)
            status = "RESOLVED" if all_distinct else "STILL COLLIDING"
            print(f"\n  Thompson collision: {names}")
            for nm, cs in cay_sigs.items():
                print(f"    {nm:<20}: Cayley(involutions) K4={cs[1]}")
            print(f"  -> {status} by Cayley K4 invariant!")

    print()
    print("="*65)
    print("SUMMARY: CAYLEY COUNTING REVOLUTION")
    print("="*65)
    all_cayley = [str(v) for v in cayley_sigs.values()]
    n_distinct = len(set(all_cayley))
    n_total = len(groups)
    print(f"  Groups tested: {n_total}")
    print(f"  Cayley(involutions) K4 distinct signatures: {n_distinct}")
    print(f"  Coverage: {n_distinct}/{n_total} = {100*n_distinct/n_total:.1f}%")

    print()
    print("KEY INSIGHT:")
    print("  Thompson Conjecture uses element order histograms (algebraic invariant).")
    print("  Cayley graph K4 counts are TOPOLOGICAL invariants of group structure.")
    print("  When Thompson fails, do Cayley invariants succeed?")
    print("  (If yes: combining algebraic + topological counting = complete invariant)")
