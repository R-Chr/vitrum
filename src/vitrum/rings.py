import warnings
from collections import Counter

import numpy as np
from ase import Atoms
from ase.data import covalent_radii
from ase.geometry import find_mic
from ase.io import write
from ase.neighborlist import NeighborList
from ase.symbols import symbols2numbers
from scipy.sparse import csr_array
from scipy.sparse.csgraph import dijkstra

from vitrum.glass_atoms import GlassAtoms


def check_ring_is_periodic(ring: list[int], offsets: dict[tuple[int, int], np.ndarray]) -> bool:
    ''' 
    Check if the ring wraps around the period cell, i.e., is not a true ring.
    
    Args:
        ring (List[int]): Atom indices of the ring.
        offsets (Dict[Tuple[int, int], np.ndarray]): Unit cell offsets for all atoms pairs.

    Returns:
        bool: True if the ring is periodic (wraps around), False otherwise (contained within cell without wrapping sum).
    '''
    total_offset = np.zeros(3)
    for i in range(len(ring) - 1):
        total_offset += offsets[(ring[i], ring[i+1])]
    total_offset += offsets[(ring[-1], ring[0])]
    return np.all(total_offset == 0)

_MAX_DEGENERATE_PATHS = 4096
_MAX_SEARCH_STEPS = 200_000


def _adjacency_lists(d: csr_array) -> list[np.ndarray]:
    '''
    Neighbour index array for each atom, extracted once from the bond graph.

    Args:
        d (csr_array): The sparse bond graph.

    Returns:
        List[np.ndarray]: One array of neighbour indices per atom.
    '''
    return [d.indices[d.indptr[i]:d.indptr[i + 1]] for i in range(d.shape[0])]


def _blocked(
    u: int,
    v: int,
    banned_nodes: frozenset[int],
    banned_edge: tuple[int, int] | None,
) -> bool:
    '''Whether the step from atom `u` to atom `v` is removed from the search graph.'''
    if u in banned_nodes or v in banned_nodes:
        return True
    if banned_edge is None:
        return False
    a, b = banned_edge
    return (u == a and v == b) or (u == b and v == a)


def _bfs_levels(
    adj: list[np.ndarray],
    source: int,
    max_depth: float,
    banned_nodes: frozenset[int] = frozenset(),
    banned_edge: tuple[int, int] | None = None,
) -> dict[int, int]:
    '''
    Hop distance from `source` to every atom within `max_depth` hops.

    Args:
        adj (List[np.ndarray]): Adjacency lists, from `_adjacency_lists`.
        source (int): Atom to search from.
        max_depth (float): Maximum number of hops; `np.inf` for no limit.
        banned_nodes (FrozenSet[int]): Atoms removed from the graph.
        banned_edge (Optional[Tuple[int, int]]): An edge removed from the graph, both directions.

    Returns:
        Dict[int, int]: Atom index -> hop distance, for the atoms reached.
    '''
    dist = {source: 0}
    frontier = [source]
    depth = 0
    while frontier and depth < max_depth:
        depth += 1
        nxt = []
        for u in frontier:
            for v in adj[u]:
                v = int(v)
                if v in dist or _blocked(u, v, banned_nodes, banned_edge):
                    continue
                dist[v] = depth
                nxt.append(v)
        frontier = nxt
    return dist


def _backtrack_paths(
    adj: list[np.ndarray],
    dist: dict[int, int],
    source: int,
    target: int,
    banned_nodes: frozenset[int] = frozenset(),
    banned_edge: tuple[int, int] | None = None,
) -> list[list[int]]:
    '''
    Every shortest path from `source` to `target`, as [source, ..., target].

    A single predecessor chain (what `dijkstra(return_predecessors=True)` gives) keeps
    only one path per search, so when several equally-short paths close the same bond
    all but one ring is lost. Ring criteria are defined over *all* shortest paths, so
    this walks back from `target` through every neighbour that sits one layer closer.

    Args:
        adj (List[np.ndarray]): Adjacency lists, from `_adjacency_lists`.
        dist (Dict[int, int]): Hop distances from `source`, from `_bfs_levels`.
        source (int): Atom the distances were measured from.
        target (int): Atom to walk back from.
        banned_nodes (FrozenSet[int]): Atoms removed from the graph.
        banned_edge (Optional[Tuple[int, int]]): An edge removed from the graph, both directions.

    Returns:
        List[List[int]]: One list per shortest path, each ordered [source, ..., target].
            Empty if `target` was not reached.
    '''
    if target not in dist:
        return []
    paths = [[target]]
    for _ in range(dist[target]):
        grown = []
        for path in paths:
            tip = path[-1]
            for u in adj[tip]:
                u = int(u)
                if dist.get(u, -1) == dist[tip] - 1 and not _blocked(u, tip, banned_nodes, banned_edge):
                    grown.append(path + [u])
        paths = grown[:_MAX_DEGENERATE_PATHS]
    return [path[::-1] for path in paths]


def _paths_of_length(
    adj: list[np.ndarray],
    source: int,
    target: int,
    length: int,
    dist_to_target: dict[int, int],
    banned_nodes: frozenset[int],
    banned_edge: tuple[int, int] | None,
) -> tuple[list[list[int]], bool]:
    '''
    Simple paths from `source` to `target` with exactly `length` bonds.

    `dist_to_target` prunes the walk: from an atom that is further from `target` than the
    remaining budget there is no way to arrive on time. At `length` equal to the hop
    distance this degenerates to enumerating the shortest paths and nothing more.

    Args:
        adj (List[np.ndarray]): Adjacency lists, from `_adjacency_lists`.
        source (int): Atom to start from.
        target (int): Atom to end on.
        length (int): Exact number of bonds the path must have.
        dist_to_target (Dict[int, int]): Hop distances to `target`, from `_bfs_levels`.
        banned_nodes (FrozenSet[int]): Atoms removed from the graph.
        banned_edge (Optional[Tuple[int, int]]): An edge removed from the graph, both directions.

    Returns:
        Tuple[List[List[int]], bool]: The paths, each ordered [source, ..., target], and
            whether the search hit its step or path-count ceiling and so may be incomplete.
    '''
    results: list[list[int]] = []
    path = [source]
    seen = {source}
    steps = 0
    capped = False

    def walk(u: int, remaining: int) -> None:
        nonlocal steps, capped
        if remaining == 0:
            if u == target:
                results.append(path.copy())
            return
        for v in adj[u]:
            if capped:
                return
            v = int(v)
            if v in seen or _blocked(u, v, banned_nodes, banned_edge):
                continue
            # Reaching the target early strands the walk: it can never come back.
            if v == target and remaining > 1:
                continue
            if dist_to_target.get(v, length + 1) > remaining - 1:
                continue
            steps += 1
            if steps > _MAX_SEARCH_STEPS or len(results) >= _MAX_DEGENERATE_PATHS:
                capped = True
                return
            seen.add(v)
            path.append(v)
            walk(v, remaining - 1)
            path.pop()
            seen.discard(v)

    walk(source, length)
    return results, capped


def _shortest_valid_rings(
    adj: list[np.ndarray],
    offsets: dict[tuple[int, int], np.ndarray],
    source: int,
    target: int,
    tail: list[int],
    hop_limit: float,
    banned_edge: tuple[int, int] | None = None,
    banned_nodes: frozenset[int] = frozenset(),
) -> tuple[list[list[int]], dict[str, int]]:
    '''
    The shortest rings closing `source` to `target` that do not wrap around the cell.
    A closed path whose bond vectors do not sum to zero winds around the periodic cell and
    is not a ring (see `check_ring_is_periodic`).

    Args:
        adj (List[np.ndarray]): Adjacency lists, from `_adjacency_lists`.
        offsets (Dict[Tuple[int, int], np.ndarray]): Unit cell offsets for all atom pairs.
        source (int): Atom the path starts from.
        target (int): Atom the path ends on.
        tail (List[int]): Atoms appended to the path to close the ring; empty for Guttman,
            the central atom for King.
        hop_limit (float): Maximum number of bonds in the path; `np.inf` for no limit.
        banned_edge (Optional[Tuple[int, int]]): An edge removed from the graph, both directions.
        banned_nodes (FrozenSet[int]): Atoms removed from the graph.

    Returns:
        Tuple[List[List[int]], Dict[str, int]]: The rings, and flags counting whether the
            search had to look past a wrapping candidate ("deepened"), hit its ceiling on
            enumerated paths ("truncated"), and gave up without a valid ring ("exhausted").
    '''
    flags = {"deepened": 0, "truncated": 0, "exhausted": 0}

    dist_back = _bfs_levels(adj, target, hop_limit, banned_nodes, banned_edge)
    shortest = dist_back.get(source)
    if shortest is None:
        return [], flags

    longest = hop_limit if np.isfinite(hop_limit) else len(adj) - 1
    longest = int(min(longest, len(adj) - 1))

    for length in range(shortest, longest + 1):
        paths, capped = _paths_of_length(
            adj, source, target, length, dist_back, banned_nodes, banned_edge
        )
        flags["deepened"] = int(length > shortest)
        flags["truncated"] = int(capped)
        rings = [path + tail for path in paths]
        valid = [ring for ring in rings if check_ring_is_periodic(ring, offsets)]
        if valid:
            return valid, flags
        if capped:
            flags["exhausted"] = 1
            return [], flags

    flags["deepened"] = int(longest > shortest)
    flags["exhausted"] = 1
    return [], flags


def _find_guttman_rings(
    len_ats: int,
    d: csr_array,
    offsets: dict[tuple[int, int], np.ndarray],
    limit: float,
) -> tuple[list[list[int]], dict[str, int]]:
    '''
    Find Guttman rings, in the indices of the (possibly repeated) search cell.

    For each bond (i, j), remove that edge and find the shortest paths between i and j in
    the remaining graph. Each path plus the removed edge forms a ring. Every shortest path
    is kept, not just one.
    '''
    adj = _adjacency_lists(d)
    hop_limit = limit - 1 if np.isfinite(limit) else np.inf
    rings: list[list[int]] = []
    stats = {"deepened": 0, "truncated": 0, "exhausted": 0}
    for i in range(len_ats):
        for j in adj[i]:
            j = int(j)
            if j == i:
                continue
            found, flags = _shortest_valid_rings(
                adj, offsets, i, j, [], hop_limit, banned_edge=(i, j)
            )
            rings.extend(found)
            for key, value in flags.items():
                stats[key] += value
    return rings, stats


def _find_king_rings(
    len_ats: int,
    d: csr_array,
    offsets: dict[tuple[int, int], np.ndarray],
    limit: float,
) -> tuple[list[list[int]], dict[str, int]]:
    '''
    Find King's rings, in the indices of the (possibly repeated) search cell.

    For each atom c and each pair of its bonded neighbors (n1, n2), remove c from the graph
    and find the shortest paths between n1 and n2 in the remaining graph. Each path plus the
    two edges to c forms a ring. As for the Guttman criterion, every shortest path is kept
    rather than a single predecessor chain.
    '''
    adj = _adjacency_lists(d)
    hop_limit = limit - 2 if np.isfinite(limit) else np.inf
    rings: list[list[int]] = []
    stats = {"deepened": 0, "truncated": 0, "exhausted": 0}
    for c in range(len_ats):
        shell = {int(n) for n in adj[c]} - {c}
        neighbors = sorted(shell)
        if len(neighbors) < 2:
            continue
        for p, n1 in enumerate(neighbors):
            for n2 in neighbors[p + 1:]:
                banned = {c} | (shell - {n1, n2})
                found, flags = _shortest_valid_rings(
                    adj, offsets, n1, n2, [c], hop_limit, banned_nodes=frozenset(banned)
                )
                rings.extend(found)
                for key, value in flags.items():
                    stats[key] += value
    return rings, stats


def _is_shortcut_free(ring: list[int], d: csr_array) -> bool:
    '''
    Whether a ring cannot be decomposed into two smaller ones (Franzblau's criterion).

    For every pair of atoms in the ring, the shortest path between them in the full bond
    graph must equal the geodesic distance between them measured along the ring. Any pair
    connected by a shorter path (a "shortcut") means the ring decomposes.
    '''
    n = len(ring)
    # A shortcut can only matter if it's shorter than the longest possible ring-arc
    # distance (n // 2), so the search never needs to look further than that.
    dist_matrix = dijkstra(d, indices=ring, directed=False, unweighted=True, limit=n // 2)
    for p in range(n):
        for q in range(p + 1, n):
            ring_dist = min(q - p, n - (q - p))
            if dist_matrix[p, ring[q]] < ring_dist:
                return False
    return True


def _find_primitive_rings(
    len_ats: int,
    d: csr_array,
    offsets: dict[tuple[int, int], np.ndarray],
    limit: float,
) -> tuple[list[list[int]], dict[str, int]]:
    '''
    Find primitive rings, in the indices of the (possibly repeated) search cell.
    From each root atom, take every pair of equally long shortest
    paths that share no interior atom. If the two paths end on the same atom they close an
    even ring, and if they end on bonded atoms they close an odd one. Candidates are then
    kept only if they are shortcut-free.
    '''
    adj = _adjacency_lists(d)
    adj_sets = [{int(x) for x in a} for a in adj]
    max_size = int(limit) if np.isfinite(limit) else d.shape[0]
    kmax = max_size // 2

    stats = {"deepened": 0, "truncated": 0, "exhausted": 0, "wrapped": 0}
    candidates: dict[tuple[int, ...], list[int]] = {}
    for root in range(len_ats):
        dist = _bfs_levels(adj, root, kmax)
        by_depth: dict[int, list[int]] = {}
        for node, hops in dist.items():
            if hops:
                by_depth.setdefault(hops, []).append(node)
        for k in range(1, kmax + 1):
            paths: list[list[int]] = []
            for target in by_depth.get(k, ()):
                to_target = _backtrack_paths(adj, dist, root, target)
                stats["truncated"] += len(to_target) >= _MAX_DEGENERATE_PATHS
                paths.extend(to_target)
            # Interior atoms only: the root is shared by construction, and the endpoints
            # sit at a different depth from every interior atom so they cannot collide.
            interiors = [frozenset(path[1:k]) for path in paths]
            for ia, first in enumerate(paths):
                for ib in range(ia + 1, len(paths)):
                    if interiors[ia] & interiors[ib]:
                        continue
                    second = paths[ib]
                    end_a, end_b = first[-1], second[-1]
                    if end_a == end_b:
                        ring = first + second[1:-1][::-1]
                    elif end_b in adj_sets[end_a]:
                        ring = first + second[1:][::-1]
                    else:
                        continue
                    if 3 <= len(ring) <= max_size:
                        candidates.setdefault(tuple(sorted(ring)), ring)

    rings: list[list[int]] = []
    for ring in candidates.values():
        if not _is_shortcut_free(ring, d):
            continue
        if not check_ring_is_periodic(ring, offsets):
            stats["wrapped"] += 1
            continue
        rings.append(ring)
    return rings, stats


def find_rings(
    ats: Atoms,
    radii_factor: float = 1.3,
    repeat: tuple[int, int, int] | None = None,
    bonds: list[tuple[str, str]] | None = None,
    limit: float = np.inf,
    criterion: str = "guttman",
) -> list[list[int]]:
    '''
    Find rings in the unit cell.

    Three ring criteria are supported via `criterion`, each following the corresponding
    routine of the R.I.N.G.S. code (S. Le Roux and P. Jund, Comput. Mater. Sci. 2010, 49, 70-83):

    - "guttman": L. Guttman, J. Non-Cryst. Solids 1990, 116. For each bond (i, j), remove that
      edge and find the shortest path between i and j in the remaining graph.
    - "king": S. V. King, Nature 1967, 213, 425. For each atom c and each pair of its bonded
      neighbors (n1, n2), remove c from the graph and find the shortest path between n1 and n2
      in the remaining graph.
    - "primitive": D. S. Franzblau, Phys. Rev. B 1991, 44, 4925. A ring that cannot be
      decomposed into two smaller rings, i.e. one with no "shortcut": no pair of ring atoms is
      connected, in the full bond graph, by a path shorter than the corresponding arc of the
      ring. Enumerated from pairs of equally long shortest paths out of each atom.

    When the shortest ring closing a bond winds around the periodic cell it is not a ring, and
    the search carries on to the next length rather than returning nothing for that bond.

    Args:
        ats (ase.Atoms): Atoms object containing the structure
        radii_factor (float): Factor to multiply covalent radii for neighbor search
        repeat (Optional[Tuple[int, int, int]]): How often to repeat the unit cell in each
            direction. Increase for small cells. Defaults to (1, 1, 1), except for
            `criterion="primitive"` on a periodic cell, where it defaults to (3, 3, 3)
            because a single cell cannot represent every primitive ring — the same
            replication R.I.N.G.S. applies for this criterion.
        bonds (Optional[List[Tuple[str, str]]]): List of allowed bonds, e.g., [('C', 'C'), ('C', 'O')], can be None to allow all bonds.
            This filters which bonds enter the graph; it is not R.I.N.G.S.'s ABAB option,
            which constrains the ring itself to alternate between two species.
        limit (float): Maximum ring size (number of atoms) to search for. Rings larger than
            this are not returned, for every criterion. Defaults to no limit; setting it is
            strongly recommended for `criterion="primitive"`.
        criterion (str): Ring criterion to use, one of "guttman", "king", or "primitive".

    Returns:
        List[List[int]]: A list of rings, where each ring is a list of atom indices.

    Raises:
        ValueError: If `criterion` is unknown, or `limit` is a finite value below 3.
    '''
    if criterion not in ("guttman", "king", "primitive"):
        raise ValueError(f"Unknown ring criterion '{criterion}', expected 'guttman', 'king', or 'primitive'.")
    if np.isfinite(limit) and limit < 3:
        raise ValueError(f"limit is a maximum ring size and must be at least 3, got {limit}.")

    if repeat is None:
        repeat = (3, 3, 3) if criterion == "primitive" and np.any(ats.get_pbc()) else (1, 1, 1)

    s = ats.repeat(repeat)
    pos = s.get_positions()
    nat = len(s)
    lat = s.get_cell()
    els = s.get_chemical_symbols()
    radii = covalent_radii[symbols2numbers(els)]

    if bonds is not None:
        # Don't need to find neighbors for elements not included in bonds
        elements = set().union(*bonds)
        radii = [x if el in elements else 0. for el, x in zip(els, radii)]
        radii = np.array(radii, dtype=float)
           
    nl = NeighborList(radii * radii_factor, self_interaction=False, bothways=False, skin=0.)
    nl.update(s)

    d_idx = []
    d_val = []
    # unit cell offsets for all atoms
    all_offsets = {}

    n_ambiguous = 0

    for i in range(nat):
        indices, offsets = nl.get_neighbors(i)

        rs = pos[indices, :] + offsets @ lat - pos[i, :]
        ds = np.linalg.norm(rs, axis=1)
        for j, r, o in zip(indices, ds, offsets):
            j = int(j)
            # Ignore bonds that are not included; bonds=None allows all bonds
            if bonds is None or (els[i], els[j]) in bonds or (els[j], els[i]) in bonds:
                # A bond to a periodic image of the atom itself, or a second image of a pair
                # already bonded, cannot be represented: the graph holds one edge per pair.
                if i == j or (i, j) in all_offsets:
                    n_ambiguous += 1
                    continue
                d_idx.append((i, j))
                d_val.append(r)
                d_idx.append((j, i))
                d_val.append(r)
                all_offsets[(i, j)] = o
                all_offsets[(j, i)] = -o

    if n_ambiguous:
        warnings.warn(
            f'{n_ambiguous} bond(s) connect the same pair of atoms through more than one '
            'periodic image, so the ring graph cannot represent them unambiguously. The '
            'cell is too small for this bond cutoff — increase `repeat`.'
        )

    if not d_idx:
        return []

    # sparse matrix of bonds, removes zero entries
    d = csr_array((d_val, np.array(d_idx, dtype=np.int32).T), shape=(nat, nat))

    if criterion == "guttman":
        raw_rings, stats = _find_guttman_rings(len(ats), d, all_offsets, limit)
    elif criterion == "king":
        raw_rings, stats = _find_king_rings(len(ats), d, all_offsets, limit)
    else:
        raw_rings, stats = _find_primitive_rings(len(ats), d, all_offsets, limit)

    rings = {}
    for ring in raw_rings:
        ring = [x % len(ats) for x in ring]  # take it back to primary cell
        rings[tuple(sorted(ring))] = ring

    if stats["deepened"]:
        warnings.warn(
            f'{stats["deepened"]} ring search(es) had to look past a candidate that wraps '
            'around the periodic cell, and returned a larger ring than the bond graph alone '
            'suggests. The result is correct, but a larger `repeat` avoids the ambiguity.'
        )
    if stats["exhausted"]:
        warnings.warn(
            f'{stats["exhausted"]} ring search(es) found no ring that stays inside the '
            'periodic cell and returned nothing. The cell is too small for this ring size — '
            'increase `repeat`.'
        )
    if stats["truncated"]:
        warnings.warn(
            f'{stats["truncated"]} ring search(es) hit the ceiling of '
            f'{_MAX_DEGENERATE_PATHS} equally short paths and may have missed rings. '
            'Lower `limit` to bound the search.'
        )
    if stats.get("wrapped"):
        warnings.warn(
            f'{stats["wrapped"]} primitive ring candidate(s) wrap around the periodic cell '
            'and were discarded. Consider increasing `repeat`.'
        )

    return list(rings.values())


def _fit_ellipse_axes(x: np.ndarray, y: np.ndarray) -> tuple[float, float] | None:
    '''
    Semi-major and semi-minor axes of the least-squares best-fit ellipse of 2D points.

    Returns None when the points do not determine an ellipse: fewer than the five needed to
    fix a conic, or a degenerate configuration (collinear points, or a fit that comes out
    parabolic or hyperbolic).
    '''
    if len(x) < 5:
        return None

    # The conic fit is badly conditioned on raw Angstrom coordinates; work on points scaled
    # to unit RMS radius and scale the axes back at the end.
    scale = float(np.sqrt(np.mean(x ** 2 + y ** 2)))
    if not np.isfinite(scale) or scale == 0.0:
        return None
    x, y = x / scale, y / scale

    # Split the design matrix into its quadratic and linear parts, so that the constrained
    # eigenproblem is 3x3 on the quadratic coefficients alone.
    d1 = np.column_stack((x * x, x * y, y * y))
    d2 = np.column_stack((x, y, np.ones_like(x)))
    s1, s2, s3 = d1.T @ d1, d1.T @ d2, d2.T @ d2
    try:
        t = -np.linalg.solve(s3, s2.T)
    except np.linalg.LinAlgError:
        return None
    m = s1 + s2 @ t
    # Premultiply by the inverse of the ellipse constraint matrix [[0, 0, 2], [0, -1, 0], [2, 0, 0]].
    m = np.array((m[2] / 2.0, -m[1], m[0] / 2.0))

    eigenvectors = np.linalg.eig(m)[1]
    # The ellipse solution is the eigenvector satisfying 4AC - B^2 > 0.
    valid = np.flatnonzero(4.0 * eigenvectors[0] * eigenvectors[2] - eigenvectors[1] ** 2 > 0)
    if valid.size == 0:
        return None
    quadratic = np.real(eigenvectors[:, valid[0]])
    a, b, c = quadratic
    dd, e, f = t @ quadratic

    # Shift the conic to its own center, where it reduces to a x'^2 + b x'y' + c y'^2 = -f',
    # then diagonalise the quadratic part: the semi-axes are sqrt(-f' / lambda) for the two
    # eigenvalues of [[a, b/2], [b/2, c]].
    discriminant = 4.0 * a * c - b * b
    if discriminant <= 0.0:  # not an ellipse
        return None
    x0 = (b * e - 2.0 * c * dd) / discriminant
    y0 = (b * dd - 2.0 * a * e) / discriminant
    constant = f + (dd * x0 + e * y0) / 2.0
    root = np.sqrt((a - c) ** 2 + b * b)
    axes = []
    for eigenvalue in (((a + c) - root) / 2.0, ((a + c) + root) / 2.0):
        squared_axis = -constant / eigenvalue
        if squared_axis <= 0.0:
            return None
        axes.append(np.sqrt(squared_axis) * scale)
    return float(max(axes)), float(min(axes))


class Ring:
    """
    A class representing a ring in a atomistic system.
    """

    def __init__(self, atoms: Atoms, indexes: list[int] | None = None):
        """
        Initialize a Ring object.

        Args:
            atoms (Atoms): An Atoms object representing the atoms in the ring.
            indexes (Optional[List[int]], optional): A list of indices of the atoms involved in the ring. Defaults to None.
        """
        self.atoms = atoms
        self.indexes = indexes
        self._unwrapped_positions_cache = None
        self._roundness = None
        self._roughness = None
        self.ellipsoid_lengths = None
        self._principal_axes = None
        self._ellipse_axes_cache = None
        self.atom_symbols = np.array(self.atoms.get_chemical_symbols())
        self.atom_types = np.unique(self.atom_symbols).tolist()
        self.atom_ids = {atom_type: np.where(self.atom_symbols == atom_type)[0] for atom_type in self.atom_types}

    def _unwrapped_positions(self) -> np.ndarray:
        """
        Ring atom positions with periodic-boundary jumps removed.

        Rings are only ever kept by `find_rings` if they don't wind around the periodic
        cell (see `check_ring_is_periodic`), so every atom has a single well-defined
        position relative to the first ring atom: its minimum-image displacement from it.

        Returns:
            np.ndarray: An array of shape (size(), 3) of PBC-unwrapped Cartesian positions.
        """
        if self._unwrapped_positions_cache is None:
            positions = self.atoms.get_positions()
            displacements, _ = find_mic(positions - positions[0], self.atoms.get_cell(), self.atoms.get_pbc())
            self._unwrapped_positions_cache = positions[0] + displacements
        return self._unwrapped_positions_cache

    def center(self) -> np.ndarray:
        """
        Calculate the center of the ring.

        Returns:
            np.ndarray: An array of shape (3,) representing the center of the ring, wrapped into the primary cell.
        """
        center = self._unwrapped_positions().mean(axis=0)
        cell = self.atoms.get_cell()
        return cell.cartesian_positions(cell.scaled_positions(center) % 1.0)

    def size(self) -> int:
        """
        Calculate the size of the ring, i.e., the number of atoms in the ring.

        Returns:
            int: The size of the ring.
        """
        return len(self.atoms)

    def perimeter(self) -> float:
        """
        Calculate the perimeter of the ring, i.e., the sum of the consecutive bond
        lengths around the ring (including the closing bond from the last atom back
        to the first).

        Returns:
            float: The perimeter of the ring.
        """
        xyz = self._unwrapped_positions()
        edges = np.diff(xyz, axis=0, append=xyz[:1])
        return float(np.linalg.norm(edges, axis=1).sum())

    def _compute_ellipsoid(self) -> None:
        """
        Fit the ring atoms with a best-fit ellipsoid via the SVD of their centered,
        PBC-unwrapped positions. The resulting singular values (descending) are the
        lengths of the ellipsoid's three principal axes and are the basis for
        `roundness`, `roughness`, and `radius_of_gyration`.

        The right singular vectors are kept as well: the first two span the ring's
        best-fit plane and the third is its normal, which `planeness` and
        `ellipse_eccentricity` need.
        """
        xyz = self._unwrapped_positions()
        xyz = xyz - xyz.mean(axis=0)
        _, singular_values, right_vectors = np.linalg.svd(xyz)
        self.ellipsoid_lengths = singular_values
        self._principal_axes = right_vectors

    def roundness(self) -> float:
        """
        Calculate the roundness of the ring: the ratio of the second-largest to the
        largest best-fit ellipsoid axis, i.e. how circular (close to 1) vs. elongated
        (close to 0) the ring is within its own best-fit plane.

        Returns:
            float: The roundness of the ring.
        """
        if self.ellipsoid_lengths is None:
            self._compute_ellipsoid()
        if self._roundness is None:
            self._roundness = self.ellipsoid_lengths[1] / self.ellipsoid_lengths[0]
        return self._roundness

    def roughness(self) -> float:
        """
        Calculate the roughness of the ring: the ratio of the smallest best-fit
        ellipsoid axis to the geometric mean of the other two, i.e. how far the ring
        deviates out of its own best-fit plane.

        Returns:
            float: The roughness of the ring.
        """
        if self.ellipsoid_lengths is None:
            self._compute_ellipsoid()
        if self._roughness is None:
            self._roughness = self.ellipsoid_lengths[2] / np.sqrt(self.ellipsoid_lengths[0] * self.ellipsoid_lengths[1])
        return self._roughness

    def radius_of_gyration(self) -> float:
        """
        Calculate the radius of gyration of the ring: the root-mean-square distance
        of the ring atoms from their centroid.

        Returns:
            float: The radius of gyration of the ring.
        """
        if self.ellipsoid_lengths is None:
            self._compute_ellipsoid()
        return float(np.sqrt(np.sum(self.ellipsoid_lengths ** 2) / self.size()))

    def area(self) -> float:
        """
        Calculate the center area of the ring: the ring is split into triangles between its
        center and each pair of adjacent atoms, and the triangle areas are summed. Each
        triangle contributes half the norm of the cross product of the two center-to-atom
        vectors that bound it.

        The triangles are summed unsigned, so a strongly non-convex or self-shadowing ring
        gives a larger area than the planar polygon it projects onto.

        Returns:
            float: The center area of the ring, in Å².
        """
        spokes = self._unwrapped_positions()
        spokes = spokes - spokes.mean(axis=0)
        triangles = np.cross(spokes, np.roll(spokes, -1, axis=0))
        return float(0.5 * np.linalg.norm(triangles, axis=1).sum())

    def eccentricity(self) -> float:
        """
        Calculate the eccentricity of the ring as sqrt(1 - b²/a²), where a and b are the
        distances from the ring center to its farthest and closest atom respectively.

        This is a radial measure taken in three dimensions and needs no plane fit, so unlike
        `ellipse_eccentricity` it is defined for rings of any size. It reads 0 for a ring
        whose atoms are all equidistant from the center and approaches 1 as the ring is
        drawn out.

        Returns:
            float: The eccentricity of the ring, between 0 and 1.
        """
        spokes = self._unwrapped_positions()
        radii = np.linalg.norm(spokes - spokes.mean(axis=0), axis=1)
        longest = radii.max()
        if longest == 0.0:
            return 0.0
        return float(np.sqrt(max(0.0, 1.0 - (radii.min() / longest) ** 2)))

    def planeness(self) -> float:
        """
        Calculate the planeness of the ring: the mean distance from its atoms to their
        best-fit plane, i.e. the plane residuals. A planar ring reads 0, and larger values
        mean a more puckered ring.

        Returns:
            float: The mean point-to-plane distance, in Å.
        """
        if self._principal_axes is None:
            self._compute_ellipsoid()
        spokes = self._unwrapped_positions()
        spokes = spokes - spokes.mean(axis=0)
        return float(np.abs(spokes @ self._principal_axes[2]).mean())

    def _ellipse_axes(self) -> tuple[float, float] | None:
        """
        Project the ring atoms onto their best-fit plane and fit an ellipse to the
        projection, returning its (semi-major, semi-minor) axes, or None if the points do
        not determine one.
        """
        if self._ellipse_axes_cache is None:
            if self._principal_axes is None:
                self._compute_ellipsoid()
            spokes = self._unwrapped_positions()
            spokes = spokes - spokes.mean(axis=0)
            in_plane = spokes @ self._principal_axes[:2].T
            self._ellipse_axes_cache = (_fit_ellipse_axes(in_plane[:, 0], in_plane[:, 1]),)
        return self._ellipse_axes_cache[0]

    def ellipse_eccentricity(self) -> float:
        """
        Calculate the eccentricity of the ring the second way: project the atoms onto their
        best-fit plane, fit an ellipse to the projection, and take sqrt(1 - b²/a²) from its
        semi-major axis a and semi-minor axis b.

        Unlike `eccentricity`, which uses the extreme center-to-atom distances directly,
        this uses every atom and is insensitive to a single outlying one. It needs at least
        five atoms, because that is how many points fix a conic, so it is not defined for
        3- and 4-rings.

        Returns:
            float: The eccentricity of the fitted ellipse, between 0 and 1, or NaN if the
                ring has fewer than five atoms or its projection does not determine an
                ellipse.
        """
        axes = self._ellipse_axes()
        if axes is None:
            return float("nan")
        semi_major, semi_minor = axes
        if semi_major == 0.0:
            return 0.0
        return float(np.sqrt(max(0.0, 1.0 - (semi_minor / semi_major) ** 2)))


class RingAnalysis:
    """
    A class for calculating and analyzing rings in atomistic systems.
    """

    def __init__(self, atoms: Atoms, included_atoms: list[str], bonding_dict: list[tuple[str, str]] | None = None):
        """
        Initialize the RingAnalysis class.

        Args:
            atoms (Atoms): An Atoms object representing the atoms in the system.
            included_atoms (List[str]): A list of strings representing the chemical symbols of the atoms to include in the analysis.
            bonding_dict (Optional[List[Tuple[str, str]]]): A list of allowed bonds, e.g., [('Si', 'O')].
        """
        super().__init__()
        self.bonding_dict = bonding_dict
        atoms = atoms[[atom.symbol in included_atoms for atom in atoms]]
        self.atoms = GlassAtoms(atoms)
        self.num_atoms = len(self.atoms)
        self.atom_symbols = np.array(self.atoms.get_chemical_symbols())
        self.atom_types = np.unique(self.atom_symbols).tolist()
        self.atom_ids = [np.where(self.atom_symbols == atom_type)[0] for atom_type in self.atom_types]
        self.rings = None


    def calculate(
        self,
        radii_factor: float = 1.3,
        repeat: tuple[int, int, int] | None = None,
        max_size: float = np.inf,
        criterion: str = "guttman",
    ) -> list[Ring]:
        """
        Calculate the rings in the system.

        Args:
            radii_factor (float): Factor to multiply covalent radii for neighbor search.
            repeat (Optional[Tuple[int, int, int]]): Repeat unit cell. Defaults to (1, 1, 1),
                or (3, 3, 3) for `criterion="primitive"` on a periodic cell.
            max_size (float): Maximum ring size, in number of atoms. Applies identically
                to every criterion. Defaults to no limit.
            criterion (str): Ring criterion to use, one of "guttman", "king", or "primitive".
                See `vitrum.rings.find_rings` for details on each criterion.

        Returns:
            List[Ring]: A list of Ring objects representing the rings in the system.
        """
        bonds = self.bonding_dict

        rings = find_rings(
            ats=self.atoms,
            radii_factor=radii_factor,
            repeat=repeat,
            bonds=bonds,
            limit=max_size,
            criterion=criterion,
        )

        self.rings = [Ring(self.atoms[list(r)], list(r)) for r in rings]
        return self.rings

    def write_rings(self, filename: str, format: str = 'extxyz'):
        """
        Write the rings to a file.

        Args:
            filename (str): The name of the file to write the rings to.
            format (str): The format of the file.
        """
        if self.rings is None:
            raise ValueError("Rings have not been calculated yet.")
        write(filename, [r.atoms for r in self.rings], format=format)
    
    def get_ring_size_distribution(self) -> dict[int, int]:
        """
        Get the distribution of ring sizes.

        Returns:
            Dict[int, int]: A dictionary where keys are ring sizes and values are counts.
        """
        if self.rings is None:
            raise ValueError("Rings have not been calculated yet.")
        ring_sizes = [len(r.atoms) for r in self.rings]
        return dict(Counter(ring_sizes))
    
    def plot_ring_size_distribution(self, ax=None, **plot_kwargs):
        """
        Plots the distribution of ring sizes using matplotlib.
        """
        import matplotlib.pyplot as plt
        
        dist = self.get_ring_size_distribution()
        if not dist:
            print("No rings found. Ensure you have run .calculate() first.")
            return

        sizes = sorted(dist.keys())
        counts = np.array([dist[size] for size in sizes])
        frequency = counts / self.atoms.get_volume()  # Normalize by volume to get frequency

        if ax is None:
            fig, ax = plt.subplots(figsize=(9, 6))
            ax.set_xlabel('Ring Size (N$_{atoms}$)', fontsize=12)
            ax.set_ylabel('Ring Frequency [N$_{rings}$ / V] (Å$^{-3}$)', fontsize=12)
            ax.set_xticks(sizes)  

        ax.plot(sizes, frequency, **plot_kwargs)

        return ax