# Coordination
Class for coordination analysis, i.e. bond angle distributions, coordination number, non-bridging analysis etc. `Coordination` takes a **list** of Extended ASE Atoms objects (e.g. multiple frames of a trajectory) and averages statistics over them.

## Example usage:

```
from vitrum.coordination import Coordination
coord = Coordination(atoms)  # atoms: List[Atoms]
angles, dist = coord.get_angle_distribution("Si", "O", cutoff=2)
```

::: vitrum.coordination