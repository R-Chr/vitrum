# Coordination
Class for coordination analysis, i.e. bond angle distributions, coordination number, non-bridging analysis etc.

## Example usage:

```
from vitrum.coordination import coordination
coord = coordination(atoms)
angles, dist = coord.get_angle_distribution("Si", "O", cutoff=2)
```

::: vitrum.coordination