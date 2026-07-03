# Diffusion

The diffusion class requires a list of Atoms objects with unwrapped atom coordinates during diffusion and a list of sampled times.

Note: `get_van_hove_dist_correlation` and `get_velocity_autocorrelation` are currently unimplemented stubs (they do nothing and return `None`) — only `get_mean_square_displacements`, `get_diffusion_coef`, and `get_van_hove_self_correlation` are functional.

See [Quick start](quickstart.md) for a worked example.

::: vitrum.diffusion