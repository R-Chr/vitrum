"""Render ase.Atoms glass structures to static images or Jupyter widgets via OVITO."""

from typing import Dict, Optional, Sequence, Tuple, Union

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from ovito.data import DataCollection
from ovito.io.ase import ase_to_ovito
from ovito.modifiers import CreateBondsModifier
from ovito.pipeline import Pipeline as OvitoPipeline
from ovito.pipeline import StaticSource
from ovito.vis import TachyonRenderer, Viewport

_DEFAULT_COLOR_MAP = {"c_min": "#FF0000", "c_max": "#0000FF"}
_RENDERERS = {"tachyon": TachyonRenderer}


def hex_to_rgb(hex_color: str) -> Tuple[float, float, float]:
    """
    Convert a '#RRGGBB' (or 'RRGGBB') hex color string to an (r, g, b) tuple.

    Args:
        hex_color (str): A hex color string, e.g. "#FF0000".

    Returns:
        Tuple[float, float, float]: The red, green and blue channels, each in [0, 1].

    Raises:
        ValueError: If the string is not exactly 6 hex digits after stripping '#'.
    """
    digits = hex_color.lstrip("#")
    if len(digits) != 6:
        raise ValueError(f"Expected a '#RRGGBB' hex color, got {hex_color!r}.")
    return tuple(int(digits[i:i + 2], 16) / 255 for i in (0, 2, 4))


def _colors_by_element(symbols: Sequence[str], element_colors: Dict[str, Union[str, Tuple[float, float, float]]]) -> np.ndarray:
    """
    Build a per-atom (N, 3) RGB array by looking each atom's symbol up in a color map.

    Args:
        symbols (Sequence[str]): Chemical symbol of each atom, e.g. from atoms.get_chemical_symbols().
        element_colors (Dict[str, Union[str, Tuple[float, float, float]]]): Map from chemical
            symbol to either a '#RRGGBB' hex string or an (r, g, b) tuple.

    Returns:
        np.ndarray: Array of shape (len(symbols), 3) with one RGB row per atom.
    """
    rgb_by_symbol = {
        symbol: hex_to_rgb(color) if isinstance(color, str) else tuple(color)
        for symbol, color in element_colors.items()
    }
    unique_symbols, inverse = np.unique(symbols, return_inverse=True)
    palette = np.array([rgb_by_symbol[symbol] for symbol in unique_symbols])
    return palette[inverse]


def _colors_by_scalar(values: Sequence[float], low_hex: str, high_hex: str) -> np.ndarray:
    """
    Linearly map a scalar array to per-atom RGB colors between two hex colors.

    Args:
        values (Sequence[float]): One scalar per atom (e.g. a charge or displacement).
        low_hex (str): Hex color assigned to the minimum value.
        high_hex (str): Hex color assigned to the maximum value.

    Returns:
        np.ndarray: Array of shape (len(values), 3) with one RGB row per atom.
    """
    values = np.asarray(values, dtype=float)
    low_rgb = np.array(hex_to_rgb(low_hex))
    high_rgb = np.array(hex_to_rgb(high_hex))
    vmin, vmax = values.min(), values.max()
    span = vmax - vmin
    fraction = np.zeros_like(values) if span == 0 else (values - vmin) / span
    return low_rgb + fraction[:, None] * (high_rgb - low_rgb)


class StructureRenderer:
    """
    Wraps an OVITO pipeline around an ase.Atoms structure for Tachyon rendering.
    """

    def __init__(
        self,
        atoms: Atoms,
        radii: Optional[Dict[str, float]] = None,
        bonds: Optional[Dict[Tuple[str, str], float]] = None,
        colors: Optional[Union[Dict[str, str], Sequence[float]]] = None,
        color_map: Optional[Dict[str, str]] = None,
        bond_width: float = 1.0,
    ) -> None:
        """
        Build the OVITO pipeline for a structure and apply the requested styling.

        Args:
            atoms (Atoms): The structure to render.
            radii (Optional[Dict[str, float]], optional): Per-element particle radius. Defaults to None.
            bonds (Optional[Dict[Tuple[str, str], float]], optional): Per-element-pair bond cutoff
                distances, passed to CreateBondsModifier. Defaults to None (no bonds).
            colors (Optional[Union[Dict[str, str], Sequence[float]]], optional): Either a
                {symbol: hex_color} map, or one scalar per atom to be mapped through `color_map`.
                Defaults to None (OVITO's default per-type colors).
            color_map (Optional[Dict[str, str]], optional): {"c_min": hex, "c_max": hex} used to
                interpolate colors when `colors` is a scalar array. Defaults to red-to-blue.
            bond_width (float, optional): Rendered bond line width. Defaults to 1.0.
        """
        self.atoms = atoms
        self.color_map = color_map or dict(_DEFAULT_COLOR_MAP)
        self.pipeline = OvitoPipeline(source=StaticSource(data=ase_to_ovito(atoms)))

        if radii is not None:
            self.set_particle_radii(radii)
        if bonds is not None:
            self.set_bonds(bonds, width=bond_width)
        if colors is not None:
            self.set_particle_colors(colors)

        self.configure_camera()

    def compute(self) -> DataCollection:
        """
        Evaluate the pipeline and return the resulting data collection.

        Returns:
            DataCollection: The current pipeline output.
        """
        return self.pipeline.compute()

    def set_particle_radii(self, radii: Dict[str, float]) -> None:
        """
        Set the display radius of each particle type by chemical symbol.

        Args:
            radii (Dict[str, float]): Map from chemical symbol to display radius.
        """
        self.radii = radii

        def apply_radii(frame: int, data: DataCollection) -> None:
            types = data.particles_.particle_types_
            for symbol, radius in self.radii.items():
                types.type_by_name_(symbol).radius = radius

        self.pipeline.modifiers.append(apply_radii)

    def set_bonds(self, cutoffs: Dict[Tuple[str, str], float], width: float = 1.0) -> None:
        """
        Create bonds between nearby particles using per-element-pair cutoff distances.

        Args:
            cutoffs (Dict[Tuple[str, str], float]): Map from (symbol_a, symbol_b) to cutoff distance.
            width (float, optional): Rendered bond line width. Defaults to 1.0.
        """
        self.bond_cutoffs = cutoffs
        modifier = CreateBondsModifier(mode=CreateBondsModifier.Mode.Pairwise)
        for (symbol_a, symbol_b), cutoff in cutoffs.items():
            modifier.set_pairwise_cutoff(symbol_a, symbol_b, cutoff)
        modifier.vis.width = width
        self.pipeline.modifiers.append(modifier)

    def set_particle_colors(self, colors: Union[Dict[str, str], Sequence[float]]) -> None:
        """
        Color particles either per chemical symbol or by mapping a per-atom scalar through `color_map`.

        Args:
            colors (Union[Dict[str, str], Sequence[float]]): Either a {symbol: hex_color} map, or one
                scalar value per atom.

        Raises:
            TypeError: If `colors` is neither a dict nor a sequence of scalars.
        """
        if isinstance(colors, dict):
            rgb = _colors_by_element(self.atoms.get_chemical_symbols(), colors)
        elif isinstance(colors, (list, tuple, np.ndarray)):
            rgb = _colors_by_scalar(colors, self.color_map["c_min"], self.color_map["c_max"])
        else:
            raise TypeError(f"colors must be a dict or a sequence of scalars, got {type(colors)!r}.")
        self.colors = rgb

        def apply_colors(frame: int, data: DataCollection) -> None:
            data.particles_.create_property("Color", data=self.colors)

        self.pipeline.modifiers.append(apply_colors)

    def configure_camera(
        self,
        renderer: str = "tachyon",
        shadows: bool = False,
        direct_light_intensity: float = 1.1,
        antialiasing_samples: int = 40,
        ambient_occlusion_brightness: float = 0.7,
        ambient_occlusion_samples: int = 24,
        direction: Sequence[float] = (3, 2, -1),
        distance: Optional[float] = None,
        fov: Optional[float] = None,
        look_at_shift: Sequence[float] = (0, 0, 0),
        perspective: bool = False,
    ) -> None:
        """
        Set up the OVITO viewport and renderer used by `render`.

        Args:
            renderer (str, optional): Renderer name, currently only "tachyon" is supported. Defaults to "tachyon".
            shadows (bool, optional): Whether to render cast shadows. Defaults to False: with ambient
                occlusion already handling contact shading between atoms, cast shadows are invisible in
                the common case of atoms rendered without a ground plane, so this just costs render time.
            direct_light_intensity (float, optional): Intensity of the directional light. Defaults to 1.1.
            antialiasing_samples (int, optional): Samples per pixel for edge antialiasing. Higher gives
                smoother sphere/bond outlines at the cost of render time. Defaults to 40 (OVITO's own
                default is 12).
            ambient_occlusion_brightness (float, optional): Brightness of ambient occlusion shading; lower
                values darken contact areas between atoms more, giving a rounder, less flat look. Defaults
                to 0.7 (OVITO's own default is 0.8).
            ambient_occlusion_samples (int, optional): Samples used to estimate ambient occlusion. Higher
                reduces shading noise at the cost of render time. Defaults to 24 (OVITO's own default is 12).
            direction (Sequence[float], optional): Camera viewing direction. Defaults to (3, 2, -1).
            distance (Optional[float], optional): Camera distance from the structure's center of mass.
                Defaults to None (three times the cube root of the cell volume).
            fov (Optional[float], optional): Camera field of view: an angle in radians for a perspective
                camera, or the visible half-height in world units for an orthographic camera. Unlike
                `distance`, this is what actually controls zoom for an orthographic camera. Defaults to
                None (OVITO's own default angle for perspective; the cube root of the cell volume for
                orthographic, so the structure fills the frame instead of using OVITO's fixed default
                of 100 world units).
            look_at_shift (Sequence[float], optional): Offset added to the look-at point. Defaults to (0, 0, 0).
            perspective (bool, optional): Use a perspective (vs. orthographic) camera. Defaults to False.

        Raises:
            ValueError: If `renderer` is not a supported renderer name.
        """
        if renderer not in _RENDERERS:
            raise ValueError(f"Unsupported renderer {renderer!r}; choose from {sorted(_RENDERERS)}.")
        self.renderer = _RENDERERS[renderer](
            shadows=shadows,
            direct_light_intensity=direct_light_intensity,
            antialiasing_samples=antialiasing_samples,
            ambient_occlusion_brightness=ambient_occlusion_brightness,
            ambient_occlusion_samples=ambient_occlusion_samples,
        )

        scale = self.atoms.get_volume() ** (1.0 / 3.0)
        if distance is None:
            distance = scale * 3.0
        direction = np.asarray(direction, dtype=float)
        look_at_shift = np.asarray(look_at_shift, dtype=float)
        unit_direction = direction / np.linalg.norm(direction)

        self.viewport = Viewport()
        self.viewport.type = Viewport.Type.Perspective if perspective else Viewport.Type.Ortho
        self.viewport.camera_dir = direction
        self.viewport.camera_pos = look_at_shift + self.atoms.get_center_of_mass() - unit_direction * distance
        if fov is not None:
            self.viewport.fov = fov
        elif not perspective:
            self.viewport.fov = scale

    def render(
        self,
        target: str = "image",
        filename: str = "atoms.png",
        size: Tuple[int, int] = (1024, 1024),
        alpha: bool = True,
        show: bool = True,
        crop: bool = False,
    ) -> Optional[str]:
        """
        Render the structure to a static image file or an interactive Jupyter widget.

        Args:
            target (str, optional): Either "image" or "jupyter". Defaults to "image".
            filename (str, optional): Output path, used when target is "image". Defaults to "atoms.png".
            size (Tuple[int, int], optional): Image size in pixels. Defaults to (1024, 1024).
            alpha (bool, optional): Whether to render a transparent background. Defaults to True.
            show (bool, optional): Whether to display the rendered image with matplotlib. Defaults to True.
            crop (bool, optional): Whether to crop the image to its non-empty bounding box. Defaults to False.

        Returns:
            Optional[str]: The written image filename when target is "image", otherwise None.

        Raises:
            ValueError: If `target` is not "image" or "jupyter".
        """
        if target not in ("image", "jupyter"):
            raise ValueError(f"target must be 'image' or 'jupyter', got {target!r}.")

        self.pipeline.add_to_scene()
        try:
            if target == "image":
                self.viewport.render_image(filename=filename, size=size, alpha=alpha, renderer=self.renderer, crop=crop)
                if show:
                    self._show_image(filename)
                return filename
            self._show_jupyter_widget()
            return None
        finally:
            self.pipeline.remove_from_scene()

    @staticmethod
    def _show_image(filename: str) -> None:
        """
        Display a rendered image file inline with matplotlib.

        Args:
            filename (str): Path to the image file to display.
        """
        plt.figure(figsize=(6, 6))
        plt.imshow(mpimg.imread(filename))
        plt.axis("off")

    def _show_jupyter_widget(self) -> None:
        """
        Display an interactive OVITO viewport widget in a Jupyter notebook.
        """
        import ipywidgets
        from IPython.display import display

        widget = self.viewport.create_jupyter_widget()
        widget.layout = ipywidgets.Layout(width="500px", height="400px")
        display(widget)
