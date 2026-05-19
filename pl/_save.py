from __future__ import annotations

from pathlib import Path


def save_figure(ax, save=None, save_kwargs=None):
    """
    Save an axes' parent figure when ``save`` is provided.

    Plotting functions keep their normal return contracts; this helper is a
    side effect only. Parent directories are created automatically.
    """
    if save in (None, False):
        return

    path = Path(save)
    if path.parent and path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)

    kwargs = {"bbox_inches": "tight"}
    if save_kwargs:
        kwargs.update(save_kwargs)
    ax.figure.savefig(path, **kwargs)
