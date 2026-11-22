"""Trimap editing with stroke ownership, selective undo and stale-write guards.

The original RGB image is never modified.  Edits produce *new* trimap arrays;
the history holds immutable states so every operation can be reverted.  Markers
are hard labels (definite bg / definite fg); unknown pixels remain unknown
until the solver runs, and the solver's output is never written back into the
trimap.

Beyond simple snapshot undo, the editor tracks *which stroke owns each pixel*:

* ``undo_stroke(sid)`` releases exactly one stroke's constraints.  Pixels the
  stroke still owns are restored to their pre-stroke labels; pixels a later
  stroke has since taken over keep the later label, and the later stroke's
  record is re-threaded so its own undo restores the pre-stroke value.
* ``begin_stroke`` / ``end_stroke`` group any number of brush segments into
  ONE atomic history entry, so a drag that crosses tile blocks undoes in a
  single step.
* ``apply_external(trimap, base_version)`` merges a trimap that was computed
  from an older editor state (e.g. a delayed async result).  Pixels locked by
  strokes committed after ``base_version`` are protected: the delayed map can
  never re-release them.
"""
from __future__ import annotations

import numpy as np

from .imageio import TRIMAP_BG, TRIMAP_FG, TRIMAP_UNKNOWN

_VALID_LABELS = (TRIMAP_BG, TRIMAP_UNKNOWN, TRIMAP_FG)


class _Stroke:
    """One committed edit: per-pixel new labels plus the values/owners it
    displaced, so it can be released individually later."""

    __slots__ = ("sid", "pixels", "labels", "prev_labels", "prev_owners",
                 "commit_version", "active", "_pos")

    def __init__(self, sid, pixels, labels, prev_labels, prev_owners,
                 commit_version):
        self.sid = sid
        self.pixels = pixels              # flat indices, unique, sorted
        self.labels = labels              # new label per pixel
        self.prev_labels = prev_labels    # label before this stroke
        self.prev_owners = prev_owners    # owner before this stroke (0=initial)
        self.commit_version = commit_version
        self.active = True
        self._pos = None                  # lazy pixel -> position lookup

    def copy(self) -> "_Stroke":
        s = _Stroke(self.sid, self.pixels.copy(), self.labels.copy(),
                    self.prev_labels.copy(), self.prev_owners.copy(),
                    self.commit_version)
        s.active = self.active
        return s

    def position(self, pixel: int) -> int:
        if self._pos is None:
            self._pos = {int(p): i for i, p in enumerate(self.pixels)}
        return self._pos[pixel]


class _State:
    __slots__ = ("trimap", "owner", "strokes", "version")

    def __init__(self, trimap, owner, strokes, version):
        self.trimap = trimap
        self.owner = owner                # flat int32: owning stroke sid, 0=init
        self.strokes = strokes            # list[_Stroke]
        self.version = version

    def copy(self) -> "_State":
        return _State(self.trimap.copy(), self.owner.copy(),
                      [s.copy() for s in self.strokes], self.version)


class TrimapEditor:
    def __init__(self, trimap: np.ndarray, max_history: int = 50):
        self._initial = trimap.copy()
        self._counter = 0                 # global, monotonic commit counter
        self._history: list[_State] = [
            _State(trimap.copy(),
                   np.zeros(trimap.size, dtype=np.int32), [], 0)
        ]
        self._cursor = 0
        self._max_history = max_history
        # Open grouped stroke (between begin_stroke and end_stroke).
        self._group_work: np.ndarray | None = None
        self._group_record: dict[int, list] | None = None

    # ------------------------------------------------------------------ state
    @property
    def _state(self) -> _State:
        return self._history[self._cursor]

    @property
    def trimap(self) -> np.ndarray:
        if self._group_work is not None:
            return self._group_work
        return self._state.trimap

    @property
    def version(self) -> int:
        """Commit version of the current state; use it as ``base_version`` for
        any asynchronous computation that will later be written back."""
        return self._state.version

    @property
    def can_undo(self) -> bool:
        return self._cursor > 0

    @property
    def can_redo(self) -> bool:
        return self._cursor < len(self._history) - 1

    @property
    def stroke_ids(self) -> list[int]:
        """Sids of the strokes that are active in the current state."""
        return [s.sid for s in self._state.strokes if s.active]

    # ---------------------------------------------------------------- commits
    def _commit(self, trimap: np.ndarray, owner: np.ndarray,
                strokes: list[_Stroke]) -> np.ndarray:
        self._counter += 1
        del self._history[self._cursor + 1:]
        self._history.append(_State(trimap, owner, strokes, self._counter))
        if len(self._history) > self._max_history:
            self._history.pop(0)
        self._cursor = len(self._history) - 1
        return self.trimap

    def _check_mask(self, mask: np.ndarray) -> None:
        if mask.shape != self.trimap.shape or mask.dtype != bool:
            raise ValueError("paint mask must be a boolean array of trimap shape")

    def _apply_pixels(self, pixels: np.ndarray, labels: np.ndarray
                      ) -> np.ndarray:
        """Commit one stroke record for already-grouped pixel/label arrays."""
        state = self._state
        new_map = state.trimap.copy()
        new_owner = state.owner.copy()
        flat = new_map.ravel()
        prev_labels = flat[pixels].copy()
        prev_owners = state.owner[pixels].copy()
        flat[pixels] = labels
        sid = len(state.strokes) + 1
        new_owner[pixels] = sid
        stroke = _Stroke(sid, pixels, labels.copy(), prev_labels, prev_owners,
                         commit_version=self._counter + 1)
        return self._commit(new_map, new_owner, state.strokes + [stroke])

    def paint(self, mask: np.ndarray, label: int) -> np.ndarray:
        """Apply a hard marker (TRIMAP_BG / TRIMAP_FG) on the given pixel mask."""
        if label not in (TRIMAP_BG, TRIMAP_FG):
            raise ValueError("hard markers must be definite bg (0) or fg (255)")
        self._check_mask(mask)
        if self._group_work is not None:
            self._group_apply(mask, label)
            return self._group_work
        pixels = np.flatnonzero(mask.ravel())
        if pixels.size == 0:
            return self.trimap
        labels = np.full(pixels.shape, label, dtype=np.uint8)
        return self._apply_pixels(pixels, labels)

    def erase_to_unknown(self, mask: np.ndarray) -> np.ndarray:
        self._check_mask(mask)
        if self._group_work is not None:
            self._group_apply(mask, TRIMAP_UNKNOWN)
            return self._group_work
        pixels = np.flatnonzero(mask.ravel())
        if pixels.size == 0:
            return self.trimap
        labels = np.full(pixels.shape, TRIMAP_UNKNOWN, dtype=np.uint8)
        return self._apply_pixels(pixels, labels)

    # ------------------------------------------------------- grouped strokes
    def begin_stroke(self) -> None:
        """Start an atomic stroke: subsequent paint/erase calls accumulate
        into a single history entry until ``end_stroke``.  A drag crossing
        tile blocks therefore undoes in one step."""
        if self._group_work is not None:
            raise RuntimeError("a stroke is already in progress")
        self._group_work = self._state.trimap.copy()
        self._group_record = {}

    def _group_apply(self, mask: np.ndarray, label: int) -> None:
        state = self._state
        flat_work = self._group_work.ravel()
        flat_base = state.trimap.ravel()
        for p in np.flatnonzero(mask.ravel()):
            p = int(p)
            if p not in self._group_record:
                self._group_record[p] = [label, flat_base[p],
                                         int(state.owner[p])]
            else:
                self._group_record[p][0] = label
            flat_work[p] = label

    def end_stroke(self) -> np.ndarray:
        """Commit the grouped stroke as ONE undoable entry."""
        if self._group_work is None:
            raise RuntimeError("no stroke in progress")
        record = self._group_record
        work = self._group_work
        self._group_work = None
        self._group_record = None
        if not record:
            return self.trimap
        pixels = np.array(sorted(record), dtype=np.int64)
        labels = np.array([record[p][0] for p in pixels], dtype=np.uint8)
        prev_labels = np.array([record[p][1] for p in pixels], dtype=np.uint8)
        prev_owners = np.array([record[p][2] for p in pixels], dtype=np.int32)
        state = self._state
        new_owner = state.owner.copy()
        sid = len(state.strokes) + 1
        new_owner[pixels] = sid
        stroke = _Stroke(sid, pixels, labels, prev_labels, prev_owners,
                         commit_version=self._counter + 1)
        return self._commit(work, new_owner, state.strokes + [stroke])

    def _require_no_open_stroke(self) -> None:
        if self._group_work is not None:
            raise RuntimeError("finish the current stroke first (end_stroke)")

    # ------------------------------------------------------------- undo/redo
    def undo(self) -> np.ndarray:
        self._require_no_open_stroke()
        if self._cursor > 0:
            self._cursor -= 1
        return self.trimap

    def redo(self) -> np.ndarray:
        self._require_no_open_stroke()
        if self._cursor < len(self._history) - 1:
            self._cursor += 1
        return self.trimap

    def undo_stroke(self, sid: int) -> np.ndarray:
        """Release exactly one stroke's constraints, per pixel.

        Pixels the stroke still owns return to their pre-stroke labels
        (complete undo outside overlaps).  Pixels a later stroke took over
        keep the later label; the later record is re-threaded so its own
        undo restores the pre-stroke value rather than resurrecting the
        released constraint.
        """
        self._require_no_open_stroke()
        state = self._state
        target = None
        for s in state.strokes:
            if s.sid == sid:
                target = s
                break
        if target is None:
            raise ValueError(f"unknown stroke id {sid}")
        if not target.active:
            return self.trimap

        new_state = state.copy()
        strokes = {s.sid: s for s in new_state.strokes}
        t = strokes[sid]
        flat = new_state.trimap.ravel()
        for i, p in enumerate(t.pixels):
            p = int(p)
            owner = int(new_state.owner[p])
            if owner == sid:
                flat[p] = t.prev_labels[i]
                new_state.owner[p] = t.prev_owners[i]
            else:
                later = strokes[owner]
                j = later.position(p)
                later.prev_labels[j] = t.prev_labels[i]
                later.prev_owners[j] = t.prev_owners[i]
        t.active = False
        return self._commit(new_state.trimap, new_state.owner,
                            new_state.strokes)

    def reset(self) -> np.ndarray:
        self._require_no_open_stroke()
        self._counter += 1
        self._history = [_State(self._initial.copy(),
                                np.zeros(self._initial.size, dtype=np.int32),
                                [], self._counter)]
        self._cursor = 0
        return self.trimap

    # -------------------------------------------------------- guarded writes
    def apply_external(self, trimap: np.ndarray,
                       base_version: int) -> np.ndarray:
        """Merge a trimap computed from an older editor state.

        Pixels locked by strokes committed after ``base_version`` are
        protected: a delayed write-back can update everything else but can
        never re-release those locks.  The merge is one atomic undoable
        entry.
        """
        self._require_no_open_stroke()
        if not isinstance(base_version, (int, np.integer)):
            raise TypeError("base_version must be an int returned by .version")
        if base_version < 0 or base_version > self.version:
            raise ValueError(
                f"base_version {base_version} does not belong to the current "
                f"history (current version {self.version})")
        incoming = np.asarray(trimap)
        if incoming.shape != self.trimap.shape:
            raise ValueError("external trimap shape mismatch")
        if not np.isin(incoming, _VALID_LABELS).all():
            raise ValueError("external trimap contains invalid labels")

        state = self._state
        owner_version = np.zeros(state.trimap.size, dtype=np.int64)
        for s in state.strokes:
            if s.active:
                owner_version[s.pixels] = s.commit_version
        protected = owner_version > base_version
        new_map = np.where(protected.reshape(state.trimap.shape),
                           state.trimap, incoming)
        changed = np.flatnonzero(
            (new_map != state.trimap).ravel())
        if changed.size == 0:
            return self.trimap
        labels = new_map.ravel()[changed].astype(np.uint8)
        return self._apply_pixels(changed, labels)


def disk_mask(shape: tuple[int, int], center: tuple[int, int],
              radius: int) -> np.ndarray:
    h, w = shape
    y, x = np.ogrid[:h, :w]
    cy, cx = center
    return (y - cy) ** 2 + (x - cx) ** 2 <= radius ** 2


def line_mask(shape: tuple[int, int], p0: tuple[int, int],
              p1: tuple[int, int], radius: int) -> np.ndarray:
    """Rasterised thick segment used for brush strokes between pointer points."""
    h, w = shape
    y0, x0 = p0
    y1, x1 = p1
    n = max(abs(y1 - y0), abs(x1 - x0), 1) + 1
    ys = np.linspace(y0, y1, n).round().astype(int)
    xs = np.linspace(x0, x1, n).round().astype(int)
    mask = np.zeros((h, w), dtype=bool)
    yy, xx = np.ogrid[:h, :w]
    for cy, cx in zip(ys, xs):
        mask |= (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2
    return mask
