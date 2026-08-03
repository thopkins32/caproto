#!/usr/bin/env python3
"""Serve a caproto IOC from an XRT XML beamline.

Usage
-----
python -m caproto.ioc_examples.xrt_xml_ioc --xml beamline.xml --prefix XRT:

The XML file defines the live configurable PVs. The same file is loaded into XRT
for in-memory simulation. The XML file on disk is never modified.
"""

from __future__ import annotations

import ast
import asyncio
import inspect
import logging
import re
import time
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from caproto import ChannelType
from caproto.server import PVSpec, run, template_arg_parser


logger = logging.getLogger("caproto.ctx.xrt_xml_ioc")

DEFAULT_IMAGE_MAX_LENGTH = 1024 * 1024
DEFAULT_FLOAT_PRECISION = 6
PATH_STRING_MAX_LENGTH = 4096
FILENAME_STRING_MAX_LENGTH = 1024
STATUS_STRINGS = ["Idle", "Acquiring", "Writing", "Error"]
BINARY_STRINGS = ["Off", "On"]
STRING_KWARGS = dict(string_encoding="utf-8", report_as_string=True)
STRUCTURAL_COMPONENTS = {"properties", "parameters"}
REF_OR_STRUCTURAL_ATTRS = {
    "bl",
    "uuid",
    "material",
    "material2",
    "figureError",
    "baseFE",
    "elements",
    "coating",
    "substrate",
    "tLayer",
    "bLayer",
}
COMPOUND_FIELDS = {
    "center": ["x", "y", "z"],
    "x": ["x", "y", "z"],
    "z": ["x", "y", "z"],
    "limPhysX": ["lmin", "lmax"],
    "limPhysY": ["lmin", "lmax"],
    "histShape": ["width", "height"],
    "opening": ["left", "right", "bottom", "top"],
    "blades": ["left", "right", "bottom", "top"],
}
DISCRETE_INTEGER_FIELDS = {
    "bins",
    "eN",
    "ePos",
    "histShape",
    "nrays",
    "nx",
    "nz",
    "pickleEvery",
    "ppb",
    "processes",
    "repeats",
    "threads",
    "updateEvery",
    "xPos",
    "yPos",
}
INTEGER_RE = re.compile(r"^[+-]?\d+$")
FLOAT_RE = re.compile(r"^[+-]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[eE][+-]?\d+)?$")


@dataclass
class XmlEntry:
    path: tuple[str, ...]
    raw_text: str
    value: Any
    field_name: str | None = None
    field_index: int | None = None

    @property
    def xml_path(self) -> str:
        return "/".join(self.path)


@dataclass
class LiveBinding:
    read: Callable[[], Any]
    write: Callable[[Any], None]
    integer_hint: bool = False
    internal: bool = False


@dataclass
class XmlPV:
    suffix: str
    path: tuple[str, ...]
    raw_text: str
    value: Any
    string_pv: bool
    binding: LiveBinding

    @property
    def xml_path(self) -> str:
        return "/".join(self.path)


@dataclass
class ScreenCapture:
    h5_path: Path | None = None
    h5_file: Any = None
    dataset: Any = None

    @property
    def is_open(self) -> bool:
        return self.h5_file is not None

    def open(
        self,
        screen: "ScreenState",
        *,
        overwrite: bool,
    ) -> None:
        try:
            import h5py
        except ImportError as exc:
            raise RuntimeError("h5py is required when Capture=1") from exc

        if self.is_open:
            return

        self.h5_path = screen.target_h5_path()
        self.h5_path.parent.mkdir(parents=True, exist_ok=True)
        self.h5_file = h5py.File(self.h5_path, "w" if overwrite else "x")

        height, width = screen.image_shape()
        group = self.h5_file.require_group("/entry/data")
        self.dataset = group.create_dataset(
            "data",
            shape=(0, height, width),
            maxshape=(None, height, width),
            chunks=(1, height, width),
            dtype="float64",
            compression="lzf",
        )
        logger.info(
            "Opened HDF5 capture for %s at %s with frame shape (%d, %d)",
            screen.name,
            self.h5_path,
            height,
            width,
        )

    def append(self, screen_name: str, frame: np.ndarray) -> None:
        if self.dataset is None:
            return
        if tuple(self.dataset.shape[1:]) != tuple(frame.shape):
            raise RuntimeError(
                f"{screen_name} image shape changed from "
                f"{self.dataset.shape[1:]} to {frame.shape}; close and reopen "
                "Capture to create a new dataset"
            )
        index = self.dataset.shape[0]
        self.dataset.resize((index + 1, *frame.shape))
        self.dataset[index, :, :] = frame
        self.h5_file.flush()
        logger.debug(
            "Appended frame %d for %s to %s; frame sum=%g max=%g",
            index,
            screen_name,
            self.h5_path,
            float(np.sum(frame)),
            float(np.max(frame)) if frame.size else 0.0,
        )

    def close(self) -> None:
        if self.h5_file is not None:
            logger.info("Closed HDF5 capture file %s", self.h5_path)
            self.h5_file.close()
        self.h5_path = None
        self.h5_file = None
        self.dataset = None


@dataclass
class ScreenState:
    name: str
    safe_name: str
    pv_suffix_base: str
    obj: Any
    capture: ScreenCapture = field(default_factory=ScreenCapture)
    acquire_pv: Any = None
    status_pv: Any = None
    capture_pv: Any = None
    file_path_pv: Any = None
    file_name_pv: Any = None
    num_images_pv: Any = None
    frames_written_pv: Any = None
    image_pv: Any = None

    def target_h5_path(self) -> Path:
        directory = Path(str(self.file_path_pv.value)).expanduser()
        return (directory / str(self.file_name_pv.value)).resolve(strict=False)

    def image_shape(self) -> tuple[int, int]:
        image = getattr(self.obj, "image", None)
        if image is not None:
            arr = np.asarray(image)
            if arr.ndim == 2 and arr.size:
                return int(arr.shape[0]), int(arr.shape[1])

        hist_shape = getattr(self.obj, "histShape", [256, 256])
        try:
            width, height = int(hist_shape[0]), int(hist_shape[1])
        except Exception:
            width, height = 256, 256
        return height, width


def _import_raycing():
    try:
        import xrt.backends.raycing as raycing
    except ModuleNotFoundError as exc:
        if exc.name != "xrt":
            raise
        raise RuntimeError(
            "Install xrt into the active environment; with pixi, add it as a "
            "conda or PyPI dependency."
        ) from exc
    return raycing


def _split_top_level(text: str) -> list[str]:
    parts = []
    start = 0
    depth = 0
    quote = ""
    escape = False
    for index, char in enumerate(text):
        if quote:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == quote:
                quote = ""
            continue
        if char in "'\"":
            quote = char
        elif char in "([{":
            depth += 1
        elif char in ")]}" and depth:
            depth -= 1
        elif char == "," and depth == 0:
            parts.append(text[start:index].strip())
            start = index + 1
    tail = text[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def _parse_text(text: str | None) -> Any:
    text = "" if text is None else str(text).strip()
    if text == "":
        return ""

    lowered = text.lower()
    if lowered == "none":
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if INTEGER_RE.match(text):
        return int(text)
    if FLOAT_RE.match(text):
        return float(text)

    if text[0] in "([{\"'":
        try:
            return ast.literal_eval(text)
        except (SyntaxError, ValueError):
            pass

    if len(text) >= 2 and text[0] in "[(" and text[-1] in ")]":
        values = [_parse_text(part) for part in _split_top_level(text[1:-1])]
        return tuple(values) if text[0] == "(" else values

    return text


def _format_text(value: Any) -> str:
    if value is None:
        return "None"
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, str):
        return value
    return repr(value)


def _coerce_put_value(value: Any) -> Any:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, np.ndarray):
        value = value.item() if value.size == 1 else value.tolist()
    if isinstance(value, str):
        return _parse_text(value)
    return value


def _bool_value(value: Any) -> bool:
    value = _coerce_put_value(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "on", "yes", "true"}
    return bool(value)


def _is_scalar(value: Any) -> bool:
    return isinstance(value, (bool, int, float, str, type(None), np.number))


def _string_pv_required(value: Any) -> bool:
    return not isinstance(value, (bool, int, float, np.number))


def _safe_component(part: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_-]+", "_", str(part).strip()).strip("_")
    return text or "item"


def _suffix_from_parts(parts: tuple[str, ...]) -> str:
    return ":".join(_safe_component(part) for part in parts if part)


def _path_parts(path: tuple[str, ...], *, drop_structural: bool) -> tuple[str, ...]:
    parts = path[1:] if path and path[0] == "Project" else path
    if drop_structural:
        parts = tuple(part for part in parts if part not in STRUCTURAL_COMPONENTS)
    return parts


def _compound_values(tag: str, raw_text: str, parsed_value: Any) -> list[Any] | None:
    fields = COMPOUND_FIELDS.get(tag)
    if fields is None:
        return None
    if isinstance(parsed_value, dict):
        try:
            values = [parsed_value[field] for field in fields]
        except KeyError:
            return None
    elif isinstance(parsed_value, (list, tuple)):
        values = list(parsed_value)
    elif tag.startswith("lim") and "," in raw_text:
        values = [
            _parse_text(part) for part in _split_top_level(raw_text.strip("[]() "))
        ]
    else:
        return None

    if len(values) != len(fields) or not all(_is_scalar(value) for value in values):
        return None
    return values


def _iter_xml_param_paths(root: ET.Element) -> list[tuple[tuple[str, ...], ET.Element]]:
    params = []

    def walk(node: ET.Element, path: tuple[str, ...]) -> None:
        if node.attrib.get("type") == "param":
            params.append((path, node))
        for child in node:
            walk(child, (*path, child.tag))

    walk(root, (root.tag,))
    return params


def _iter_xml_entries(root: ET.Element) -> list[XmlEntry]:
    entries: list[XmlEntry] = []
    for path, element in _iter_xml_param_paths(root):
        raw_text = "" if element.text is None else element.text.strip()
        parsed = _parse_text(raw_text)
        values = _compound_values(path[-1], raw_text, parsed)
        if values is None:
            entries.append(XmlEntry(path=path, raw_text=raw_text, value=parsed))
            continue
        for index, (field_name, value) in enumerate(
            zip(COMPOUND_FIELDS[path[-1]], values)
        ):
            entries.append(
                XmlEntry(
                    path=path,
                    raw_text=raw_text,
                    value=value,
                    field_name=field_name,
                    field_index=index,
                )
            )
    return entries


def _entry_suffix(entry: XmlEntry, *, drop_structural: bool) -> str:
    parts = _path_parts(entry.path, drop_structural=drop_structural)
    if entry.field_name is not None:
        parts = (*parts, entry.field_name)
    return _suffix_from_parts(parts)


def _unique_suffixes(entries: list[XmlEntry]) -> list[str]:
    dropped_suffixes = [_entry_suffix(entry, drop_structural=True) for entry in entries]
    full_suffixes = [_entry_suffix(entry, drop_structural=False) for entry in entries]
    counts = Counter(dropped_suffixes)
    used: Counter[str] = Counter()
    suffixes = []
    for dropped, full in zip(dropped_suffixes, full_suffixes):
        suffix = dropped if counts[dropped] == 1 else full
        used[suffix] += 1
        if used[suffix] > 1:
            suffix = f"{suffix}_{used[suffix]}"
        suffixes.append(suffix)
    return suffixes


def _child_text(parent: ET.Element | None, name: str) -> str | None:
    if parent is None:
        return None
    child = parent.find(name)
    if child is None or child.text is None:
        return None
    return child.text.strip()


def _find_beamline_node(root: ET.Element, beamline: Any) -> ET.Element | None:
    for child in root:
        object_text = _child_text(child, "_object")
        if object_text and object_text.endswith(".BeamLine"):
            return child
    return root.find(str(getattr(beamline, "name", "")))


class SimulationCoordinator:
    """Queue, batch, and execute all XRT screen acquisition requests."""

    coalesce_s = 0.05

    def __init__(
        self,
        *,
        raycing: Any,
        beamline: Any,
        screens: dict[str, ScreenState],
        image_max_length: int,
        overwrite: bool,
    ):
        self.raycing = raycing
        self.beamline = beamline
        self.screens = screens
        self.image_max_length = int(image_max_length)
        self.overwrite = overwrite
        self.queue: asyncio.Queue[str] = asyncio.Queue()
        self.worker_task: asyncio.Task | None = None
        self.capture_lock = asyncio.Lock()

    async def request(self, screen_name: str) -> None:
        logger.info("Queued acquisition request for %s", self.screens[screen_name].name)
        await self.screens[screen_name].status_pv.write("Acquiring")
        await self.queue.put(screen_name)
        if self.worker_task is None or self.worker_task.done():
            self.worker_task = asyncio.create_task(self._worker())

    async def set_capture(self, screen: ScreenState, enabled: bool) -> bool:
        loop = asyncio.get_running_loop()
        try:
            async with self.capture_lock:
                if enabled:
                    was_open = screen.capture.is_open
                    if was_open:
                        logger.info(
                            "Capture already open for %s at %s",
                            screen.name,
                            screen.capture.h5_path,
                        )
                    target = screen.target_h5_path()
                    for other in self.screens.values():
                        if other is screen:
                            continue
                        if other.capture.is_open and other.capture.h5_path == target:
                            raise RuntimeError(
                                f"{target} is already open for {other.name}; "
                                "each screen must capture to its own HDF5 file"
                            )
                    await loop.run_in_executor(
                        None,
                        lambda: screen.capture.open(
                            screen,
                            overwrite=self.overwrite,
                        ),
                    )
                    if not was_open:
                        await screen.frames_written_pv.write(0)
                else:
                    await loop.run_in_executor(None, screen.capture.close)
                    await screen.frames_written_pv.write(0)
        except Exception as exc:
            await screen.status_pv.write("Error")
            logger.exception("Failed to set Capture=%s for %s", enabled, screen.name)
            return False

        if not enabled and screen.status_pv.value != "Error":
            await screen.status_pv.write("Idle")
        return True

    async def close_all(self) -> None:
        loop = asyncio.get_running_loop()
        async with self.capture_lock:
            await loop.run_in_executor(
                None,
                lambda: [screen.capture.close() for screen in self.screens.values()],
            )

    async def _worker(self) -> None:
        # Requests queued during a run become the next batch; only one XRT run
        # sequence is active at a time.
        while True:
            try:
                first = await asyncio.wait_for(self.queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                return

            requested = {first}
            await asyncio.sleep(self.coalesce_s)
            while True:
                try:
                    requested.add(self.queue.get_nowait())
                except asyncio.QueueEmpty:
                    break
            logger.info(
                "Starting acquisition batch for screens: %s",
                ", ".join(self.screens[name].name for name in sorted(requested)),
            )
            await self._run_batch(requested)

    async def _run_batch(self, requested: set[str]) -> None:
        if not requested:
            return
        for name in requested:
            await self.screens[name].status_pv.write("Acquiring")

        num_images = {
            name: max(1, int(_coerce_put_value(self.screens[name].num_images_pv.value)))
            for name in requested
        }
        logger.info(
            "Batch frame counts: %s",
            ", ".join(
                f"{self.screens[name].name}={num_images[name]}"
                for name in sorted(requested)
            ),
        )
        loop = asyncio.get_running_loop()
        try:
            for image_index in range(max(num_images.values())):
                logger.debug("Running XRT frame index %d", image_index)
                images = await loop.run_in_executor(None, self._run_xrt_once)
                await self._update_previews(images)

                write_names = [
                    name
                    for name in requested
                    if image_index < num_images[name]
                    and name in images
                    and _bool_value(self.screens[name].capture_pv.value)
                ]
                if write_names:
                    for name in write_names:
                        await self.screens[name].status_pv.write("Writing")
                    async with self.capture_lock:
                        frames_written = await loop.run_in_executor(
                            None,
                            lambda names=write_names, frames=images: (
                                self._append_captures(names, frames)
                            ),
                        )
                    for name, count in frames_written.items():
                        await self.screens[name].frames_written_pv.write(count)
                        logger.info(
                            "%s FramesWritten=%d", self.screens[name].name, count
                        )
                    for name in write_names:
                        await self.screens[name].status_pv.write("Acquiring")
        except Exception as exc:
            for name in requested:
                await self.screens[name].status_pv.write("Error")
            logger.exception("XRT acquisition batch failed")
            return

        for name in requested:
            await self.screens[name].status_pv.write("Idle")
        logger.info("Completed acquisition batch")

    def _append_captures(
        self, names: list[str], images: dict[str, np.ndarray]
    ) -> dict[str, int]:
        frames_written = {}
        for name in names:
            screen = self.screens[name]
            screen.capture.append(screen.name, images[name])
            frames_written[name] = int(screen.capture.dataset.shape[0])
        return frames_written

    def _run_xrt_once(self) -> dict[str, np.ndarray]:
        start = time.monotonic()
        self.raycing.run_process_from_file(self.beamline)
        elapsed = time.monotonic() - start
        images: dict[str, np.ndarray] = {}
        for name, screen in self.screens.items():
            image = getattr(screen.obj, "image", None)
            if image is None:
                continue
            arr = np.asarray(image, dtype=np.float64)
            if arr.ndim == 2 and arr.size:
                images[name] = arr.copy()
                logger.debug(
                    "Screen %s image shape=%s sum=%g max=%g nonzero=%d",
                    screen.name,
                    arr.shape,
                    float(np.sum(arr)),
                    float(np.max(arr)),
                    int(np.count_nonzero(arr)),
                )
        logger.info(
            "XRT run completed in %.3f s with %d screen image(s)", elapsed, len(images)
        )
        return images

    def enable_screen_histograms(self) -> int:
        enabled = 0
        for oeid, methods in getattr(self.beamline, "flowU", {}).items():
            try:
                obj = self.beamline.oesDict[oeid][0]
            except Exception:
                continue
            for method, kwargs in methods.items():
                if method != "expose":
                    continue
                try:
                    parameters = inspect.signature(getattr(obj, method)).parameters
                except (TypeError, ValueError):
                    continue
                if "withHistogram" in parameters:
                    kwargs["withHistogram"] = True
                    enabled += 1
        logger.info("Enabled histogram output for %d screen expose flow(s)", enabled)
        return enabled

    async def _update_previews(self, images: dict[str, np.ndarray]) -> None:
        for name, frame in images.items():
            flat = np.asarray(frame, dtype=np.float64).ravel()
            if flat.size > self.image_max_length:
                flat = flat[: self.image_max_length]
                logger.debug(
                    "Truncated preview image for %s to %d elements",
                    self.screens[name].name,
                    self.image_max_length,
                )
            await self.screens[name].image_pv.write(flat, verify_value=False)


class XrtXmlIOC:
    def __init__(
        self,
        *,
        xml_path: str,
        prefix: str,
        image_max_length: int = DEFAULT_IMAGE_MAX_LENGTH,
        overwrite: bool = False,
    ):
        self.xml_path = Path(xml_path).expanduser().resolve()
        self.prefix = prefix
        self.image_max_length = max(1, int(image_max_length))
        self.overwrite = overwrite
        self.raycing = _import_raycing()
        self.tree = ET.parse(self.xml_path)
        self.root = self.tree.getroot()
        self.beamline = self.raycing.BeamLine(fileName=str(self.xml_path))
        self.beamline_node = _find_beamline_node(self.root, self.beamline)
        self.beamline_name = (
            self.beamline_node.tag
            if self.beamline_node is not None
            else self.beamline.name
        )
        self.element_uuids = self._element_uuid_map()
        self.materials = self._named_object_map(
            "Materials", "matnamesToUUIDs", "materialsDict"
        )
        self.figure_errors = self._named_object_map(
            "FigureErrors", "fenamesToUUIDs", "fesDict"
        )
        self.mapping: dict[str, XmlPV] = {}
        self.screens = self._screen_states()
        self.coordinator = SimulationCoordinator(
            raycing=self.raycing,
            beamline=self.beamline,
            screens=self.screens,
            image_max_length=self.image_max_length,
            overwrite=self.overwrite,
        )
        self.coordinator.enable_screen_histograms()
        self.pvdb = self._build_pvdb()
        logger.info(
            "Loaded XRT XML IOC from %s: beamline=%s, live_config_pvs=%d, screens=%d",
            self.xml_path,
            self.beamline_name,
            len(self.mapping),
            len(self.screens),
        )
        for screen in self.screens.values():
            logger.info(
                "Screen %s controls exposed under %s%s:",
                screen.name,
                self.prefix,
                screen.pv_suffix_base,
            )

    def _element_uuid_map(self) -> dict[str, str]:
        result: dict[str, str] = {}
        if self.beamline_node is None:
            return result
        for child in self.beamline_node:
            if child.tag in {"properties", "_object"}:
                continue
            if child.tag in getattr(self.beamline, "oesDict", {}):
                result[child.tag] = child.tag
                continue
            uuid = getattr(self.beamline, "oenamesToUUIDs", {}).get(child.tag)
            if uuid is not None:
                result[child.tag] = uuid
                continue
            name = _child_text(child.find("properties"), "name")
            uuid = getattr(self.beamline, "oenamesToUUIDs", {}).get(name)
            if uuid is not None:
                result[child.tag] = uuid
        return result

    def _named_object_map(
        self, section: str, names_attr: str, dict_attr: str
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}
        section_node = self.root.find(section)
        if section_node is None:
            return result
        names = getattr(self.beamline, names_attr, {})
        objects = getattr(self.beamline, dict_attr, {})
        for child in section_node:
            if child.tag in objects:
                result[child.tag] = objects[child.tag]
                continue
            uuid = names.get(child.tag)
            if uuid in objects:
                result[child.tag] = objects[uuid]
                continue
            name = _child_text(child.find("properties"), "name")
            uuid = names.get(name)
            if uuid in objects:
                result[child.tag] = objects[uuid]
        return result

    def _screen_states(self) -> dict[str, ScreenState]:
        screens: dict[str, ScreenState] = {}
        used_pv_suffixes: Counter[str] = Counter()
        screen_xml_names = {uuid: name for name, uuid in self.element_uuids.items()}
        for screen in getattr(self.beamline, "screens", []):
            name = str(getattr(screen, "name", "") or getattr(screen, "uuid", "screen"))
            safe_name = _safe_component(name)
            if safe_name in screens:
                safe_name = f"{safe_name}_{str(getattr(screen, 'uuid', ''))[:8]}"
            xml_name = screen_xml_names.get(getattr(screen, "uuid", None))
            if xml_name is None:
                pv_suffix_base = safe_name
            else:
                pv_suffix_base = _suffix_from_parts((self.beamline_name, xml_name))
            used_pv_suffixes[pv_suffix_base] += 1
            if used_pv_suffixes[pv_suffix_base] > 1:
                pv_suffix_base = f"{pv_suffix_base}_{used_pv_suffixes[pv_suffix_base]}"
            screens[safe_name] = ScreenState(
                name=name,
                safe_name=safe_name,
                pv_suffix_base=pv_suffix_base,
                obj=screen,
            )
        return screens

    def _build_pvdb(self) -> dict[str, Any]:
        specs = [*self._xml_pv_specs(), *self._screen_pv_specs()]
        pvdb = {spec.name: spec.create(group=None) for spec in specs}
        for screen in self.screens.values():
            base = f"{self.prefix}{screen.pv_suffix_base}"
            screen.acquire_pv = pvdb[f"{base}:Acquire"]
            screen.status_pv = pvdb[f"{base}:AcquireStatus"]
            screen.capture_pv = pvdb[f"{base}:Capture"]
            screen.file_path_pv = pvdb[f"{base}:FilePath"]
            screen.file_name_pv = pvdb[f"{base}:FileName"]
            screen.num_images_pv = pvdb[f"{base}:NumImages"]
            screen.frames_written_pv = pvdb[f"{base}:FramesWritten"]
            screen.image_pv = pvdb[f"{base}:Image"]
        return pvdb

    def _xml_pv_specs(self) -> list[PVSpec]:
        bound_entries: list[tuple[XmlEntry, LiveBinding]] = []
        for entry in _iter_xml_entries(self.root):
            binding = self._live_binding_for(entry)
            if binding is None or binding.internal:
                continue
            try:
                value = binding.read()
            except Exception:
                logger.debug(
                    "Skipping XML parameter without readable live binding: %s",
                    entry.xml_path,
                    exc_info=True,
                )
                continue
            if isinstance(value, np.generic):
                value = value.item()
            entry.value = value
            bound_entries.append((entry, binding))

        suffixes = _unique_suffixes([entry for entry, _binding in bound_entries])
        specs = []
        for suffix, (entry, binding) in zip(suffixes, bound_entries):
            xml_pv = XmlPV(
                suffix=suffix,
                path=entry.path,
                raw_text=entry.raw_text,
                value=entry.value,
                string_pv=_string_pv_required(entry.value),
                binding=binding,
            )
            self.mapping[suffix] = xml_pv
            specs.append(self._config_spec(xml_pv))
        return specs

    def _config_spec(self, xml_pv: XmlPV) -> PVSpec:
        async def getter(instance, *, xml_pv=xml_pv):
            return self._pv_readback_value(xml_pv, xml_pv.binding.read())

        async def putter(instance, value, *, xml_pv=xml_pv):
            return self._write_live_pv(xml_pv, value)

        value = self._pv_readback_value(xml_pv, xml_pv.value)
        if xml_pv.string_pv:
            return PVSpec(
                name=self.prefix + xml_pv.suffix,
                value=value,
                dtype=str,
                get=getter,
                put=putter,
                max_length=4096,
                cls_kwargs=STRING_KWARGS,
                doc=f"XML {xml_pv.xml_path}; raw XML value {xml_pv.raw_text!r}",
            )
        if isinstance(value, bool):
            return PVSpec(
                name=self.prefix + xml_pv.suffix,
                value=value,
                dtype=bool,
                record="bo",
                get=getter,
                put=putter,
                doc=f"XML {xml_pv.xml_path}; raw XML value {xml_pv.raw_text!r}",
            )
        if self._should_use_integer_pv(xml_pv) and isinstance(value, (int, float)):
            return PVSpec(
                name=self.prefix + xml_pv.suffix,
                value=int(value),
                dtype=int,
                get=getter,
                put=putter,
                doc=f"XML {xml_pv.xml_path}; raw XML value {xml_pv.raw_text!r}",
            )
        return PVSpec(
            name=self.prefix + xml_pv.suffix,
            value=float(value),
            dtype=float,
            get=getter,
            put=putter,
            cls_kwargs={"precision": DEFAULT_FLOAT_PRECISION},
            doc=f"XML {xml_pv.xml_path}; raw XML value {xml_pv.raw_text!r}",
        )

    def _pv_readback_value(self, xml_pv: XmlPV, value: Any) -> Any:
        if isinstance(value, np.generic):
            value = value.item()
        if xml_pv.string_pv:
            return _format_text(value)
        if self._should_use_integer_pv(xml_pv) and isinstance(value, (int, float)):
            return int(value)
        if isinstance(value, int) and not isinstance(value, bool):
            return float(value)
        return value

    def _should_use_integer_pv(self, xml_pv: XmlPV) -> bool:
        return xml_pv.binding.integer_hint

    def _screen_pv_specs(self) -> list[PVSpec]:
        specs: list[PVSpec] = []
        for screen in self.screens.values():
            base = screen.pv_suffix_base

            async def acquire_putter(instance, value, *, screen=screen):
                if _bool_value(value):
                    await self.coordinator.request(screen.safe_name)
                await instance.write("Off", verify_value=False)
                return "Off"

            async def capture_putter(instance, value, *, screen=screen):
                enabled = _bool_value(value)
                ok = await self.coordinator.set_capture(screen, enabled)
                return "On" if enabled and ok else "Off"

            async def num_images_putter(instance, value):
                return max(1, int(_coerce_put_value(value)))

            async def shutdown(instance, async_lib):
                await self.coordinator.close_all()

            specs.extend(
                [
                    PVSpec(
                        name=self.prefix + f"{base}:Acquire",
                        value="Off",
                        dtype=ChannelType.ENUM,
                        record="bo",
                        cls_kwargs={"enum_strings": BINARY_STRINGS},
                        put=acquire_putter,
                        doc="Per-screen software trigger",
                    ),
                    PVSpec(
                        name=self.prefix + f"{base}:AcquireStatus",
                        value="Idle",
                        dtype=ChannelType.ENUM,
                        record="mbbi",
                        read_only=True,
                        cls_kwargs={"enum_strings": STATUS_STRINGS},
                        doc="Idle, Acquiring, Writing, or Error",
                    ),
                    PVSpec(
                        name=self.prefix + f"{base}:Capture",
                        value="Off",
                        dtype=ChannelType.ENUM,
                        record="bo",
                        cls_kwargs={"enum_strings": BINARY_STRINGS},
                        put=capture_putter,
                        shutdown=shutdown,
                        doc="Open or close this screen's HDF5 file",
                    ),
                    PVSpec(
                        name=self.prefix + f"{base}:FilePath",
                        value=str(Path.cwd()),
                        dtype=str,
                        max_length=PATH_STRING_MAX_LENGTH,
                        doc="Directory used when Capture changes to 1",
                    ),
                    PVSpec(
                        name=self.prefix + f"{base}:FileName",
                        value=f"{base}.h5",
                        dtype=str,
                        max_length=FILENAME_STRING_MAX_LENGTH,
                        doc="Filename used when Capture changes to 1",
                    ),
                    PVSpec(
                        name=self.prefix + f"{base}:NumImages",
                        value=1,
                        dtype=int,
                        put=num_images_putter,
                        doc="Number of frames to acquire; minimum is 1",
                    ),
                    PVSpec(
                        name=self.prefix + f"{base}:FramesWritten",
                        value=0,
                        dtype=int,
                        read_only=True,
                        doc="Frames appended to this screen's open HDF5 file",
                    ),
                    PVSpec(
                        name=self.prefix + f"{base}:Image",
                        value=[0.0],
                        dtype=float,
                        max_length=self.image_max_length,
                        record="waveform",
                        read_only=True,
                        doc="Flattened latest image preview",
                    ),
                ]
            )
        return specs

    def _live_binding_for(self, entry: XmlEntry) -> LiveBinding | None:
        path = entry.path
        if len(path) < 3 or path[0] != "Project":
            return None
        section = path[1]

        if section == self.beamline_name:
            if len(path) == 4 and path[2] == "properties":
                return self._attr_binding(entry, self.beamline, path[3])
            if len(path) < 5:
                return None
            oeid = self.element_uuids.get(path[2])
            if oeid is None:
                return None
            target = self.beamline.oesDict[oeid][0]
            if len(path) == 5 and path[3] == "properties":
                return self._attr_binding(entry, target, path[4])
            if len(path) == 6 and path[4] == "parameters":
                return self._flow_binding(entry, oeid, path[3], path[5])
            return None

        if section == "Materials" and len(path) == 5 and path[3] == "properties":
            target = self.materials.get(path[2])
            if target is not None:
                return self._attr_binding(entry, target, path[4])
        elif section == "FigureErrors" and len(path) == 5 and path[3] == "properties":
            target = self.figure_errors.get(path[2])
            if target is not None:
                return self._attr_binding(entry, target, path[4])
        return None

    def _attr_binding(
        self, entry: XmlEntry, target: Any, attr: str
    ) -> LiveBinding | None:
        if attr in REF_OR_STRUCTURAL_ATTRS:
            return None

        def read() -> Any:
            return self._item_value(
                getattr(target, attr), entry.field_index, entry.field_name
            )

        def write(value: Any) -> None:
            if entry.field_index is None:
                setattr(target, attr, self._xrt_value(value))
                return

            current = getattr(target, attr)
            if isinstance(current, dict):
                current[entry.field_name] = value
                setattr(target, attr, current)
                return

            try:
                values = list(current)
            except TypeError:
                values = []
            while len(values) <= entry.field_index:
                values.append(0)
            values[entry.field_index] = self._xrt_value(value)
            setattr(target, attr, self._named_value(attr, values))

        return LiveBinding(
            read=read,
            write=write,
            integer_hint=self._integer_hint(attr, entry.value),
        )

    def _flow_binding(
        self, entry: XmlEntry, oeid: str, method: str, arg: str
    ) -> LiveBinding | None:
        methods = self.beamline.flowU.get(oeid, {})
        kwargs = methods.get(method)
        if kwargs is None or arg not in kwargs:
            return None

        def read() -> Any:
            value = kwargs.get(arg, entry.value)
            if arg == "beam":
                value = self._beam_name(value)
            return self._item_value(value, entry.field_index, entry.field_name)

        def write(value: Any) -> None:
            if entry.field_index is None:
                kwargs[arg] = self._flow_value(arg, value)
                return

            values = list(kwargs.get(arg, []))
            while len(values) <= entry.field_index:
                values.append(None)
            values[entry.field_index] = self._xrt_value(value)
            kwargs[arg] = values

        return LiveBinding(
            read=read,
            write=write,
            integer_hint=self._integer_hint(arg, entry.value),
            internal=method == "expose" and arg == "withHistogram",
        )

    def _item_value(
        self, value: Any, index: int | None, field_name: str | None
    ) -> Any:
        if index is not None:
            value = value[field_name] if isinstance(value, dict) else value[index]
        if isinstance(value, np.generic):
            return value.item()
        return value

    def _beam_name(self, value: Any) -> Any:
        for beam_name, beam_tag in self.beamline.beamNamesDict.items():
            if beam_tag[0] == value:
                return beam_name
        return value

    def _integer_hint(self, name: str, value: Any) -> bool:
        if isinstance(value, bool):
            return False
        return name in DISCRETE_INTEGER_FIELDS

    def _write_live_pv(self, xml_pv: XmlPV, value: Any) -> Any:
        value = _coerce_put_value(value)
        try:
            xml_pv.binding.write(value)
        except Exception:
            logger.exception("Could not update live XRT binding for %s", xml_pv.suffix)
            raise
        readback = xml_pv.binding.read()
        xml_pv.value = readback
        return self._pv_readback_value(xml_pv, readback)

    def _xrt_value(self, value: Any) -> Any:
        if isinstance(value, str):
            try:
                return self.raycing.parametrize(value)
            except Exception:
                return value
        return value

    def _flow_value(self, arg: str, value: Any) -> Any:
        if arg != "beam":
            return self._xrt_value(value)
        if value in {None, "None", ""}:
            return None
        if self.raycing.is_valid_uuid(value):
            return value
        beam_tag = self.beamline.beamNamesDict.get(str(value))
        return beam_tag[0] if beam_tag is not None else value

    def _named_value(self, attr: str, values: list[Any]) -> Any:
        if attr.startswith("limPhys") and all(
            not isinstance(value, str) for value in values
        ):
            return self.raycing.Limits(values)
        if attr == "histShape":
            return self.raycing.Image2D([int(value) for value in values])
        return values


def main() -> None:
    parser, split_args = template_arg_parser(
        default_prefix="xrt:",
        desc=__doc__,
        supported_async_libs=["asyncio"],
    )
    parser.add_argument("--xml", required=True, help="Path to the XRT XML beamline")
    parser.add_argument(
        "--image-max-length",
        type=int,
        default=DEFAULT_IMAGE_MAX_LENGTH,
        help="Maximum flattened Image waveform length",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow Capture=1 to overwrite an existing HDF5 file",
    )
    args = parser.parse_args()
    ioc_options, run_options = split_args(args)
    ioc = XrtXmlIOC(
        xml_path=args.xml,
        prefix=ioc_options["prefix"],
        image_max_length=args.image_max_length,
        overwrite=args.overwrite,
    )
    run(ioc.pvdb, **run_options)


if __name__ == "__main__":
    main()
