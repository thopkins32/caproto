#!/usr/bin/env python3
"""Serve a caproto IOC from an XRT XML beamline.

Usage
-----
python -m caproto.ioc_examples.xrt_xml_ioc --xml beamline.xml --prefix XRT:

The XML file defines the configurable PVs. The same file is loaded into XRT for
in-memory simulation. The XML file on disk is never modified.
"""

from __future__ import annotations

import ast
import asyncio
import inspect
import re
import time
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from caproto import ChannelType
from caproto.server import PVSpec, run, template_arg_parser


DEFAULT_IMAGE_MAX_LENGTH = 1024 * 1024
DEFAULT_FLOAT_PRECISION = 6
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
FLOAT_RE = re.compile(
    r"^[+-]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[eE][+-]?\d+)?$"
)


@dataclass
class XmlPV:
    suffix: str
    path: tuple[str, ...]
    element: ET.Element
    value: Any
    string_pv: bool
    read_only: bool = True
    field_name: str | None = None
    field_index: int | None = None
    live_kind: str | None = None
    target: Any = None
    attr: str | None = None
    oeid: str | None = None
    method: str | None = None
    arg: str | None = None

    @property
    def xml_path(self) -> str:
        return "/".join(self.path)

    @property
    def raw_text(self) -> str:
        return "" if self.element.text is None else self.element.text.strip()

    @property
    def parsed_value(self) -> Any:
        value = _parse_text(self.raw_text)
        if self.field_index is None:
            return value
        try:
            return value[self.field_index]
        except Exception:
            return self.value


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
        source_xml: str,
        pv_prefix: str,
        beamline_name: str,
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
        self.h5_file.attrs["source_xml"] = source_xml
        self.h5_file.attrs["pv_prefix"] = pv_prefix
        self.h5_file.attrs["xrt_beamline_name"] = beamline_name
        self.h5_file.attrs["created"] = time.time()

        height, width = screen.image_shape()
        group = self.h5_file.require_group(f"/entry/screens/{screen.safe_name}")
        group.attrs["screen_name"] = screen.name
        group.attrs["source_xml"] = source_xml
        group.attrs["pv_prefix"] = pv_prefix
        group.attrs["timestamp"] = time.time()
        group.attrs["xrt_beamline_name"] = beamline_name
        self.dataset = group.create_dataset(
            "image",
            shape=(0, height, width),
            maxshape=(None, height, width),
            chunks=(1, height, width),
            dtype="float64",
            compression="lzf",
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

    def close(self) -> None:
        if self.h5_file is not None:
            self.h5_file.close()
        self.h5_path = None
        self.h5_file = None
        self.dataset = None


@dataclass
class ScreenState:
    name: str
    safe_name: str
    obj: Any
    capture: ScreenCapture = field(default_factory=ScreenCapture)
    acquire_pv: Any = None
    status_pv: Any = None
    capture_pv: Any = None
    file_path_pv: Any = None
    file_name_pv: Any = None
    num_images_pv: Any = None
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
        values = [_parse_text(part) for part in _split_top_level(raw_text.strip("[]() "))]
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
        xml_path: Path,
        prefix: str,
        image_max_length: int,
        overwrite: bool,
    ):
        self.raycing = raycing
        self.beamline = beamline
        self.screens = screens
        self.xml_path = xml_path
        self.prefix = prefix
        self.image_max_length = int(image_max_length)
        self.overwrite = overwrite
        self.queue: asyncio.Queue[str] = asyncio.Queue()
        self.worker_task: asyncio.Task | None = None
        self.capture_lock = asyncio.Lock()

    async def request(self, screen_name: str) -> None:
        await self.screens[screen_name].status_pv.write("Acquiring")
        await self.queue.put(screen_name)
        if self.worker_task is None or self.worker_task.done():
            self.worker_task = asyncio.create_task(self._worker())

    async def set_capture(self, screen: ScreenState, enabled: bool) -> bool:
        loop = asyncio.get_running_loop()
        try:
            async with self.capture_lock:
                if enabled:
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
                            source_xml=str(self.xml_path),
                            pv_prefix=self.prefix,
                            beamline_name=str(getattr(self.beamline, "name", "")),
                            overwrite=self.overwrite,
                        ),
                    )
                else:
                    await loop.run_in_executor(None, screen.capture.close)
        except Exception as exc:
            await screen.status_pv.write("Error")
            print(f"{screen.name}: {exc}")
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
        loop = asyncio.get_running_loop()
        try:
            for image_index in range(max(num_images.values())):
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
                        await loop.run_in_executor(
                            None,
                            lambda names=write_names, frames=images: self._append_captures(names, frames),
                        )
                    for name in write_names:
                        await self.screens[name].status_pv.write("Acquiring")
        except Exception as exc:
            for name in requested:
                await self.screens[name].status_pv.write("Error")
            print(f"XRT acquisition failed: {exc}")
            return

        for name in requested:
            await self.screens[name].status_pv.write("Idle")

    def _append_captures(self, names: list[str], images: dict[str, np.ndarray]) -> None:
        for name in names:
            screen = self.screens[name]
            screen.capture.append(screen.name, images[name])

    def _run_xrt_once(self) -> dict[str, np.ndarray]:
        self._force_histograms()
        self.raycing.run_process_from_file(self.beamline)
        images: dict[str, np.ndarray] = {}
        for name, screen in self.screens.items():
            image = getattr(screen.obj, "image", None)
            if image is None:
                continue
            arr = np.asarray(image, dtype=np.float64)
            if arr.ndim == 2 and arr.size:
                images[name] = arr.copy()
        return images

    def _force_histograms(self) -> None:
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

    async def _update_previews(self, images: dict[str, np.ndarray]) -> None:
        for name, frame in images.items():
            flat = np.asarray(frame, dtype=np.float64).ravel()
            if flat.size > self.image_max_length:
                flat = flat[: self.image_max_length]
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
            self.beamline_node.tag if self.beamline_node is not None else self.beamline.name
        )
        self.element_uuids = self._element_uuid_map()
        self.materials = self._named_object_map("Materials", "matnamesToUUIDs", "materialsDict")
        self.figure_errors = self._named_object_map("FigureErrors", "fenamesToUUIDs", "fesDict")
        self.mapping: dict[str, XmlPV] = {}
        self.screens = self._screen_states()
        self.coordinator = SimulationCoordinator(
            raycing=self.raycing,
            beamline=self.beamline,
            screens=self.screens,
            xml_path=self.xml_path,
            prefix=self.prefix,
            image_max_length=self.image_max_length,
            overwrite=self.overwrite,
        )
        self.coordinator._force_histograms()
        self.pvdb = self._build_pvdb()

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

    def _named_object_map(self, section: str, names_attr: str, dict_attr: str) -> dict[str, Any]:
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
        for screen in getattr(self.beamline, "screens", []):
            name = str(getattr(screen, "name", "") or getattr(screen, "uuid", "screen"))
            safe_name = _safe_component(name)
            if safe_name in screens:
                safe_name = f"{safe_name}_{str(getattr(screen, 'uuid', ''))[:8]}"
            screens[safe_name] = ScreenState(name=name, safe_name=safe_name, obj=screen)
        return screens

    def _build_pvdb(self) -> dict[str, Any]:
        specs = [*self._xml_pv_specs(), *self._screen_pv_specs()]
        pvdb = {spec.name: spec.create(group=None) for spec in specs}
        for screen in self.screens.values():
            base = f"{self.prefix}{screen.safe_name}"
            screen.acquire_pv = pvdb[f"{base}:Acquire"]
            screen.status_pv = pvdb[f"{base}:AcquireStatus"]
            screen.capture_pv = pvdb[f"{base}:Capture"]
            screen.file_path_pv = pvdb[f"{base}:FilePath"]
            screen.file_name_pv = pvdb[f"{base}:FileName"]
            screen.num_images_pv = pvdb[f"{base}:NumImages"]
            screen.image_pv = pvdb[f"{base}:Image"]
        return pvdb

    def _xml_pv_specs(self) -> list[PVSpec]:
        entries: list[tuple[tuple[str, ...], ET.Element, str | None, int | None, Any]] = []
        for path, element in _iter_xml_param_paths(self.root):
            raw_text = "" if element.text is None else element.text.strip()
            parsed = _parse_text(raw_text)
            values = _compound_values(path[-1], raw_text, parsed)
            if values is None:
                entries.append((path, element, None, None, parsed))
                continue
            for index, (field_name, value) in enumerate(zip(COMPOUND_FIELDS[path[-1]], values)):
                entries.append((path, element, field_name, index, value))

        dropped_suffixes = []
        full_suffixes = []
        for path, _element, field_name, _index, _value in entries:
            dropped = _path_parts(path, drop_structural=True)
            full = _path_parts(path, drop_structural=False)
            if field_name is not None:
                dropped = (*dropped, field_name)
                full = (*full, field_name)
            dropped_suffixes.append(_suffix_from_parts(dropped))
            full_suffixes.append(_suffix_from_parts(full))

        counts = Counter(dropped_suffixes)
        used: Counter[str] = Counter()
        specs = []
        for entry, dropped, full in zip(entries, dropped_suffixes, full_suffixes):
            path, element, field_name, field_index, value = entry
            suffix = dropped if counts[dropped] == 1 else full
            used[suffix] += 1
            if used[suffix] > 1:
                suffix = f"{suffix}_{used[suffix]}"

            xml_pv = XmlPV(
                suffix=suffix,
                path=path,
                element=element,
                value=value,
                string_pv=_string_pv_required(value),
                field_name=field_name,
                field_index=field_index,
            )
            self._attach_live_target(xml_pv)
            self._set_initial_live_value(xml_pv)
            self.mapping[suffix] = xml_pv
            specs.append(self._config_spec(xml_pv))
        return specs

    def _set_initial_live_value(self, xml_pv: XmlPV) -> None:
        if xml_pv.live_kind is None:
            return
        try:
            value = self._read_live_value(xml_pv)
        except Exception:
            xml_pv.live_kind = None
            xml_pv.read_only = True
            return
        if isinstance(value, np.generic):
            value = value.item()
        xml_pv.value = value
        xml_pv.string_pv = _string_pv_required(value)
        xml_pv.read_only = False

    def _config_spec(self, xml_pv: XmlPV) -> PVSpec:
        async def getter(instance, *, xml_pv=xml_pv):
            return self._pv_readback_value(xml_pv, self._read_live_value(xml_pv))

        async def putter(instance, value, *, xml_pv=xml_pv):
            return self._write_live_pv(xml_pv, value)

        value = self._pv_readback_value(xml_pv, xml_pv.value)
        get = getter if not xml_pv.read_only else None
        put = putter if not xml_pv.read_only else None
        if xml_pv.string_pv:
            return PVSpec(
                name=self.prefix + xml_pv.suffix,
                value=value,
                dtype=str,
                get=get,
                put=put,
                read_only=xml_pv.read_only,
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
                get=get,
                put=put,
                read_only=xml_pv.read_only,
                doc=f"XML {xml_pv.xml_path}; raw XML value {xml_pv.raw_text!r}",
            )
        if self._should_use_integer_pv(xml_pv) and isinstance(value, (int, float)):
            return PVSpec(
                name=self.prefix + xml_pv.suffix,
                value=int(value),
                dtype=int,
                get=get,
                put=put,
                read_only=xml_pv.read_only,
                doc=f"XML {xml_pv.xml_path}; raw XML value {xml_pv.raw_text!r}",
            )
        return PVSpec(
            name=self.prefix + xml_pv.suffix,
            value=float(value),
            dtype=float,
            get=get,
            put=put,
            read_only=xml_pv.read_only,
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
        if xml_pv.live_kind is None:
            return isinstance(xml_pv.value, int) and not isinstance(xml_pv.value, bool)
        attr = xml_pv.attr or xml_pv.arg or xml_pv.path[-1]
        if attr in DISCRETE_INTEGER_FIELDS:
            return True
        if xml_pv.field_name is not None and attr == "histShape":
            return True
        return False

    def _screen_pv_specs(self) -> list[PVSpec]:
        specs: list[PVSpec] = []
        for screen in self.screens.values():
            base = screen.safe_name

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
                        record="stringout",
                        max_length=4096,
                        cls_kwargs=STRING_KWARGS,
                        doc="Directory used when Capture changes to 1",
                    ),
                    PVSpec(
                        name=self.prefix + f"{base}:FileName",
                        value=f"{base}.h5",
                        dtype=str,
                        record="stringout",
                        max_length=1024,
                        cls_kwargs=STRING_KWARGS,
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

    def _attach_live_target(self, xml_pv: XmlPV) -> None:
        path = xml_pv.path
        if len(path) < 3 or path[0] != "Project":
            return
        section = path[1]

        if section == self.beamline_name:
            if len(path) == 4 and path[2] == "properties":
                self._set_attr_live(xml_pv, self.beamline, path[3])
                return
            if len(path) < 5:
                return
            oeid = self.element_uuids.get(path[2])
            if oeid is None:
                return
            target = self.beamline.oesDict[oeid][0]
            if len(path) == 5 and path[3] == "properties":
                self._set_attr_live(xml_pv, target, path[4])
                return
            if len(path) == 6 and path[4] == "parameters":
                xml_pv.live_kind = "flow"
                xml_pv.target = target
                xml_pv.oeid = oeid
                xml_pv.method = path[3]
                xml_pv.arg = path[5]
                return

        if section == "Materials" and len(path) == 5 and path[3] == "properties":
            target = self.materials.get(path[2])
            if target is not None:
                self._set_attr_live(xml_pv, target, path[4])
        elif section == "FigureErrors" and len(path) == 5 and path[3] == "properties":
            target = self.figure_errors.get(path[2])
            if target is not None:
                self._set_attr_live(xml_pv, target, path[4])

    def _set_attr_live(self, xml_pv: XmlPV, target: Any, attr: str) -> None:
        if attr in REF_OR_STRUCTURAL_ATTRS:
            return
        xml_pv.live_kind = "attr"
        xml_pv.target = target
        xml_pv.attr = attr

    def _read_live_value(self, xml_pv: XmlPV) -> Any:
        if xml_pv.live_kind == "attr":
            return self._read_live_attr(xml_pv)
        if xml_pv.live_kind == "flow":
            return self._read_live_flow(xml_pv)
        return xml_pv.value

    def _read_live_attr(self, xml_pv: XmlPV) -> Any:
        if xml_pv.attr is None or xml_pv.target is None:
            return xml_pv.value
        value = getattr(xml_pv.target, xml_pv.attr)
        if xml_pv.field_index is not None:
            value = value[xml_pv.field_index]
        if isinstance(value, np.generic):
            return value.item()
        return value

    def _read_live_flow(self, xml_pv: XmlPV) -> Any:
        methods = self.beamline.flowU.get(xml_pv.oeid, {})
        kwargs = methods.get(xml_pv.method)
        if kwargs is None or xml_pv.arg is None:
            return xml_pv.value
        value = kwargs.get(xml_pv.arg, xml_pv.value)
        if xml_pv.arg == "beam":
            for beam_name, beam_tag in self.beamline.beamNamesDict.items():
                if beam_tag[0] == value:
                    value = beam_name
                    break
        if xml_pv.field_index is not None:
            value = value[xml_pv.field_index]
        if isinstance(value, np.generic):
            return value.item()
        return value

    def _write_live_pv(self, xml_pv: XmlPV, value: Any) -> Any:
        value = _coerce_put_value(value)
        try:
            if xml_pv.live_kind == "attr":
                self._write_live_attr(xml_pv, value)
            elif xml_pv.live_kind == "flow":
                self._write_live_flow(xml_pv, value)
        except Exception as exc:
            print(f"Could not update live XRT binding for {xml_pv.suffix}: {exc}")
            raise
        readback = self._read_live_value(xml_pv)
        xml_pv.value = readback
        return self._pv_readback_value(xml_pv, readback)

    def _write_live_attr(self, xml_pv: XmlPV, value: Any) -> None:
        attr = xml_pv.attr
        target = xml_pv.target
        if attr is None or target is None:
            return
        if xml_pv.field_index is None:
            setattr(target, attr, self._xrt_value(value))
            return

        current = getattr(target, attr)
        if isinstance(current, dict):
            current[xml_pv.field_name] = value
            setattr(target, attr, current)
            return

        try:
            values = list(current)
        except TypeError:
            values = []
        while len(values) <= xml_pv.field_index:
            values.append(0)
        values[xml_pv.field_index] = self._xrt_value(value)
        setattr(target, attr, self._named_value(attr, values))

    def _write_live_flow(self, xml_pv: XmlPV, value: Any) -> None:
        methods = self.beamline.flowU.get(xml_pv.oeid, {})
        kwargs = methods.get(xml_pv.method)
        if kwargs is None or xml_pv.arg is None:
            return
        if xml_pv.field_index is None:
            kwargs[xml_pv.arg] = self._flow_value(xml_pv.arg, value)
            return

        values = list(kwargs.get(xml_pv.arg, []))
        while len(values) <= xml_pv.field_index:
            values.append(None)
        values[xml_pv.field_index] = self._xrt_value(value)
        kwargs[xml_pv.arg] = values

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
        if attr.startswith("limPhys") and all(not isinstance(value, str) for value in values):
            return self.raycing.Limits(values)
        if attr == "histShape":
            return self.raycing.Image2D([int(value) for value in values])
        return values


def _pv_value(value: Any, string_pv: bool) -> Any:
    if string_pv:
        return _format_text(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


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
