#!/usr/bin/env python3
"""Serve a small caproto IOC from an XRT XML beamline.

Usage
-----
python -m caproto.ioc_examples.xrt_xml_ioc --xml beamline.xml --prefix XRT:

This example parses the XML directly for configurable PVs and loads the same
file into XRT for live, in-memory ray tracing. It does not modify the XML file
on disk.
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
STATUS_STRINGS = ["Idle", "Acquiring", "Writing", "Error"]
BINARY_STRINGS = ["Off", "On"]
STRING_KWARGS = dict(string_encoding="utf-8", report_as_string=True)
STRUCTURAL_COMPONENTS = {"properties", "parameters"}
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
NUMERIC_RE = re.compile(
    r"^[+-]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[eE][+-]?\d+)?$"
)
INTEGER_RE = re.compile(r"^[+-]?\d+$")


@dataclass
class XmlParam:
    path: tuple[str, ...]
    element: ET.Element
    raw_text: str
    parsed_value: Any
    bindings: list["PVBinding"] = field(default_factory=list)


@dataclass
class LiveBinding:
    kind: str
    target: Any = None
    attr: str | None = None
    beamline: Any = None
    oeid: str | None = None
    method: str | None = None
    arg: str | None = None


@dataclass
class PVBinding:
    suffix: str
    param: XmlParam
    value: Any
    live: LiveBinding | None = None
    field_name: str | None = None
    field_index: int | None = None
    string_pv: bool = False

    @property
    def xml_path(self) -> str:
        return "/".join(self.param.path)

    @property
    def raw_text(self) -> str:
        return self.param.raw_text

    @property
    def parsed_value(self) -> Any:
        if self.field_index is None:
            return self.param.parsed_value
        try:
            return self.param.parsed_value[self.field_index]
        except Exception:
            return self.value


@dataclass
class ScreenState:
    name: str
    safe_name: str
    obj: Any
    h5_path: Path | None = None
    h5_file: Any = None
    h5_dataset: Any = None
    h5_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    acquire_pv: Any = None
    status_pv: Any = None
    capture_pv: Any = None
    file_path_pv: Any = None
    file_name_pv: Any = None
    num_images_pv: Any = None
    image_pv: Any = None

    def target_h5_path(self) -> Path:
        directory = Path(str(self.file_path_pv.value)).expanduser()
        return directory / str(self.file_name_pv.value)

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

    def open_capture_sync(
        self,
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

        if self.h5_file is not None:
            return

        directory = Path(str(self.file_path_pv.value)).expanduser()
        directory.mkdir(parents=True, exist_ok=True)
        self.h5_path = self.target_h5_path()

        mode = "w" if overwrite else "x"
        height, width = self.image_shape()
        self.h5_file = h5py.File(self.h5_path, mode)
        self.h5_file.attrs["source_xml"] = source_xml
        self.h5_file.attrs["pv_prefix"] = pv_prefix
        self.h5_file.attrs["xrt_beamline_name"] = beamline_name
        self.h5_file.attrs["created"] = time.time()

        group = self.h5_file.require_group(f"/entry/screens/{self.safe_name}")
        group.attrs["screen_name"] = self.name
        group.attrs["source_xml"] = source_xml
        group.attrs["pv_prefix"] = pv_prefix
        group.attrs["timestamp"] = time.time()
        group.attrs["xrt_beamline_name"] = beamline_name
        self.h5_dataset = group.create_dataset(
            "image",
            shape=(0, height, width),
            maxshape=(None, height, width),
            chunks=(1, height, width),
            dtype="float64",
            compression="lzf",
        )

    def append_frame_sync(self, frame: np.ndarray) -> None:
        if self.h5_dataset is None:
            return
        if tuple(self.h5_dataset.shape[1:]) != tuple(frame.shape):
            raise RuntimeError(
                f"{self.name} image shape changed from "
                f"{self.h5_dataset.shape[1:]} to {frame.shape}; close and "
                "reopen Capture to create a new dataset"
            )
        index = self.h5_dataset.shape[0]
        self.h5_dataset.resize((index + 1, *frame.shape))
        self.h5_dataset[index, :, :] = frame
        self.h5_file.flush()

    def close_capture_sync(self) -> None:
        if self.h5_file is not None:
            self.h5_file.close()
        self.h5_file = None
        self.h5_dataset = None


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
    if NUMERIC_RE.match(text):
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


def _coerce_put_value(value: Any) -> Any:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, np.ndarray):
        if value.size == 1:
            value = value.item()
        else:
            value = value.tolist()
    if isinstance(value, str):
        return _parse_text(value)
    return value


def _bool_value(value: Any) -> bool:
    value = _coerce_put_value(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "on", "yes", "true"}
    return bool(value)


def _format_text(value: Any) -> str:
    if value is None:
        return "None"
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, str):
        return value
    return repr(value)


def _is_scalar(value: Any) -> bool:
    return isinstance(value, (bool, int, float, str, type(None), np.number))


def _string_pv_required(value: Any) -> bool:
    if isinstance(value, (bool, int, float, np.number)) and not isinstance(value, bool):
        return False
    if isinstance(value, bool):
        return False
    return True


def _safe_component(part: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_-]+", "_", str(part).strip())
    text = text.strip("_")
    return text or "item"


def _suffix_from_parts(parts: tuple[str, ...]) -> str:
    return ":".join(_safe_component(part) for part in parts if part)


def _parts_for_path(path: tuple[str, ...], *, drop_structural: bool) -> tuple[str, ...]:
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


def _iter_xml_params(root: ET.Element) -> list[XmlParam]:
    params: list[XmlParam] = []

    def walk(node: ET.Element, path: tuple[str, ...]) -> None:
        if node.attrib.get("type") == "param":
            raw_text = "" if node.text is None else node.text.strip()
            params.append(XmlParam(path, node, raw_text, _parse_text(raw_text)))
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


def _pv_value(value: Any, string_pv: bool) -> Any:
    if string_pv:
        return _format_text(value)
    if isinstance(value, np.generic):
        value = value.item()
    return value


def _named_value(raycing: Any, attr: str, values: list[Any]) -> Any:
    if attr == "center":
        return raycing.Center(values)
    if attr.startswith("limPhys"):
        return raycing.Limits(values)
    if attr == "histShape":
        return raycing.Image2D([int(value) for value in values])
    return values


def _coerce_live_reference(raycing: Any, beamline: Any, attr: str, value: Any) -> Any:
    ref_kind_for_arg = getattr(raycing, "ref_kind_for_arg", None)
    normalize_ref = getattr(raycing, "normalize_ref", None)
    if ref_kind_for_arg is None or normalize_ref is None:
        return value
    ref_kind = ref_kind_for_arg(attr)
    if ref_kind is None:
        return value
    return normalize_ref(value, beamline, ref_kind, target="object")


class SimulationCoordinator:
    """Own all XRT execution and HDF5 writes for screen Acquire PVs."""

    # Acquire requests arriving during this short window are coalesced into one
    # multi-screen simulation batch. Requests after the batch starts are rejected.
    request_coalesce_s = 0.05

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
        self._lock = asyncio.Lock()
        self._capture_lock = asyncio.Lock()
        self._busy = False
        self._pending: set[str] = set()
        self._pending_task: asyncio.Task | None = None

    async def request(self, screen_name: str) -> bool:
        async with self._lock:
            if self._busy:
                state = self.screens[screen_name]
                await state.status_pv.write("Error")
                print(f"{screen_name}: rejected Acquire while simulation is busy")
                return False
            self._pending.add(screen_name)
            if self._pending_task is None or self._pending_task.done():
                self._pending_task = asyncio.create_task(self._run_pending())
        return True

    async def set_capture(self, state: ScreenState, enabled: bool) -> bool:
        loop = asyncio.get_running_loop()
        try:
            async with self._capture_lock:
                async with state.h5_lock:
                    if enabled:
                        target = state.target_h5_path()
                        for other in self.screens.values():
                            if other is not state and other.h5_file is not None and other.h5_path == target:
                                raise RuntimeError(
                                    f"{target} is already open for {other.name}; "
                                    "each screen must capture to its own HDF5 file"
                                )
                        await loop.run_in_executor(
                            None,
                            lambda: state.open_capture_sync(
                                source_xml=str(self.xml_path),
                                pv_prefix=self.prefix,
                                beamline_name=str(getattr(self.beamline, "name", "")),
                                overwrite=self.overwrite,
                            ),
                        )
                    else:
                        await loop.run_in_executor(None, state.close_capture_sync)
        except Exception as exc:
            await state.status_pv.write("Error")
            print(f"{state.name}: {exc}")
            return False
        if not enabled and state.status_pv.value != "Error":
            await state.status_pv.write("Idle")
        return True

    async def close_all(self) -> None:
        loop = asyncio.get_running_loop()
        for state in self.screens.values():
            async with state.h5_lock:
                await loop.run_in_executor(None, state.close_capture_sync)

    async def _run_pending(self) -> None:
        await asyncio.sleep(self.request_coalesce_s)
        async with self._lock:
            requested = set(self._pending)
            self._pending.clear()
            self._busy = True
        try:
            await self._run_requests(requested)
        finally:
            async with self._lock:
                self._busy = False

    async def _run_requests(self, requested: set[str]) -> None:
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
                for name in requested:
                    if image_index >= num_images[name] or name not in images:
                        continue
                    state = self.screens[name]
                    if not _bool_value(state.capture_pv.value):
                        continue
                    await state.status_pv.write("Writing")
                    async with state.h5_lock:
                        await loop.run_in_executor(
                            None, lambda s=state, frame=images[name]: s.append_frame_sync(frame)
                        )
                    await state.status_pv.write("Acquiring")
        except Exception as exc:
            for name in requested:
                await self.screens[name].status_pv.write("Error")
            print(f"XRT acquisition failed: {exc}")
            return

        for name in requested:
            await self.screens[name].status_pv.write("Idle")

    def _run_xrt_once(self) -> dict[str, np.ndarray]:
        self._force_histograms()
        self.raycing.run_process_from_file(self.beamline)
        images: dict[str, np.ndarray] = {}
        for name, state in self.screens.items():
            image = getattr(state.obj, "image", None)
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
            state = self.screens[name]
            flat = np.asarray(frame, dtype=np.float64).ravel()
            if flat.size > self.image_max_length:
                flat = flat[: self.image_max_length]
            await state.image_pv.write(flat, verify_value=False)


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
        self.beamline_name = self.beamline_node.tag if self.beamline_node is not None else self.beamline.name
        self.params = _iter_xml_params(self.root)
        self.element_uuids = self._element_uuid_map()
        self.materials = self._named_object_map("Materials", "matnamesToUUIDs", "materialsDict")
        self.figure_errors = self._named_object_map("FigureErrors", "fenamesToUUIDs", "fesDict")
        self.mapping: dict[str, PVBinding] = {}
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
        specs: list[PVSpec] = []
        specs.extend(self._xml_pv_specs())
        specs.extend(self._screen_pv_specs())
        pvdb = {spec.name: spec.create(group=None) for spec in specs}
        for state in self.screens.values():
            base = f"{self.prefix}{state.safe_name}"
            state.acquire_pv = pvdb[f"{base}:Acquire"]
            state.status_pv = pvdb[f"{base}:AcquireStatus"]
            state.capture_pv = pvdb[f"{base}:Capture"]
            state.file_path_pv = pvdb[f"{base}:FilePath"]
            state.file_name_pv = pvdb[f"{base}:FileName"]
            state.num_images_pv = pvdb[f"{base}:NumImages"]
            state.image_pv = pvdb[f"{base}:Image"]
        return pvdb

    def _xml_pv_specs(self) -> list[PVSpec]:
        entries = []
        for param in self.params:
            tag = param.path[-1]
            values = _compound_values(tag, param.raw_text, param.parsed_value)
            if values is None:
                entries.append((param, None, None, param.parsed_value))
                continue
            for index, (field_name, value) in enumerate(zip(COMPOUND_FIELDS[tag], values)):
                entries.append((param, field_name, index, value))

        dropped_suffixes = []
        full_suffixes = []
        for param, field_name, _index, _value in entries:
            dropped_parts = _parts_for_path(param.path, drop_structural=True)
            full_parts = _parts_for_path(param.path, drop_structural=False)
            if field_name is not None:
                dropped_parts = (*dropped_parts, field_name)
                full_parts = (*full_parts, field_name)
            dropped_suffixes.append(_suffix_from_parts(dropped_parts))
            full_suffixes.append(_suffix_from_parts(full_parts))

        dropped_counts = Counter(dropped_suffixes)
        used: Counter[str] = Counter()
        specs = []
        for entry, dropped_suffix, full_suffix in zip(entries, dropped_suffixes, full_suffixes):
            param, field_name, field_index, value = entry
            suffix = dropped_suffix if dropped_counts[dropped_suffix] == 1 else full_suffix
            used[suffix] += 1
            if used[suffix] > 1:
                suffix = f"{suffix}_{used[suffix]}"

            string_pv = _string_pv_required(value)
            binding = PVBinding(
                suffix=suffix,
                param=param,
                value=value,
                live=self._live_binding_for(param.path),
                field_name=field_name,
                field_index=field_index,
                string_pv=string_pv,
            )
            param.bindings.append(binding)
            self.mapping[suffix] = binding
            specs.append(self._config_spec(binding))
        return specs

    def _config_spec(self, binding: PVBinding) -> PVSpec:
        async def putter(instance, value, *, binding=binding):
            return self._write_config(binding, value)

        value = _pv_value(binding.value, binding.string_pv)
        if binding.string_pv:
            return PVSpec(
                name=self.prefix + binding.suffix,
                value=value,
                dtype=str,
                put=putter,
                max_length=4096,
                cls_kwargs=STRING_KWARGS,
                doc=f"XML {'/'.join(binding.param.path)}",
            )
        if isinstance(value, bool):
            return PVSpec(
                name=self.prefix + binding.suffix,
                value=value,
                dtype=bool,
                record="bo",
                put=putter,
                doc=f"XML {'/'.join(binding.param.path)}",
            )
        if isinstance(value, int) and not isinstance(value, bool):
            return PVSpec(
                name=self.prefix + binding.suffix,
                value=value,
                dtype=int,
                put=putter,
                doc=f"XML {'/'.join(binding.param.path)}",
            )
        return PVSpec(
            name=self.prefix + binding.suffix,
            value=float(value),
            dtype=float,
            put=putter,
            doc=f"XML {'/'.join(binding.param.path)}",
        )

    def _screen_pv_specs(self) -> list[PVSpec]:
        specs: list[PVSpec] = []
        for state in self.screens.values():
            base = state.safe_name

            async def acquire_putter(instance, value, *, state=state):
                if _bool_value(value):
                    await self.coordinator.request(state.safe_name)
                await instance.write("Off", verify_value=False)
                return "Off"

            async def capture_putter(instance, value, *, state=state):
                enabled = _bool_value(value)
                ok = await self.coordinator.set_capture(state, enabled)
                return "On" if enabled and ok else "Off"

            async def num_images_putter(instance, value):
                return max(1, int(_coerce_put_value(value)))

            async def shutdown(instance, async_lib):
                await self.coordinator.close_all()

            screen_specs = [
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
                    doc="Open or close the screen HDF5 capture file",
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
            for spec in screen_specs:
                specs.append(spec)
        return specs

    def _live_binding_for(self, path: tuple[str, ...]) -> LiveBinding | None:
        if len(path) < 3 or path[0] != "Project":
            return None
        section = path[1]
        if section == self.beamline_name:
            if len(path) == 4 and path[2] == "properties":
                return LiveBinding("object_attr", self.beamline, path[3], self.beamline)
            if len(path) >= 5:
                element_key = path[2]
                oeid = self.element_uuids.get(element_key)
                if oeid is None:
                    return None
                target = self.beamline.oesDict[oeid][0]
                if len(path) == 5 and path[3] == "properties":
                    if path[4] in {"bl", "uuid"}:
                        return None
                    return LiveBinding("object_attr", target, path[4], self.beamline)
                if len(path) == 6 and path[4] == "parameters":
                    return LiveBinding(
                        "flow_arg",
                        target=target,
                        beamline=self.beamline,
                        oeid=oeid,
                        method=path[3],
                        arg=path[5],
                    )
        if section == "Materials" and len(path) == 5 and path[3] == "properties":
            target = self.materials.get(path[2])
            if target is not None and path[4] not in {"bl", "uuid"}:
                return LiveBinding("object_attr", target, path[4], self.beamline)
        if section == "FigureErrors" and len(path) == 5 and path[3] == "properties":
            target = self.figure_errors.get(path[2])
            if target is not None and path[4] not in {"bl", "uuid"}:
                return LiveBinding("object_attr", target, path[4], self.beamline)
        return None

    def _write_config(self, binding: PVBinding, value: Any) -> Any:
        parsed = _coerce_put_value(value)
        if binding.field_index is None:
            binding.param.parsed_value = parsed
            binding.param.raw_text = _format_text(parsed)
            binding.param.element.text = binding.param.raw_text
            binding.value = parsed
            self._write_live(binding, parsed)
            return _pv_value(parsed, binding.string_pv)

        current = binding.param.parsed_value
        if not isinstance(current, (list, tuple)):
            values = _compound_values(binding.param.path[-1], binding.param.raw_text, current) or []
        else:
            values = list(current)
        while len(values) <= binding.field_index:
            values.append(None)
        values[binding.field_index] = parsed
        binding.param.parsed_value = tuple(values) if isinstance(current, tuple) else values
        binding.param.raw_text = _format_text(binding.param.parsed_value)
        binding.param.element.text = binding.param.raw_text
        for sibling in binding.param.bindings:
            if sibling.field_index is not None and sibling.field_index < len(values):
                sibling.value = values[sibling.field_index]
        self._write_live(binding, parsed)
        return _pv_value(parsed, binding.string_pv)

    def _write_live(self, binding: PVBinding, value: Any) -> None:
        live = binding.live
        if live is None:
            return
        try:
            if live.kind == "object_attr" and live.attr is not None:
                self._write_object_attr(live, binding, value)
            elif live.kind == "flow_arg":
                self._write_flow_arg(live, binding, value)
        except Exception as exc:
            print(f"Could not update live XRT binding for {binding.suffix}: {exc}")

    def _write_object_attr(self, live: LiveBinding, binding: PVBinding, value: Any) -> None:
        attr = live.attr
        target = live.target
        if binding.field_index is None:
            value = _coerce_live_reference(self.raycing, self.beamline, attr, value)
            setattr(target, attr, value)
            return

        current = getattr(target, attr)
        if isinstance(current, dict):
            current[binding.field_name] = value
            setattr(target, attr, current)
            return
        try:
            values = list(current)
        except TypeError:
            values = list(binding.param.parsed_value)
        while len(values) <= binding.field_index:
            values.append(0)
        values[binding.field_index] = value
        setattr(target, attr, _named_value(self.raycing, attr, values))

    def _write_flow_arg(self, live: LiveBinding, binding: PVBinding, value: Any) -> None:
        methods = self.beamline.flowU.get(live.oeid, {})
        kwargs = methods.get(live.method)
        if kwargs is None:
            return
        if binding.field_index is None:
            kwargs[live.arg] = self._flow_value(live.arg, value)
            return
        values = list(kwargs.get(live.arg, binding.param.parsed_value))
        while len(values) <= binding.field_index:
            values.append(None)
        values[binding.field_index] = value
        kwargs[live.arg] = values

    def _flow_value(self, arg: str, value: Any) -> Any:
        if arg != "beam":
            return value
        if value in {None, "None", ""}:
            return None
        if self.raycing.is_valid_uuid(value):
            return value
        beam_tag = self.beamline.beamNamesDict.get(str(value))
        return beam_tag[0] if beam_tag is not None else value


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
