"""Vendored (near-verbatim) from pyvistra.readers.imaris -- self-contained
(only h5py/numpy), copied directly rather than depending on pyvistra so
spt-pipeline can eventually stand alone outside the microscopy workspace.
"""

import re
from datetime import datetime

import h5py
import numpy as np


class ImarisReader:
    """
    A convenient and efficient reader for Imaris (.ims) HDF5 files.

    Attributes:
        filepath (str): Path to the .ims file.
        dtype (np.dtype): Data type of the image.
        shape (tuple): Dimensions (Time, Channels, Z, Y, X).
        voxel_size (tuple): (Z, Y, X) voxel size, in `length_unit` (usually microns).
        voxel_size_known (bool): whether the file really recorded the extents
            `voxel_size` is derived from, as opposed to it being a default.
        length_unit (str | None): unit the extents are recorded in, if the file says.
        timestamps (list): List of datetime objects for each time point.
        channels_info (list): List of dicts containing metadata (Name, Wavelengths, Exposure).
        resolution_levels (int): Number of resolution levels available.
        n_channels (int): number of channels

    Methods:
        read(c=0, t=0, z=None, res_level=0)
    """

    def __init__(self, filepath):
        self.filepath = filepath
        self._file = h5py.File(filepath, "r")

        # Initialize containers
        self.voxel_size = (1.0, 1.0, 1.0)
        # Whether `voxel_size` was computed from extents actually present
        # in the file, as opposed to the (0.0, 1.0) defaults below -- see
        # `_parse_metadata`. False means "no calibration", which a caller
        # cannot otherwise tell from the plausible-looking 1/width voxel
        # size those defaults produce (spt_pipeline.io_formats._load_ims).
        self.voxel_size_known = False
        # The unit the extents (and so `voxel_size`) are in, as recorded
        # in DataSetInfo/Image's "Unit" attribute. Usually "um"; None when
        # the file doesn't say.
        self.length_unit = None
        self.timestamps = []
        self.channels_info = []

        # Run setup
        self._scan_structure()
        self._parse_metadata()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        if self._file:
            self._file.close()
            self._file = None

    def _decode_imaris_attribute(self, value):
        """
        Decodes the specific Imaris byte-array attribute format into a standard string.
        Examples:
            [b'6' b'0' b'0'] -> "600"
            "MyImage" -> "MyImage"
        """
        # Case 1: Already string/bytes scalar
        if isinstance(value, bytes):
            return value.decode("utf-8")
        if isinstance(value, str):
            return value

        # Case 2: Numpy array or list (The Imaris "Character Array")
        if isinstance(value, (np.ndarray, list, tuple)):
            val_array = np.array(value).flatten()
            if val_array.size == 0:
                return ""

            # Bytes/Strings elements
            if val_array.dtype.kind in ("S", "U") or isinstance(
                val_array[0], (bytes, np.bytes_)
            ):
                chars = []
                for x in val_array:
                    if isinstance(x, (bytes, np.bytes_)):
                        chars.append(x.decode("utf-8"))
                    else:
                        chars.append(str(x))
                return "".join(chars)

            # Integer elements (ASCII codes)
            if val_array.dtype.kind in ("i", "u"):
                return "".join([chr(x) for x in val_array])

            # Numeric elements (e.g. float array), return first as string
            return str(val_array[0])

        return str(value)

    def _get_val(self, grp, key, type_func=str):
        """
        Extracts attribute. Tries to cast to `type_func`.
        If casting fails (e.g. converting "600 nm" to float), returns the String.
        """
        if grp is None or key not in grp.attrs:
            return None

        raw_val = grp.attrs[key]
        str_val = self._decode_imaris_attribute(raw_val)

        if not str_val:
            return None

        # Try strict cast
        try:
            if type_func is bool:
                return str_val.lower() == "true"
            return type_func(str_val)
        except (ValueError, TypeError):
            # If cast fails (e.g. "600 nm"), return the original string
            return str_val

    def _group_for_index(self, parent_group, base_name, index):
        """Return child group matching `base_name <index>` or `base_name<index>`."""
        exact_names = (f"{base_name} {index}", f"{base_name}{index}")
        for name in exact_names:
            if name in parent_group:
                return parent_group[name]

        pat = re.compile(rf"^{re.escape(base_name)}\s*(\d+)$")
        for name in parent_group.keys():
            m = pat.match(name)
            if m and int(m.group(1)) == int(index):
                return parent_group[name]

        return None

    def _read_image_size_attrs(self, channel_group, data_node):
        """Read logical image size from channel attrs, with safe fallbacks."""
        sx = self._get_val(channel_group, "ImageSizeX", int)
        sy = self._get_val(channel_group, "ImageSizeY", int)
        sz = self._get_val(channel_group, "ImageSizeZ", int)

        # Some writers store these on the Data dataset instead.
        if not isinstance(sx, (int, float)):
            sx = self._get_val(data_node, "ImageSizeX", int)
        if not isinstance(sy, (int, float)):
            sy = self._get_val(data_node, "ImageSizeY", int)
        if not isinstance(sz, (int, float)):
            sz = self._get_val(data_node, "ImageSizeZ", int)

        # Final fallback: use actual allocated dataset shape.
        if not isinstance(sx, (int, float)):
            sx = data_node.shape[-1]
        if not isinstance(sy, (int, float)):
            sy = data_node.shape[-2]
        if not isinstance(sz, (int, float)):
            sz = data_node.shape[-3] if data_node.ndim >= 3 else 1

        return int(sz), int(sy), int(sx)

    def _scan_structure(self):
        """Dynamically scans the HDF5 hierarchy to determine dimensions."""
        dataset_root = self._file["DataSet"]

        # 1. Identify Resolution Levels
        self._res_groups = sorted(
            [
                k
                for k in dataset_root.keys()
                if "ResolutionLevel" in k and re.search(r"\d+", k)
            ],
            key=lambda x: int(re.search(r"\d+", x).group()),
        )
        self.resolution_levels = len(self._res_groups)

        if self.resolution_levels == 0:
            raise ValueError("No ResolutionLevels found in Imaris file.")

        # 2. Inspect Level 0 to find Time and Channels
        l0 = dataset_root[self._res_groups[0]]
        time_groups = [k for k in l0.keys() if "TimePoint" in k]

        if not time_groups:
            raise ValueError("No TimePoints found in Imaris file.")

        self.n_timepoints = len(time_groups)

        t0 = self._group_for_index(l0, "TimePoint", 0)
        if t0 is None:
            raise ValueError("TimePoint 0 not found in Imaris file.")

        channel_groups = [k for k in t0.keys() if "Channel" in k]
        self.n_channels = len(channel_groups)
        if self.n_channels == 0:
            raise ValueError("No Channels found in Imaris file.")

        # 3. Get Image Dimensions
        c0 = self._group_for_index(t0, "Channel", 0)
        if c0 is None or "Data" not in c0:
            raise ValueError("Channel 0/Data not found in Imaris file.")

        data_node = c0["Data"]
        self.dtype = data_node.dtype

        self.size_z, self.size_y, self.size_x = self._read_image_size_attrs(
            c0, data_node
        )
        self.shape = (
            self.n_timepoints,
            self.n_channels,
            self.size_z,
            self.size_y,
            self.size_x,
        )

    def _parse_metadata(self):
        """Parses global metadata from /DataSetInfo."""
        info_group = self._file.get("DataSetInfo")
        if not info_group:
            return

        # --- Voxel Size ---
        # Helper: ensures we have a float for math, defaulting to 0.0/1.0 if we get a string/None
        def ensure_float(v, default=0.0):
            if isinstance(v, (float, int)):
                return float(v)
            # If it's a string like "1.5", conversion works. If "1.5 um", it fails -> returns default.
            try:
                return float(v)
            except Exception:
                return default

        img_info = info_group.get("Image")

        # Extents usually come as pure numbers ("0.0", "1024.5")
        min_x = ensure_float(self._get_val(img_info, "ExtMin0", float), 0.0)
        min_y = ensure_float(self._get_val(img_info, "ExtMin1", float), 0.0)
        min_z = ensure_float(self._get_val(img_info, "ExtMin2", float), 0.0)
        max_x = ensure_float(self._get_val(img_info, "ExtMax0", float), 1.0)
        max_y = ensure_float(self._get_val(img_info, "ExtMax1", float), 1.0)
        max_z = ensure_float(self._get_val(img_info, "ExtMax2", float), 1.0)

        vox_x = (max_x - min_x) / self.size_x if self.size_x > 0 else 1.0
        vox_y = (max_y - min_y) / self.size_y if self.size_y > 0 else 1.0
        vox_z = (max_z - min_z) / self.size_z if self.size_z > 0 else 1.0
        self.voxel_size = (vox_z, vox_y, vox_x)

        # The lateral extents are what a pixel size is derived from, so
        # "known" means those four were really in the file: with any of
        # them absent the defaults above give a voxel size of 1/width,
        # which looks like a calibration and isn't.
        lateral = [
            self._get_val(img_info, key, float)
            for key in ("ExtMin0", "ExtMin1", "ExtMax0", "ExtMax1")
        ]
        self.voxel_size_known = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in lateral)
        unit = self._get_val(img_info, "Unit", str)
        self.length_unit = unit.strip() if isinstance(unit, str) and unit.strip() else None

        # --- Timestamps ---
        time_info = info_group.get("TimeInfo")
        if time_info:
            for i in range(self.n_timepoints):
                ts_str = None
                keys = [f"TimePoint{i + 1}", f"TimePoint {i + 1}"]
                for k in keys:
                    val = self._get_val(time_info, k, str)
                    if val:
                        ts_str = val
                        break

                if ts_str:
                    formats = ["%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"]
                    dt = None
                    for fmt in formats:
                        try:
                            dt = datetime.strptime(ts_str.strip(), fmt)
                            break
                        except ValueError:
                            continue
                    self.timestamps.append(dt)
                else:
                    self.timestamps.append(None)

        # --- Channel Info ---
        self.channels_info = []
        for i in range(self.n_channels):
            c_grp = info_group.get(f"Channel {i}")
            info = {
                "id": i,
                "name": f"Channel {i}",
                "emission_wavelength": None,
                "excitation_wavelength": None,
                "exposure_time": None,
            }

            if c_grp:
                info["name"] = (
                    self._get_val(c_grp, "Name", str) or f"Channel {i}"
                )

                # We try to get float, but if it is "600 nm", it returns "600 nm" (string)
                # We prefer LSMEmissionWavelength if available
                em = self._get_val(c_grp, "LSMEmissionWavelength", float)
                if em is None:
                    em = self._get_val(c_grp, "EmissionWavelength", float)
                info["emission_wavelength"] = em

                ex = self._get_val(c_grp, "LSMExcitationWavelength", float)
                if ex is None:
                    ex = self._get_val(c_grp, "ExcitationWavelength", float)
                info["excitation_wavelength"] = ex

                info["exposure_time"] = self._get_val(
                    c_grp, "ExposureTime", float
                )

            self.channels_info.append(info)

    def read(self, c=0, t=0, z=None, res_level=0):
        """
        Reads image data.
        Args:
            c, t: Channel and Timepoint indices.
            z: Z-slice index. None for full volume.
            res_level: Resolution level (0=Full).
        """
        if res_level >= self.resolution_levels:
            raise ValueError(f"Resolution level {res_level} unavailable.")

        try:
            res_grp_name = self._res_groups[res_level]
            res_grp = self._file["DataSet"][res_grp_name]

            t_grp = self._group_for_index(res_grp, "TimePoint", t)
            if t_grp is None:
                raise ValueError(f"TimePoint {t} not found")

            c_grp = self._group_for_index(t_grp, "Channel", c)
            if c_grp is None:
                raise ValueError(f"Channel {c} not found")

            dataset = c_grp["Data"]
        except Exception as e:
            raise ValueError(f"Error locating data: {str(e)}")

        size_z, size_y, size_x = self._read_image_size_attrs(c_grp, dataset)
        size_z = min(size_z, dataset.shape[0])
        size_y = min(size_y, dataset.shape[1])
        size_x = min(size_x, dataset.shape[2])

        if z is not None:
            if z < 0 or z >= size_z:
                raise IndexError(
                    f"z index {z} out of range for size_z={size_z}"
                )
            return dataset[z, :size_y, :size_x]
        else:
            return dataset[:size_z, :size_y, :size_x]

    def __repr__(self):
        return (
            f"<ImarisReader: {self.filepath}\n"
            f"  Shape (T,C,Z,Y,X): {self.shape}\n"
            f"  Dtype: {self.dtype}\n"
            f"  Voxel Size (um): {self.voxel_size}\n"
            f"  Channels: {[c['name'] for c in self.channels_info]}>"
        )
