import h5py
import numpy as np


class Hdf5Matfile():
    """Load data from an v7.3 *.mat file. Only reading is supported, no writing.

    Usage
    -----

    To load all the variables from the file, use |load_file|:

    .. code-block:: python

        with Hdf5Matfile(filename) as file:
            data = file.load_file()

    To load a specific variable, use |load_variable|:

    .. code-block:: python

        with Hdf5Matfile(filename) as file:
            results = file.load_variable('results')

    If you're not using a context manager, make sure to close the file after
    you're done:

    .. code-block:: python

        file = Hdf5Matfile(filename)
        data = file.load_file()
        file.close()

    By default, arrays are not squeezed; since MATLAB represents even scalars
    as 2-D arrays, this means that something you expect to be a scalar will in
    fact be a 1-by-1 np.ndarray. You can change this by passing ``squeeze=True``
    to the constructor:

    .. code-block:: python

        with Hdf5Matfile(filename, squeeze=True) as file:
            data = file.load_file()

    Supported data types
    --------------------

    Data type support is pretty limited; this isn't a terribly fancy class.
    Supported MATLAB data types, and the Python objects or NumPy dtypes they map
    to:

    ===============  =============  =============
      MATLAB type    Python object   NumPy dtype
    ===============  =============  =============
    cell             np.ndarray     object
    char             str            n/a
    double           np.ndarray     np.double
    int8             np.ndarray     np.byte
    int16            np.ndarray     np.short
    int32            np.ndarray     np.intc
    int64            np.ndarray     np.int_
    logical          np.ndarray     bool8
    single           np.ndarray     single
    struct (scalar)  dict           n/a
    struct (array)   np.ndarray     object (dict)
    uint8            np.ndarray     np.ubyte
    uint16           np.ndarray     np.ushort
    uint32           np.ndarray     np.uintc
    uint64           np.ndarray     np.uint
    ===============  =============  =============

    .. |load_file| :method:`Hdf5Matfile.load_file`
    .. |load_variable| :method:`Hdf5Matfile.load_variable`
    """
    def __init__(self, filename, squeeze=False):
        """Open a MATLAB v7.3 *.mat file.

        Parameters
        ----------
        filename : path_like
            Path to the *.mat file.

        squeeze : bool, optional
            If True, squeeze loaded arrays (remove dimensions with size 1). If
            the array only has one element, it is extracted from the array using
            ``array.item()``, which returns a Python scalar. (default: False)
        """
        self._h5file = h5py.File(filename, 'r')
        self._loader_dispatch = {
            b'cell': self._load_cell,
            b'char': self._load_char,
            b'double': self._load_numeric,
            b'int8': self._load_numeric,
            b'int16': self._load_numeric,
            b'int32': self._load_numeric,
            b'int64': self._load_numeric,
            b'logical': lambda x: self._load_numeric(x, dtype=np.bool8),
            b'single': self._load_numeric,
            b'struct': self._load_struct,
            b'uint8': self._load_numeric,
            b'uint16': self._load_numeric,
            b'uint32': self._load_numeric,
            b'uint64': self._load_numeric,
        }

        def _squeeze(a):
            if a.size == 1:
                squeezed = a.item()
            else:
                squeezed = np.squeeze(a)
            return squeezed

        self._squeeze = _squeeze if squeeze else lambda x: x

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        self.close()

    def close(self):
        self._h5file.close()

    def load_file(self):
        """Load the entire file.

        Returns
        -------
        dict
            Dict whose keys are the top-level variable names.
        """
        d = {}
        for key, value in self._h5file.items():
            if key == '#refs#':
                continue
            d[key] = self._load_item(value)
        return d

    def load_variable(self, varname):
        """Load a specific variable from the file.

        Parameters
        ----------
        varname : str
            The name of the variable to load.

        Returns
        -------

        """
        if varname == '#refs#':
            raise KeyError('#refs# is not a MATLAB variable.')

        return self._load_item(self._h5file[varname])

    def _load_item(self, item):
        matlab_class = item.attrs['MATLAB_class']
        loader = self._loader_dispatch[matlab_class]
        return loader(item)

    def _load_char(self, char):
        return decodechar(char[()])

    def _load_cell(self, cell):
        cell = cell[()]
        a = np.empty(cell.shape, dtype='O')
        for i, ref in enumerate(cell.flat):
            a.flat[i] = self._load_item(self._h5file[ref])
        return a

    def _load_numeric(self, numeric, dtype=None):
        return self._squeeze(numeric[()].astype(dtype))

    def _load_struct(self, struct):
        if isstructarray(struct):
            return self._load_struct_array(struct)
        else:
            return self._load_scalar_struct(struct)

    def _load_scalar_struct(self, struct):
        d = {}
        for fieldname, item in struct.items():
            d[fieldname] = self._load_item(item)
        return d

    def _load_struct_array(self, struct):
        # Get an item from the struct to figure out how big it is, then stick it
        # back in the dict. I have no idea if there's a cleaner way, I just need
        # to inspect a single item *before* looping!
        pointers = dict(struct)
        fieldname, refarray = pointers.popitem()
        pointers[fieldname] = refarray

        # Initialize array of dict
        a = np.empty(refarray.shape, dtype='O')
        for i, _ in enumerate(a.flat):
            a[i] = dict()

        for fieldname, refarray in pointers.items():
            for i, ref in enumerate(refarray[()].flat):
                a.flat[i][fieldname] = self._load_item(self._h5file[ref])

        return self._squeeze(a)


def load_hdf5mat(filename, variables=None, squeeze=False):
    """Load a MATLAB v7.3 *.mat file. See |Hdf5Matfile| for limitations and
    supported data types.

    Parameters
    ----------
    filename : path_like
        Path to the *.mat file.

    variables : str, list[str], optional
        Variable name(s) to load. Default is None, which loads the entire file.

    squeeze : bool, optional
        If True, squeeze arrays and pop scalars. (default: False)


    .. |Hdf5Matfile| :class:`Hdf5Matfile`
    """
    with Hdf5Matfile(filename, squeeze=squeeze) as file:
        if variables is None:
            data = file.load_file()
        elif isinstance(variables, str):
            data = file.load_variable(variables)
        elif len(variables) == 1:
            data = file.load_variable(variables[0])
        else:
            data = {var: file.load_variable(var) for var in variables}

    return data


def isstructarray(struct):
    """Determine whether the given MATLAB struct is scalar or not."""
    for field in struct.values():
        # MATLAB represents scalar structs and struct arrays differently within
        # HDF5. Scalar structs are ordinary groups with named datasets and/or
        # subgroups. Struct arrays, however, are represented by a group with
        # arrays of references. The arrays all have the same size (that of the
        # struct array itself), and are grouped by field name.
        #
        # If the fields in the struct are *not* assigned a MATLAB_class, then
        # they're not actual objects. This is what differentiates a struct
        # array from a cell array -- the cell array is assigned a MATLAB_class,
        # and the fields of a struct array are not.
        try:
            matlab_class = field.attrs['MATLAB_class']
        except KeyError:
            isarray = True
            break
    else:
        # Executes if break doesn't fire
        isarray = False

    return isarray


def decodechar(char, encoding='utf-8'):
    """Decode a MATLAB char array to str."""
    # MATLAB likes to pad its char arrays with null bytes for some reason.
    return char.tobytes().decode(encoding).replace('\x00', '')
