import json
import shutil
from functools import reduce
from typing import Dict, List, Tuple

import h5py
import numpy as np

# '?'	boolean
# 'b'	(signed) byte
# 'B'	unsigned byte
# 'i'	(signed) integer
# 'u'	unsigned integer
# 'f'	floating-point
# 'c'	complex-floating point
# 'm'	timedelta
# 'M'	datetime
# 'O'	(Python) objects
# 'S', 'a'	zero-terminated bytes (not recommended)
# 'U'	Unicode string
# 'V'	raw data (void)
from h5py import Dataset, Group


class NestedDict:
    def __init__(self, *args, **kwargs):
        self.dict = dict(*args, **kwargs)

    def __getitem__(self, keys):
        # Allows getting top-level branch when a single key was provided
        if not isinstance(keys, list):
            keys = [keys]

        branch = self.dict
        for key in keys:
            branch = branch[key]

        # If we return a branch, and not a leaf value, we wrap it into a NestedDict
        return NestedDict(branch) if isinstance(branch, dict) else branch

    def __setitem__(self, keys, value):
        # Allows setting top-level item when a single key was provided
        if not isinstance(keys, list):
            keys = [keys]

        branch = self.dict
        for key in keys[:-1]:
            if key not in branch:
                branch[key] = {}
            branch = branch[key]
        branch[keys[-1]] = value


def axf2dict(axf: h5py.File) -> Dict:
    megaobject = NestedDict()

    def itemsvisitor(k, v):
        print(k, type(v), v)
        keys = k.split("/")
        if type(v) is Dataset:
            d: Dataset = v
            print(d.shape)
            if d.shape == ():
                print(d.dtype, d.value)
                if str(d.dtype).startswith('|S'):
                    megaobject[keys] = str(d.value)
                elif d.dtype == np.float32:
                    megaobject[keys] = float(d.value)
                elif d.dtype == np.int32:
                    megaobject[keys] = int(d.value)
                elif d.dtype == np.uint8:
                    megaobject[keys] = int(d.value)
                elif d.dtype == np.uint32:
                    megaobject[keys] = int(d.value)
                else:
                    megaobject[keys] = d.value
            elif len(d.shape) == 1:
                print(d.dtype, type(d[0]), np.array(d))
                if isinstance(d[0], bytes):
                    megaobject[keys] = list(map(lambda x: str(x), d))
                elif isinstance(d[0], np.float32):
                    megaobject[keys] = list(map(lambda x: float(x), d))
                elif isinstance(d[0], np.int32):
                    megaobject[keys] = list(map(lambda x: int(x), d))
                else:
                    megaobject[keys] = list(d)
                json.dumps(megaobject[keys])
            else:
                x = reduce(lambda a, b: a * b, d.shape)
                if x < 100:
                    ll = np.array(d).tolist()
                    megaobject[keys] = ll
                else:
                    megaobject[keys] = f'BIG{d.shape}'
                json.dumps(megaobject[keys])
        elif type(v) is Group:
            print('group')
            megaobject[keys] = {}
        else:
            raise Exception(f"unhandled type {type(v)} {k}")

    axf.visititems(itemsvisitor)
    return megaobject.dict


def axf2json(axf: str, jsonpath: str):
    f = h5py.File(axf, 'r')
    d = axf2dict(f)
    with open(jsonpath, 'w') as jf:
        json.dump(d, jf, indent=4)


ArrayMM = Tuple[np.ndarray, Tuple[float, float]]


class AXF:
    def __init__(self, h5f: h5py.File):
        self.h5f = h5f

    def materialids(self) -> List[str]:
        return list(self.h5f['/com.xrite.Materials'].keys())

    def dataset_key(self, path: str, materialid: str = None) -> str:
        if materialid is None:
            materialid = self.materialids()[0]
        return f'/com.xrite.Materials/{materialid}/{path}'

    def dataset(self, path: str, materialid: str = None) -> Dataset:
        """
            path 'com.xrite.Resources/DiffuseModel/Color/Data'

            materialid 'bd_fg_baron_baron_01-feather_source_2'
        """
        return self.h5f[self.dataset_key(path, materialid)]

    def array_mm(self, rootpath) -> ArrayMM:
        array = np.array(self.dataset(f'{rootpath}/Data'))
        hmm = self.dataset(f'{rootpath}/HeightMM')[()]
        wmm = self.dataset(f'{rootpath}/WidthMM')[()]
        return array, (hmm, wmm)

    def array_mm_setter(self, rootpath, value: ArrayMM):
        array, sz = value
        dk = self.dataset_key(f'{rootpath}/Data')
        hk = self.dataset_key(f'{rootpath}/HeightMM')
        wk = self.dataset_key(f'{rootpath}/WidthMM')
        del self.h5f[dk]
        self.h5f.create_dataset(dk, data=array)
        self.h5f[hk][()] = sz[0]
        self.h5f[wk][()] = sz[1]

    @property
    def diffuse_color(self) -> ArrayMM:
        return self.array_mm('com.xrite.Resources/DiffuseModel/Color')

    @diffuse_color.setter
    def diffuse_color(self, value: ArrayMM):
        self.array_mm_setter('com.xrite.Resources/DiffuseModel/Color', value)

    @property
    def diffuse_normal(self) -> ArrayMM:
        return self.array_mm('com.xrite.Resources/DiffuseModel/Normal')

    @diffuse_normal.setter
    def diffuse_normal(self, value: ArrayMM):
        self.array_mm_setter('com.xrite.Resources/DiffuseModel/Normal', value)

    @property
    def specular_color(self) -> ArrayMM:
        return self.array_mm('com.xrite.Resources/SpecularModel/Color')

    @specular_color.setter
    def specular_color(self, value: ArrayMM):
        self.array_mm_setter('com.xrite.Resources/SpecularModel/Color', value)

    @property
    def specular_lobes(self) -> ArrayMM:
        return self.array_mm('com.xrite.Resources/SpecularModel/Lobes')

    @specular_lobes.setter
    def specular_lobes(self, value: ArrayMM):
        self.array_mm_setter('com.xrite.Resources/SpecularModel/Lobes', value)

    @property
    def transparency_alpha(self) -> ArrayMM:
        return self.array_mm('com.xrite.Resources/TransparencyFilter/Alpha')

    @transparency_alpha.setter
    def transparency_alpha(self, value: ArrayMM):
        self.array_mm_setter('com.xrite.Resources/TransparencyFilter/Alpha', value)


def tile2x2(a: np.ndarray) -> np.ndarray:
    h = np.hstack((a, a))
    tiled = np.vstack((h, h))
    return tiled


if __name__ == '__main__':
    # axf2json('data/AxF/bd_fg_baron_baron_01-feather_source_2.axf', 'axf.json')
    axffile = 'data/AxF/bd_fg_baron_baron_01-feather_source_2.axf'
    h5read = h5py.File(axffile, 'r')
    axfread = AXF(h5read)
    dcolor, szdc = axfread.diffuse_color
    print(szdc, dcolor.shape)
    dnormal, szdn = axfread.diffuse_normal
    scolor, szsc = axfread.specular_color
    slobes, szsl = axfread.specular_lobes
    talpha, szta = axfread.transparency_alpha
    assert szdc == szdn and szdn == szsc and szsc == szsl and szsl == szta, f'{szdc} {szdn} {szsc} {szsl} {szta}'

    sztiled = (szdc[0] * 2, szdc[1] * 2)

    dstaxf = 'tiled2x2.axf'
    shutil.copyfile(axffile, dstaxf)
    h5write = h5py.File(dstaxf, 'r+')
    axfwrite = AXF(h5write)
    axfwrite.diffuse_color = tile2x2(dcolor), sztiled
    axfwrite.diffuse_normal = tile2x2(dnormal), sztiled
    axfwrite.specular_color = tile2x2(scolor), sztiled
    axfwrite.specular_lobes = tile2x2(slobes), sztiled
    axfwrite.transparency_alpha = tile2x2(talpha), sztiled
    h5write.close()
    axf2json(dstaxf, f'{dstaxf}.json')
