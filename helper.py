import numpy as np


class Obj:
    vs = None
    vts = None
    fvs = None
    fvts = None

    vns = None

    headers = []
    usemtls = []


def read_obj(obj_path, only_vs=False, tri=False):
    objfile = open(obj_path, encoding="utf8").read().strip().split("\n")
    if only_vs:
        obj = Obj()
        obj.vs = np.array([[float(j) for j in i[2:].strip().split()] for i in filter(lambda x:x.startswith("v "), objfile)], np.float32)
        return obj

    vs = []
    vts = []
    fvs = []
    fvts = []

    headers = []
    usemtls = []

    for line in objfile:
        if line.startswith("v "):
            vs.append(list(map(float, line[2:].strip().split())))
        elif line.startswith("vt "):
            vts.append(list(map(float, line[2:].strip().split())))
        elif line.startswith("f "):
            fv = []
            fvt = []
            for i in line[2:].strip().split():
                component = i.split("/")[:2]
                vth = int(component[0]) - 1
                fv.append(vth)

                if len(component) > 1 and component[1] != "":
                    vtth = int(component[1]) - 1
                    fvt.append(vtth)

            if tri:
                for i in range(2, len(fv)):
                    fvs.append(fv[:1] + fv[i - 1:i + 1])
                    fvts.append(fvt[:1] + fvt[i - 1:i + 1])

            else:
                fvs.append(fv)
                fvts.append(fvt)

        elif line.startswith("usemtl "):
            usemtls.append((len(fvs), line))

        elif line.startswith("mtllib "):
            headers.append(line)

    obj = Obj()
    obj.vs = np.array(vs, np.float32)
    obj.vts = np.array(vts, np.float32)

    obj.headers = headers
    obj.usemtls = usemtls

    try:
        obj.fvs = np.array(fvs, int)
        obj.fvts = np.array(fvts, int)
    except ValueError:
        obj.fvs = fvs
        obj.fvts = fvts

    return obj


def write_obj(template_path, v: "Nx3", path, only_v=False):
    assert v.shape[1] == 3

    last_template = getattr(write_obj, "last_template", (None, None))

    if last_template[0] != template_path:
        obj = open(template_path).read().strip().split("\n")
        last_template = (template_path, obj)
        setattr(write_obj, "last_template", last_template)
    obj = last_template[1]
    vs = []
    for i in v:
        vs.append(f"v {i[0]:.6f} {i[1]:.6f} {i[2]:.6f}")
    if only_v:
        obj = "\n".join(vs)
    else:
        nvs = list(filter(lambda x: not x.startswith("v "), obj))
        obj = "\n".join(vs + nvs)

    open(path, "w").write(obj)
