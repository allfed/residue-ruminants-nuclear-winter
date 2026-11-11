"""Record input/code checksums and exact package versions for a result set."""
import hashlib
import importlib.metadata
import json
import platform
import re
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from src import inputs as I


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def project_packages():
    """Record installed project dependencies, excluding unrelated host packages."""
    pending = re.findall(r"^\s*-\s*([A-Za-z0-9][A-Za-z0-9_.-]*)[=<>!]", (I.ROOT/'environment.yml').read_text(), re.M)
    packages = {}
    while pending:
        name = canonicalize_name(pending.pop())
        if name == 'python' or name in packages:
            continue
        dist = importlib.metadata.distribution(name)
        packages[name] = dist.version
        for declaration in dist.requires or []:
            requirement = Requirement(declaration)
            if requirement.marker is None or requirement.marker.evaluate({'extra': ''}):
                pending.append(requirement.name)
    return packages


def main():
    files={}
    roots=[I.DATA,I.ROOT/'src/model',I.ROOT/'src/runs',I.ROOT/'src/reach',
           I.ROOT/'src/figures']
    for root in roots:
        for path in sorted(root.rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts:
                files[str(path.relative_to(I.ROOT))]=digest(path)
    for path in (I.ROOT/'src/inputs.py',I.ROOT/'environment.yml'):
        files[str(path.relative_to(I.ROOT))]=digest(path)
    outputs={str(p.relative_to(I.ROOT)):digest(p) for p in sorted(I.RESULTS.iterdir()) if p.suffix in ('.csv','.npz')}
    packages=project_packages()
    sampling={}
    for variant, tag in [('default',''),('treated','_treated')]:
        with np.load(I.RESULTS/f'mc_draws{tag}.npz', allow_pickle=False) as archive:
            sizes={archive[key].shape[0] for key in archive.files}
            if len(sizes) != 1:
                raise ValueError(f'Inconsistent draw counts in {variant} archive')
            count=sizes.pop()
        sampling[variant]={'draws':count,'seeds':[1000,999+count]}
    sampling.update(ration_search='exact LP per herd (HiGHS) over the diet M/D: coarse scan every 1 MJ/kg DM, then bisection to 0.0001 MJ/kg DM')
    manifest=dict(created_utc=datetime.now(timezone.utc).isoformat(),
        python=platform.python_version(),platform=platform.platform(),files=files,results=outputs,
        installed_packages=packages,
        dependency_note='Installed direct and transitive project dependencies for this platform; unrelated host packages are excluded.',
        monte_carlo=sampling,
        population='UN WPP 2024 Rev.1, 1 July 2020, all 169 modeled countries')
    target=I.RESULTS
    (target/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    print(len(files),'input and code files;',len(outputs),'result files')


if __name__=='__main__': main()
