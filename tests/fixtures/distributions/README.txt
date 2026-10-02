These tiny wheels and source distributions were built with setuptools 84.0.0
through setuptools.build_meta.build_sdist/build_wheel, using the project and
package contents declared in tests/test_dist_prereleases.py. Each source project
also has README.md, LICENSE, a setuptools>=77.0.3 build backend, src package
discovery, and py.typed package data.

The filenames and generated Version metadata preserve real backend behavior:
stable, alpha, beta, release candidate, development, local build metadata, and
a release candidate with local metadata. The bundled source pyproject.toml keeps
its original SemVer spelling even though the artifact paths and core metadata
use normalized PEP 440 versions.

The fixtures deliberately require no setuptools dependency when running the
tests or verifier. SOURCE_DATE_EPOCH=1700000000 was set
when producing the wheels.
