# moza

A mosaic generator.

# Installation and usage

[Pipenv](https://docs.pipenv.org/) is used for dependency management. After
obtaining the repo, install the dependencies, open an environment shell, and
run `moza.py`:

```bash
$ pipenv install
$ pipenv shell
$ python moza.py SOURCE TILES TARGET
```

The `TILES` argument should be a directory containing images to be used as
tiles; they should be square and of the same size, and ideally small (as they
are kept in memory to produce the mosaic).
