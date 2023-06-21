# icenet

![GitHub issues](https://img.shields.io/github/issues/icenet-ai/icenet?style=plastic)
![GitHub closed issues](https://img.shields.io/github/issues-closed/icenet-ai/icenet?style=plastic)
![GitHub](https://img.shields.io/github/license/icenet-ai/icenet)
![GitHub forks](https://img.shields.io/github/forks/icenet-ai/icenet?style=social)
![GitHub forks](https://img.shields.io/github/stars/icenet-ai/icenet?style=social)

This is the core python library for the IceNet sea-ice forecasting system. 

This `README` will be worked on more, but there's plenty of information around 
in the [`icenet-ai`][3] organisations repositories, which demonstrate usage of 
this library.

## Table of contents

* [Overview](#overview)
* [Installation](#installation)
* [Implementation](#implementation)
* [Pipeline](#pipeline)
* [Contributing to IceNet](#contributing-to-icenet)
* [Credits](#credits)
* [License](#license)

## Installation

We're still working on clear dependency management using pip, Tensorflow is best through pip but obviously you need NVIDIA dependencies for GPU based training. If you're having trouble with system dependencies some advice about environment setup is given by the examples [under the pipeline repository][1].

```
pip install icenet
```

### Development installation

Please refer to [the contribution guidelines for more information.](CONTRIBUTING.rst)

## Implementation

When installed, the library will provide a series of CLI commands. Please use 
the `--help` switch for more initial information, or the documentation. 

### Documentation

The `docs/` directory has a `Makefile` that builds sphinx docs easily enough, 
once the requirements in that directory are installed. 

## Usage Pipeline / Examples

Please refer to [the icenet-pipeline repository][1] or [the icenet-notebook
repository][2] for examples of how to use this library.

## Contributing to IceNet

Please refer to [the contribution guidelines for more information.](CONTRIBUTING.rst)

## Credits

<a href="https://github.com/icenet-ai/icenet/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=icenet-ai/icenet" />
</a>

## License

This is licensed using the [MIT License](LICENSE)

[1]: https://github.com/icenet-ai/icenet-pipeline
[2]: https://github.com/icenet-ai/icenet-notebooks
[3]: https://github.com/icenet-ai
