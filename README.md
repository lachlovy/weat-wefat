This repository contains implementations of WEAT and WEFAT as described in the paper:

> Caliskan, A., Bryson, J. J., & Narayanan, A. (2017). Semantics derived automatically from language corpora contain human-like biases. Science, 356(6334), 183-186.

These methods allow researchers to detect and measure biases and stereotypes in word embeddings.

## Installation

```bash
pip install -r requirements.txt
```

### Getting the code

```bash
git clone https://github.com/lachlovy/weat-wefat.git
cd weat-wefat
```

## Usage

### Download word embeddings

Now only support GloVe Word Embedding.

```bash
python download_glove_data.py --output_dir "data" --model "glove.6B"
```

If you changed `output_dir`, you should change the directory in `weat_wefat.py` accordingly.


### Command Line interface

Run the weat and wefat test using:

```bash
python weat_wefat.py
```

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request