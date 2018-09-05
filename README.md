Mugen is a ML-based parameterised music generator.

First [set up](#Setup) the environment (see below). Then train with:

```
python -m mugen train
```

## Testing

First [set up](#Setup) the environment (see below). Then test with:

```
py.test --pylama
```

## Setup

After a fresh checkout:
```
python3 -m easy-install virtualenv
python3 -m virtualenv pyenv
source pyenv/bin/activate
pip install -r requirements.txt
```

When starting work on this project:
```
source pyenv/bin/activate
```

When requirements change:
```
pip install -r requirements.txt
```

When stopping work:
```
deactivate
```
