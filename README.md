Mugen is a ML-based parameterised music generator.


## Testing

First [set up](#setup) the environment (see below). Then test with:

```
pytest
```

Features under development are marked with `@wip` and are skipped from testing by default. To run those tests, run:

```
pytest -k wip
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

## Playing MIDI on Mac OS

1. Open the Audio MIDI Setup app
2. Choose _Window > Show MIDI Studio_
3. Double-click on _IAC Driver_
4. Check _Device is online_
5. Start Garage Band and add a software MIDI track
6. Play a short test:

  ```
  python -m mugen test-midi
  ```
