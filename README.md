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
