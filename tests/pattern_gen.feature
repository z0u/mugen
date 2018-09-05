Feature: Extend simple sequences

Scenario: Extend a sawtooth wave
  Given a sawtooth wave of 5 steps
  And the model has been trained on that sequence for 10 epochs
  When the next note is generated
  Then the extension matches the initial sequence
