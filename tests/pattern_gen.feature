Feature: Extend simple sequences

Scenario: Extend a short repeating sequence
  Given a short sawtooth wave of 5 steps
  And the model has been trained on that sequence
  When the next period is generated
  Then the extension matches the initial sequence
