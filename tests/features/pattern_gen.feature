Feature: Extend simple sequences

Scenario: Extend a progression
  Given a progression of 10 pitches
  And the model has been trained for 20 epochs
  When the next sample is generated
  Then the extension matches the initial sequence
