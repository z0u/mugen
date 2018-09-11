Feature: Read MIDI files

@wip
Scenario: Read a MIDI file into a Numpy array
  Given a C-major scale read from a MIDI file
  And a C-major scale read from a raster file
  When the MIDI data is converted into an array
  Then the array matches the known raster data
