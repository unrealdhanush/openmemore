"""`python -m openmemore`: the Windows service runs it under pythonw, which has no console."""

from openmemore.cli import app

app(prog_name="openmemore")
