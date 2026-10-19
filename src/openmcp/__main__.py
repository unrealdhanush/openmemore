"""`python -m openmcp`: the Windows service runs it under pythonw, which has no console."""

from openmcp.cli import app

app(prog_name="openmcp")
