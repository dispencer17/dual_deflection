UnitConverters = {
  "GPa": lambda gpa: (gpa) * 1_000_000_000 if str(gpa).replace(".", "", 1).isdigit() else 0,
  "nm": lambda nm: (nm) / 1_000_000_000 if str(nm).replace(".", "", 1).isdigit() else 0,
  "microns": lambda microns: (microns) / 1_000_000 if str(microns).replace(".", "", 1).isdigit() else 0, 
  "default": lambda n: n
}