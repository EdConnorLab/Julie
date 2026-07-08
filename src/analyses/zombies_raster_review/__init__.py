"""
zombies_raster_review — side investigation: readable rasters for reviewing
ANOVA-passed units against the Zombies stimulus group.

Reads curated unit lists (Excel → MixedManualSpikeSource, CSV →
SISortedSpikeSource) and renders a raster + PSTH per unit, ranked by stimulus
monkey dominance, with the significant response window highlighted.

Does not modify ``analyses/plot_rasters.py`` — it reuses the same data formats
and rank logic but ships its own, cleaner rendering. See ``README.md``.
"""
