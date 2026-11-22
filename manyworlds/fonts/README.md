# Caption fonts

Fonts burned into videos by `manyworlds/captions.py` (styles in `policy/default.yaml`, `burned_captions.styles`).
ffmpeg's `ass` filter reads them from this directory (`fontsdir`), so nothing depends on system fonts.

All are static TTFs from [google/fonts](https://github.com/google/fonts) (`ofl/<family>/`, branch `main`,
downloaded 2026-11-22), licensed under the **SIL Open Font License 1.1**: free to use, embed and bundle,
including in commercial work and burned-in video; the fonts themselves may not be sold on their own. Each
family's licence text sits next to it.

| File | Name in the ASS style | Family | Licence |
|---|---|---|---|
| `Poppins-SemiBold.ttf` | `Poppins SemiBold` | Poppins (Indian Type Foundry) | OFL 1.1, `Poppins-OFL.txt` |
| `Poppins-Bold.ttf` | `Poppins` + bold | Poppins | OFL 1.1, `Poppins-OFL.txt` |
| `Poppins-ExtraBold.ttf` | `Poppins ExtraBold` | Poppins | OFL 1.1, `Poppins-OFL.txt` |
| `Anton-Regular.ttf` | `Anton` | Anton (Vernon Adams) | OFL 1.1, `Anton-OFL.txt` |
| `DMSerifDisplay-Regular.ttf` | `DM Serif Display` | DM Serif Display (Colophon, Google) | OFL 1.1, `DMSerifDisplay-OFL.txt` |
| `DMSerifDisplay-Italic.ttf` | `DM Serif Display` + italic | DM Serif Display | OFL 1.1, `DMSerifDisplay-OFL.txt` |

The ASS name must match the font's family name (name ID 1) or full name (ID 4); a style also names its file
(`file`), which is used to size text exactly and by the fallback renderer when an ffmpeg has no libass.
To add a font: put the static TTF and its OFL.txt here, add a row above, and point a style at it.
