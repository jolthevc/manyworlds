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
| `TikTokSans-SemiBold.ttf` | `TikTok Sans SemiBold` | TikTok Sans (Grilli Type, Contrast Foundry, Type Network for TikTok) | OFL 1.1, `TikTokSans-OFL.txt` |
| `TikTokSans-Bold.ttf` | `TikTok Sans` + bold | TikTok Sans | OFL 1.1, `TikTokSans-OFL.txt` |

TikTok Sans (on-screen labels, `manyworlds/overlays.py`, and the `tiktok` caption style) is published in google/fonts
only as a variable font (`ofl/tiktoksans/TikTokSans[opsz,slnt,wdth,wght].ttf`), which libass would draw at its
default instance. The two files here are Google Fonts' own static instances of it (weights 600 and 700, served by
fonts.gstatic.com for the css2 API, downloaded 2026-11-23); the licence is google/fonts' `ofl/tiktoksans/OFL.txt`
(no Reserved Font Name). No bundled font has emoji glyphs: on-screen text drops characters its font cannot draw
(policy `missing_glyphs: strip`).

The ASS name must match the font's family name (name ID 1) or full name (ID 4); a style also names its file
(`file`), which is used to size text exactly and by the fallback renderer when an ffmpeg has no libass.
To add a font: put the static TTF and its OFL.txt here, add a row above, and point a style at it.
